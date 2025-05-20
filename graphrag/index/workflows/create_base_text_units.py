# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import json
from typing import Any, Dict, List, Optional, cast

import pandas as pd

from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.chunking_config import ChunkStrategyType
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.operations.chunk_text.chunk_text import chunk_text
from graphrag.index.operations.chunk_text.strategies import get_encoding_fn
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.utils.hashing import gen_sha512_hash
from graphrag.logger.progress import Progress
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to transform base text_units."""
    documents = await load_table_from_storage("documents", context.storage)

    chunks = config.chunks

    output = create_base_text_units(
        documents,
        context.callbacks,
        chunks.group_by_columns,
        chunks.size,
        chunks.overlap,
        chunks.encoding_model,
        strategy=chunks.strategy,
        prepend_metadata=chunks.prepend_metadata,
        chunk_size_includes_metadata=chunks.chunk_size_includes_metadata,
    )

    # Store the original dataset for later use
    # Make a copy to ensure it's serializable
    documents_copy = documents.copy()
    
    # Convert any problematic columns to JSON strings
    if "html_attributes" in documents_copy.columns:
        documents_copy["html_attributes"] = documents_copy["html_attributes"].apply(
            lambda x: json.dumps(x) if x is not None else None
        )
    
    await write_table_to_storage(documents_copy, "dataset", context.storage)
    await write_table_to_storage(output, "text_units", context.storage)

    return WorkflowFunctionOutput(result=output)


def create_base_text_units(
    documents: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    group_by_columns: list[str],
    size: int,
    overlap: int,
    encoding_model: str,
    strategy: ChunkStrategyType,
    prepend_metadata: bool = False,
    chunk_size_includes_metadata: bool = False,
) -> pd.DataFrame:
    """All the steps to transform base text_units."""
    sort = documents.sort_values(by=["id"], ascending=[True])

    # Preserve HTML attributes if they exist
    if "html_attributes" not in sort.columns:
        sort["html_attributes"] = None
    
    sort["text_with_ids"] = list(
        zip(*[sort[col] for col in ["id", "text"]], strict=True)
    )

    callbacks.progress(Progress(percent=0))

    agg_dict = {"text_with_ids": list}
    if "metadata" in documents:
        agg_dict["metadata"] = "first"  # type: ignore
    
    # Add HTML attributes to aggregation
    if "html_attributes" in documents:
        agg_dict["html_attributes"] = "first"  # type: ignore

    aggregated = (
        (
            sort.groupby(group_by_columns, sort=False)
            if len(group_by_columns) > 0
            else sort.groupby(lambda _x: True)
        )
        .agg(agg_dict)
        .reset_index()
    )
    aggregated.rename(columns={"text_with_ids": "texts"}, inplace=True)

    def chunker(row: dict[str, Any]) -> Any:
        line_delimiter = ".\n"
        metadata_str = ""
        metadata_tokens = 0

        if prepend_metadata and "metadata" in row:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            if isinstance(metadata, dict):
                metadata_str = (
                    line_delimiter.join(f"{k}: {v}" for k, v in metadata.items())
                    + line_delimiter
                )

            if chunk_size_includes_metadata:
                encode, _ = get_encoding_fn(encoding_model)
                metadata_tokens = len(encode(metadata_str))
                if metadata_tokens >= size:
                    message = "Metadata tokens exceeds the maximum tokens per chunk. Please increase the tokens per chunk."
                    raise ValueError(message)

        chunked = chunk_text(
            pd.DataFrame([row]).reset_index(drop=True),
            column="texts",
            size=size - metadata_tokens,
            overlap=overlap,
            encoding_model=encoding_model,
            strategy=strategy,
            callbacks=callbacks,
        )[0]

        if prepend_metadata:
            for index, chunk in enumerate(chunked):
                if isinstance(chunk, str):
                    chunked[index] = metadata_str + chunk
                else:
                    chunked[index] = (
                        (chunk[0], metadata_str + chunk[1], chunk[2]) if chunk else None
                    )

        row["chunks"] = chunke