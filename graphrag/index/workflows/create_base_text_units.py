# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import json
import logging
from typing import Any, Dict, List, cast

import pandas as pd

from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.chunking_config import ChunkStrategyType
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.operations.chunk_text.chunk_text import chunk_text
from graphrag.index.operations.chunk_text.strategies import get_encoding_fn
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.utils.hashing import gen_sha512_hash
from graphrag.logger.progress import Progress
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage

# Add logger
log = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to transform base text_units."""
    documents = await load_table_from_storage("documents", context.storage)

    chunks = config.chunks

    # Create the text units
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
    
    # Make sure all complex structures are JSON serialized
    for column in documents_copy.columns:
        if documents_copy[column].dtype == 'object':
            # Check for dict or list values that need serialization
            contains_complex = documents_copy[column].apply(
                lambda x: isinstance(x, (dict, list)) and not isinstance(x, str)
            ).any()
            
            if contains_complex:
                documents_copy[column] = documents_copy[column].apply(
                    lambda x: json.dumps(x) if isinstance(x, (dict, list)) and not isinstance(x, str) else x
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

    sort["text_with_ids"] = list(
        zip(*[sort[col] for col in ["id", "text"]], strict=True)
    )

    callbacks.progress(Progress(percent=0))

    agg_dict = {"text_with_ids": list}
    if "metadata" in documents:
        agg_dict["metadata"] = "first"  # type: ignore
    
    # Handle HTML attributes if they exist
    html_attributes_present = "html_attributes" in documents.columns
    if html_attributes_present:
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
                try:
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    # If it fails to parse, use as is
                    metadata = {"raw": metadata}
                    
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

        row["chunks"] = chunked
        return row

    aggregated = aggregated.apply(lambda row: chunker(row), axis=1)

    # Determine columns to keep
    columns_to_keep = [*group_by_columns, "chunks"]
    if html_attributes_present:
        columns_to_keep.append("html_attributes")
    
    aggregated = cast("pd.DataFrame", aggregated[columns_to_keep])
    aggregated = aggregated.explode("chunks")
    aggregated.rename(
        columns={
            "chunks": "chunk",
        },
        inplace=True,
    )
    aggregated["id"] = aggregated.apply(
        lambda row: gen_sha512_hash(row, ["chunk"]), axis=1
    )
    
    # Original approach for extracting columns
    chunk_df = pd.DataFrame(
        aggregated["chunk"].tolist(), 
        index=aggregated.index
    )
    
    # Make sure column names are correct and data is properly aligned
    if len(chunk_df.columns) >= 3:
        chunk_df.columns = ["document_ids", "text", "n_tokens"]
        for col in ["document_ids", "text", "n_tokens"]:
            aggregated[col] = chunk_df[col]
    else:
        # Handle case where chunk DataFrame doesn't have expected columns
        log.warning(f"Chunk DataFrame has unexpected columns: {chunk_df.columns}")
        # Set default values for missing columns
        if 0 < len(chunk_df.columns):
            aggregated["document_ids"] = chunk_df.iloc[:, 0]
        else:
            aggregated["document_ids"] = None
            
        if 1 < len(chunk_df.columns):
            aggregated["text"] = chunk_df.iloc[:, 1]
        else:
            aggregated["text"] = None
            
        if 2 < len(chunk_df.columns):
            aggregated["n_tokens"] = chunk_df.iloc[:, 2]
        else:
            aggregated["n_tokens"] = None
    
    # Process HTML attributes to extract essential info
    if html_attributes_present:
        # Create attributes column with HTML information
        aggregated["attributes"] = aggregated.apply(
            lambda row: extract_html_attributes(row), axis=1
        )
    else:
        # Ensure attributes column exists with empty dict value
        aggregated["attributes"] = [{}] * len(aggregated)
    
    # Ensure document_ids is always a list
    aggregated["document_ids"] = aggregated["document_ids"].apply(
        lambda x: [] if x is None else (x if isinstance(x, list) else [x])
    )
    
    # Convert attributes to JSON string for serialization
    aggregated["attributes"] = aggregated["attributes"].apply(json.dumps)
    
    # Filter out rows with no text and reset index
    result = cast(
        "pd.DataFrame", aggregated[aggregated["text"].notna()].reset_index(drop=True)
    )
    
    return result


def extract_html_attributes(row: pd.Series) -> Dict[str, Any]:
    """Extract HTML attributes from existing html_attributes."""
    # Default structure
    attributes = {
        "page": None,
        "paragraph": None,
        "char_position": {
            "start": None,
            "end": None
        }
    }
    
    # Get text content for matching
    chunk_text = row.get("text", "")
    if not chunk_text:
        return attributes
    
    # Get HTML attributes
    html_attrs = row.get("html_attributes")
    if not html_attrs:
        return attributes
    
    # Parse if JSON string
    if isinstance(html_attrs, str):
        try:
            html_attrs = json.loads(html_attrs)
        except json.JSONDecodeError:
            return attributes
    
    # Use the already parsed information
    if isinstance(html_attrs, dict):
        # Extract page info - use the last page as default
        if "pages" in html_attrs and isinstance(html_attrs["pages"], list) and html_attrs["pages"]:
            pages = html_attrs["pages"]
            last_page = pages[-1]
            if isinstance(last_page, dict):
                attributes["page"] = {
                    "id": last_page.get("page_id"),
                    "number": last_page.get("page_num")
                }
        
        # Extract paragraph that matches this chunk
        if "paragraphs" in html_attrs and isinstance(html_attrs["paragraphs"], list):
            paragraphs = html_attrs["paragraphs"]
            for para in paragraphs:
                if isinstance(para, dict):
                    para_text = para.get("text", "")
                    # Check if paragraph text overlaps with chunk text
                    if para_text and (para_text in chunk_text or chunk_text in para_text):
                        attributes["paragraph"] = {
                            "id": para.get("para_id"),
                            "number": para.get("para_num")
                        }
                        attributes["char_position"]["start"] = para.get("char_start")
                        attributes["char_position"]["end"] = para.get("char_end")
                        break
    
    return attributes