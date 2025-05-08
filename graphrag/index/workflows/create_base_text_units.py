# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition with source tracking."""

import json
import logging
from typing import Any, cast

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

log = logging.getLogger(__name__)


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to transform base text_units with source tracking."""
    documents = await load_table_from_storage("documents", context.storage)

    chunks = config.chunks

    # Get source tracking configuration options if they exist
    track_sources = getattr(chunks, 'track_sources', False)
    source_path_column = getattr(chunks, 'source_path_column', 'source_path')
    
    log.info(f"Creating text units with source tracking: {track_sources}, source path column: {source_path_column}")

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
        track_sources=track_sources,
        source_path_column=source_path_column,
    )

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
    track_sources: bool = False,  # New parameter for source tracking
    source_path_column: str = "source_path",  # New parameter for source path column
) -> pd.DataFrame:
    """All the steps to transform base text_units with source tracking."""
    # Check if we can do source tracking
    if track_sources and source_path_column in documents.columns:
        log.info(f"Source tracking enabled and source path column '{source_path_column}' found in documents")
    elif track_sources:
        log.warning(f"Source tracking enabled but source path column '{source_path_column}' not found in documents")
        log.warning(f"Available columns: {list(documents.columns)}")
        track_sources = False
    
    sort = documents.sort_values(by=["id"], ascending=[True])

    # Prepare text_with_ids tuples
    if track_sources and source_path_column in sort.columns:
        # Include source paths in the tuples
        log.info("Including source paths in text_with_ids tuples")
        sort["text_with_ids"] = list(
            zip(*[sort[col] for col in ["id", "text", source_path_column]], strict=True)
        )
    else:
        # Original behavior without source paths
        sort["text_with_ids"] = list(
            zip(*[sort[col] for col in ["id", "text"]], strict=True)
        )

    callbacks.progress(Progress(percent=0))

    agg_dict = {"text_with_ids": list}
    if "metadata" in documents:
        agg_dict["metadata"] = "first"  # type: ignore

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

        # Check for source paths if tracking is enabled
        file_path_column_to_use = None
        df_row = pd.DataFrame([row]).reset_index(drop=True)
        
        # Extract source paths if available and source tracking is enabled
        if track_sources:
            has_source_paths = False
            texts_with_paths = []
            
            for item in row["texts"]:
                if isinstance(item, tuple) and len(item) >= 3:
                    # Format is (id, text, source_path)
                    has_source_paths = True
                    texts_with_paths.append((item[0], item[1], item[2]))
                elif isinstance(item, tuple):
                    # Format is (id, text) without source path
                    texts_with_paths.append((item[0], item[1], "unknown"))
                else:
                    # Format is just text
                    texts_with_paths.append((None, item, "unknown"))
            
            if has_source_paths:
                # Create a temporary column with source paths
                df_row["source_paths"] = [paths[2] for paths in texts_with_paths]
                file_path_column_to_use = "source_paths"
                # Update the texts to just have id and text
                df_row["texts"] = [(item[0], item[1]) for item in texts_with_paths]
                log.info(f"Created temporary source_paths column with {len(texts_with_paths)} paths")

        # Call chunk_text with or without source tracking
        chunked = chunk_text(
            df_row,
            column="texts",
            size=size - metadata_tokens,
            overlap=overlap,
            encoding_model=encoding_model,
            strategy=strategy,
            callbacks=callbacks,
            file_path_column=file_path_column_to_use,  # Pass file paths if available
        )[0]

        # Handle metadata prepending for both regular and source-tracked chunks
        if prepend_metadata:
            for index, chunk in enumerate(chunked):
                if isinstance(chunk, str):
                    chunked[index] = metadata_str + chunk
                elif isinstance(chunk, tuple) and len(chunk) == 3:
                    # Standard tuple format (doc_ids, text, n_tokens)
                    chunked[index] = (
                        chunk[0], metadata_str + chunk[1], chunk[2]
                    )
                elif isinstance(chunk, tuple) and len(chunk) >= 4:
                    # Extended tuple format with source location (doc_ids, text, n_tokens, source_location)
                    chunked[index] = (
                        chunk[0], metadata_str + chunk[1], chunk[2], chunk[3]
                    )
                else:
                    # Unknown format, leave as is
                    pass

        row["chunks"] = chunked
        return row

    aggregated = aggregated.apply(lambda row: chunker(row), axis=1)

    aggregated = cast("pd.DataFrame", aggregated[[*group_by_columns, "chunks"]])
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
    
    # Check if chunks contain source location information
    first_chunk = aggregated["chunk"].iloc[0] if not aggregated.empty else None
    has_source_location = (
        isinstance(first_chunk, tuple) and 
        len(first_chunk) >= 4 and 
        first_chunk[3] is not None
    )
    
    if has_source_location:
        # Extract with source location
        log.info("Found source location information in chunks, extracting it")
        aggregated[["document_ids", "chunk", "n_tokens", "source_location"]] = pd.DataFrame(
            aggregated["chunk"].tolist(), index=aggregated.index
        )
    else:
        # Extract without source location (original behavior)
        log.info("No source location information found in chunks")
        aggregated[["document_ids", "chunk", "n_tokens"]] = pd.DataFrame(
            aggregated["chunk"].tolist(), index=aggregated.index
        )

    # rename for downstream consumption
    aggregated.rename(columns={"chunk": "text"}, inplace=True)
    
    # If we have source location information, add it to attributes
    if has_source_location and "source_location" in aggregated.columns:
        log.info("Adding source location to attributes column")
        # Create attributes column if it doesn't exist
        if "attributes" not in aggregated.columns:
            aggregated["attributes"] = None
            
        # Add source location to attributes
        for idx, row in aggregated.iterrows():
            if row["source_location"] is not None:
                source_loc = row["source_location"]
                attrs = {}
                if hasattr(source_loc, "file_path"):
                    attrs["source_file"] = source_loc.file_path
                if hasattr(source_loc, "start_line"):
                    attrs["source_line_start"] = source_loc.start_line
                if hasattr(source_loc, "end_line"):
                    attrs["source_line_end"] = source_loc.end_line
                if hasattr(source_loc, "start_char"):
                    attrs["source_char_start"] = source_loc.start_char
                if hasattr(source_loc, "end_char"):
                    attrs["source_char_end"] = source_loc.end_char
                
                if attrs:
                    aggregated.at[idx, "attributes"] = attrs
    
    # Determine which columns to include in the result
    result_columns = [*group_by_columns, "id", "document_ids", "text", "n_tokens"]
    if "attributes" in aggregated.columns:
        result_columns.append("attributes")
        
    result = aggregated[aggregated["text"].notna()].reset_index(drop=True)
    
    # Only select columns that actually exist in the DataFrame
    existing_columns = [col for col in result_columns if col in result.columns]
    
    return cast("pd.DataFrame", result[existing_columns])