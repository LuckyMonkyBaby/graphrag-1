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
    log.info("Starting run_workflow process")
    documents = await load_table_from_storage("documents", context.storage)
    log.info(f"Loaded {len(documents)} documents from storage")

    chunks = config.chunks
    log.info(f"Using chunk configuration: size={chunks.size}, overlap={chunks.overlap}, strategy={chunks.strategy}")

    # Create the text units
    log.info("Creating base text units")
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
    log.info(f"Created {len(output)} text units")

    # Store the original dataset for later use
    # Make a copy to ensure it's serializable
    log.info("Preparing documents for storage")
    documents_copy = documents.copy()
    
    # Make sure all complex structures are JSON serialized
    for column in documents_copy.columns:
        if documents_copy[column].dtype == 'object':
            # Check for dict or list values that need serialization
            contains_complex = documents_copy[column].apply(
                lambda x: isinstance(x, (dict, list)) and not isinstance(x, str)
            ).any()
            
            if contains_complex:
                log.info(f"Serializing complex objects in column '{column}'")
                documents_copy[column] = documents_copy[column].apply(
                    lambda x: json.dumps(x) if isinstance(x, (dict, list)) and not isinstance(x, str) else x
                )
    
    # Normalize data types before storage to prevent Parquet errors
    log.info("Normalizing column data types before storage")
    
    # Function to ensure consistent types
    def normalize_column_types(df):
        for col in df.columns:
            # Skip index columns
            if col == 'index':
                continue
                
            # Handle columns with mixed types
            if df[col].dtype == 'object':
                # Check for mixed list/non-list
                contains_list = df[col].apply(lambda x: isinstance(x, list)).any()
                contains_nonlist = df[col].apply(lambda x: x is not None and not isinstance(x, list)).any()
                
                if contains_list and contains_nonlist:
                    log.warning(f"Column '{col}' has mixed list/non-list values - converting all to JSON strings")
                    df[col] = df[col].apply(lambda x: json.dumps(x) if x is not None else None)
                
                # Convert dict columns to strings
                contains_dict = df[col].apply(lambda x: isinstance(x, dict)).any()
                if contains_dict:
                    log.warning(f"Column '{col}' contains dictionaries - converting to JSON strings")
                    df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
        
        return df
    
    # Apply normalization to both dataframes
    documents_copy = normalize_column_types(documents_copy)
    output = normalize_column_types(output)
    
    # Remove the 'chunk' column from output if it exists
    # This column is likely causing the error
    if 'chunk' in output.columns:
        log.info("Removing 'chunk' column with mixed types to prevent Parquet errors")
        output = output.drop(columns=['chunk'])
    
    log.info("Writing processed data to storage")
    try:
        await write_table_to_storage(documents_copy, "dataset", context.storage)
        log.info("Successfully wrote documents_copy to storage")
    except Exception as e:
        log.error(f"Error writing documents_copy to storage: {e}")
        # Try to provide more detailed error information
        log.error(f"Column types in documents_copy: {documents_copy.dtypes}")
        for col in documents_copy.columns:
            if documents_copy[col].dtype == 'object':
                sample_types = documents_copy[col].apply(type).unique()
                log.error(f"Column '{col}' contains types: {sample_types}")
    
    try:
        await write_table_to_storage(output, "text_units", context.storage)
        log.info("Successfully wrote text_units to storage")
    except Exception as e:
        log.error(f"Error writing text_units to storage: {e}")
        # Try to provide more detailed error information
        log.error(f"Column types in output: {output.dtypes}")
        for col in output.columns:
            if output[col].dtype == 'object':
                sample_types = output[col].apply(type).unique()
                log.error(f"Column '{col}' contains types: {sample_types}")
    
    log.info("Workflow execution completed")

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
    log.info(f"Creating text units with strategy: {strategy}, size: {size}, overlap: {overlap}")
    log.info(f"Grouping by columns: {group_by_columns}")
    log.info(f"Prepend metadata: {prepend_metadata}, Chunk size includes metadata: {chunk_size_includes_metadata}")

    sort = documents.sort_values(by=["id"], ascending=[True])
    log.debug(f"Sorted {len(sort)} documents by ID")

    sort["text_with_ids"] = list(
        zip(*[sort[col] for col in ["id", "text"]], strict=True)
    )
    log.debug("Created text_with_ids column for document tracking")

    callbacks.progress(Progress(percent=0))

    agg_dict = {"text_with_ids": list}
    if "metadata" in documents:
        agg_dict["metadata"] = "first"  # type: ignore
        log.debug("Including metadata in aggregation")
    
    # Handle HTML attributes if they exist
    html_attributes_present = "html_attributes" in documents.columns
    if html_attributes_present:
        agg_dict["html_attributes"] = "first"  # type: ignore
        log.info("HTML attributes column detected - will be included in processing")

    log.info(f"Aggregating documents by {'group columns' if len(group_by_columns) > 0 else 'single group'}")
    aggregated = (
        (
            sort.groupby(group_by_columns, sort=False)
            if len(group_by_columns) > 0
            else sort.groupby(lambda _x: True)
        )
        .agg(agg_dict)
        .reset_index()
    )
    log.info(f"Aggregation resulted in {len(aggregated)} groups")
    aggregated.rename(columns={"text_with_ids": "texts"}, inplace=True)

    def chunker(row: dict[str, Any]) -> Any:
        log.debug(f"Processing row with keys: {list(row.keys())}")
        line_delimiter = ".\n"
        metadata_str = ""
        metadata_tokens = 0

        if prepend_metadata and "metadata" in row:
            metadata = row["metadata"]
            log.debug(f"Processing metadata of type: {type(metadata)}")
            
            if isinstance(metadata, str):
                try:
                    log.debug("Attempting to parse metadata from string")
                    metadata = json.loads(metadata)
                except json.JSONDecodeError:
                    # If it fails to parse, use as is
                    log.warning("Failed to parse metadata JSON, using raw string")
                    metadata = {"raw": metadata}
                    
            if isinstance(metadata, dict):
                log.debug(f"Metadata keys: {list(metadata.keys())}")
                metadata_str = (
                    line_delimiter.join(f"{k}: {v}" for k, v in metadata.items())
                    + line_delimiter
                )
                log.debug(f"Prepared metadata string of length {len(metadata_str)}")

            if chunk_size_includes_metadata:
                encode, _ = get_encoding_fn(encoding_model)
                metadata_tokens = len(encode(metadata_str))
                log.info(f"Metadata requires {metadata_tokens} tokens")
                if metadata_tokens >= size:
                    message = "Metadata tokens exceeds the maximum tokens per chunk. Please increase the tokens per chunk."
                    log.error(message)
                    raise ValueError(message)

        log.debug(f"Chunking text with effective size: {size - metadata_tokens}")
        chunked = chunk_text(
            pd.DataFrame([row]).reset_index(drop=True),
            column="texts",
            size=size - metadata_tokens,
            overlap=overlap,
            encoding_model=encoding_model,
            strategy=strategy,
            callbacks=callbacks,
        )[0]
        log.info(f"Text chunked into {len(chunked)} chunks")

        if prepend_metadata:
            log.info("Prepending metadata to each chunk")
            for index, chunk in enumerate(chunked):
                if isinstance(chunk, str):
                    chunked[index] = metadata_str + chunk
                else:
                    chunked[index] = (
                        (chunk[0], metadata_str + chunk[1], chunk[2]) if chunk else None
                    )

        row["chunks"] = chunked
        return row

    log.info("Applying chunker to each aggregated row")
    aggregated = aggregated.apply(lambda row: chunker(row), axis=1)

    # Determine columns to keep
    columns_to_keep = [*group_by_columns, "chunks"]
    if html_attributes_present:
        columns_to_keep.append("html_attributes")
        log.debug("Including html_attributes column in output")
    
    log.info(f"Keeping columns: {columns_to_keep}")
    aggregated = cast("pd.DataFrame", aggregated[columns_to_keep])
    aggregated = aggregated.explode("chunks")
    log.info(f"After exploding chunks, dataframe has {len(aggregated)} rows")
    
    aggregated.rename(
        columns={
            "chunks": "chunk",
        },
        inplace=True,
    )
    aggregated["id"] = aggregated.apply(
        lambda row: gen_sha512_hash(row, ["chunk"]), axis=1
    )
    log.debug("Generated unique IDs for each chunk")
    
    # Original approach for extracting columns
    log.info("Extracting chunk components")
    chunk_df = pd.DataFrame(
        aggregated["chunk"].tolist(), 
        index=aggregated.index
    )
    
    # Make sure column names are correct and data is properly aligned
    if len(chunk_df.columns) >= 3:
        chunk_df.columns = ["document_ids", "text", "n_tokens"]
        for col in ["document_ids", "text", "n_tokens"]:
            aggregated[col] = chunk_df[col]
        log.info(f"Extracted expected columns from chunks: {list(chunk_df.columns)}")
    else:
        # Handle case where chunk DataFrame doesn't have expected columns
        log.warning(f"Chunk DataFrame has unexpected columns: {chunk_df.columns}")
        # Set default values for missing columns
        if 0 < len(chunk_df.columns):
            aggregated["document_ids"] = chunk_df.iloc[:, 0]
            log.debug("Set document_ids from first column")
        else:
            aggregated["document_ids"] = None
            log.warning("No document_ids column available")
            
        if 1 < len(chunk_df.columns):
            aggregated["text"] = chunk_df.iloc[:, 1]
            log.debug("Set text from second column")
        else:
            aggregated["text"] = None
            log.warning("No text column available")
            
        if 2 < len(chunk_df.columns):
            aggregated["n_tokens"] = chunk_df.iloc[:, 2]
            log.debug("Set n_tokens from third column")
        else:
            aggregated["n_tokens"] = None
            log.warning("No n_tokens column available")
    
    # Process HTML attributes to extract essential info
    if html_attributes_present:
        log.info("Processing HTML attributes for each chunk")
        # Create attributes column with HTML information
        aggregated["attributes"] = aggregated.apply(
            lambda row: extract_html_attributes(row), axis=1
        )
        sample_attrs = aggregated["attributes"].iloc[0] if len(aggregated) > 0 else {}
        log.debug(f"Sample attributes structure: {json.dumps(sample_attrs, indent=2)}")
    else:
        # Ensure attributes column exists with empty dict value
        log.info("No HTML attributes present, creating empty attributes")
        aggregated["attributes"] = [{}] * len(aggregated)
    
    # Ensure document_ids is always a list
    log.info("Normalizing document_ids format")
    aggregated["document_ids"] = aggregated["document_ids"].apply(
        lambda x: [] if x is None else (x if isinstance(x, list) else [x])
    )
    
    # Convert attributes to JSON string for serialization
    log.info("Converting attributes to JSON strings")
    aggregated["attributes"] = aggregated["attributes"].apply(json.dumps)
    
    # Filter out rows with no text and reset index
    log.info("Filtering out rows with no text")
    text_na_count = aggregated["text"].isna().sum()
    if text_na_count > 0:
        log.warning(f"Removing {text_na_count} rows with missing text values")
    
    result = cast(
        "pd.DataFrame", aggregated[aggregated["text"].notna()].reset_index(drop=True)
    )
    log.info(f"Final result has {len(result)} rows")
    
    return result


def extract_html_attributes(row: pd.Series) -> Dict[str, Any]:
    """Extract HTML attributes from existing html_attributes."""
    chunk_id = row.get("id", "unknown")
    log.debug(f"Extracting HTML attributes for chunk {chunk_id}")
    
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
        log.debug(f"No text content found for chunk {chunk_id}")
        return attributes
    
    # Get HTML attributes
    html_attrs = row.get("html_attributes")
    if not html_attrs:
        log.debug(f"No HTML attributes found for chunk {chunk_id}")
        return attributes
    
    log.debug(f"HTML attributes type: {type(html_attrs)}")
    
    # Parse if JSON string
    if isinstance(html_attrs, str):
        try:
            log.debug("Attempting to parse HTML attributes from JSON string")
            html_attrs = json.loads(html_attrs)
        except json.JSONDecodeError:
            log.warning(f"Failed to parse HTML attributes JSON for chunk {chunk_id}")
            return attributes