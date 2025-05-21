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

    # Write the output
    log.info("Writing output to storage")
    await write_table_to_storage(output, "text_units", context.storage)
    log.info("Successfully wrote text_units to storage")
    
    # Return the result
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
    
    # Sort documents by ID
    sort = documents.sort_values(by=["id"], ascending=[True])
    
    # Create text_with_ids column for tracking
    sort["text_with_ids"] = list(
        zip(*[sort[col] for col in ["id", "text"]], strict=True)
    )
    
    # Update progress
    callbacks.progress(Progress(percent=0))
    
    # Check for HTML structure in metadata
    has_html_in_metadata = False
    if "metadata" in documents.columns:
        try:
            has_html_in_metadata = documents["metadata"].apply(
                lambda m: isinstance(m, dict) and "html" in m 
                or (isinstance(m, str) and "html" in m)
            ).any()
            
            if has_html_in_metadata:
                log.info("Detected HTML structure in document metadata")
        except Exception as e:
            log.warning(f"Error checking for HTML in metadata: {e}")

    # Prepare aggregation dictionary
    agg_dict = {"text_with_ids": list}
    if "metadata" in documents:
        agg_dict["metadata"] = "first"  # type: ignore
    
    # Group documents
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
    
    # Define chunker function
    def chunker(row: dict[str, Any]) -> Any:
        line_delimiter = ".\n"
        metadata_str = ""
        metadata_tokens = 0
        
        # Handle metadata prepending if needed
        if prepend_metadata and "metadata" in row:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    # If JSON parsing fails, use as is
                    metadata = {"raw": metadata}
            
            if isinstance(metadata, dict):
                # Remove large HTML arrays if present to save space
                if "html" in metadata and isinstance(metadata["html"], dict):
                    metadata_copy = metadata.copy()
                    html = metadata_copy["html"].copy()
                    
                    # Remove large arrays
                    if "pages" in html:
                        html.pop("pages")
                    if "paragraphs" in html:
                        html.pop("paragraphs")
                    
                    metadata_copy["html"] = html
                    metadata = metadata_copy
                
                # Create metadata string
                metadata_str = (
                    line_delimiter.join(f"{k}: {v}" for k, v in metadata.items())
                    + line_delimiter
                )
            
            # Check if metadata fits in chunk size
            if chunk_size_includes_metadata:
                encode, _ = get_encoding_fn(encoding_model)
                metadata_tokens = len(encode(metadata_str))
                if metadata_tokens >= size:
                    message = "Metadata tokens exceeds the maximum tokens per chunk. Please increase the tokens per chunk."
                    log.error(message)
                    raise ValueError(message)
        
        # Chunk the text
        chunked = chunk_text(
            pd.DataFrame([row]).reset_index(drop=True),
            column="texts",
            size=size - metadata_tokens,
            overlap=overlap,
            encoding_model=encoding_model,
            strategy=strategy,
            callbacks=callbacks,
        )[0]
        
        # Prepend metadata if needed
        if prepend_metadata:
            for index, chunk in enumerate(chunked):
                if isinstance(chunk, str):
                    chunked[index] = metadata_str + chunk
                else:
                    chunked[index] = (
                        (chunk[0], metadata_str + chunk[1], chunk[2]) if chunk else None
                    )
        
        # Add chunk metadata with HTML info
        if "metadata" in row and has_html_in_metadata:
            orig_metadata = row["metadata"]
            if isinstance(orig_metadata, str):
                try:
                    orig_metadata = json.loads(orig_metadata)
                except:
                    orig_metadata = {}
            
            # Extract HTML position info for each chunk
            if isinstance(orig_metadata, dict) and "html" in orig_metadata and isinstance(orig_metadata["html"], dict):
                html_meta = orig_metadata["html"]
                
                # Create enhanced chunks with position info
                for i, chunk in enumerate(chunked):
                    # Skip if chunk is not in expected format
                    if not chunk or not isinstance(chunk, tuple) or len(chunk) < 3:
                        continue
                    
                    # Create base chunk
                    base_chunk = (chunk[0], chunk[1], chunk[2])
                    
                    # Add basic metadata without large arrays
                    chunk_meta = {}
                    if isinstance(html_meta, dict):
                        # Copy basic HTML properties
                        chunk_meta = {
                            "html": {
                                "doc_type": html_meta.get("doc_type"),
                                "has_pages": html_meta.get("has_pages", False),
                                "has_paragraphs": html_meta.get("has_paragraphs", False),
                                "page_count": html_meta.get("page_count", 0),
                                "paragraph_count": html_meta.get("paragraph_count", 0)
                            }
                        }
                    
                    # Store enhanced chunk
                    chunked[i] = (*base_chunk, chunk_meta)
        
        # Store the chunks
        row["chunks"] = chunked
        return row
    
    # Apply chunker
    log.info("Chunking documents")
    aggregated = aggregated.apply(lambda row: chunker(row), axis=1)
    
    # Keep only necessary columns
    log.info("Processing chunked results")
    aggregated = cast("pd.DataFrame", aggregated[[*group_by_columns, "chunks"]])
    
    # Explode to create one row per chunk
    aggregated = aggregated.explode("chunks")
    
    # Rename chunks column
    aggregated.rename(
        columns={
            "chunks": "chunk",
        },
        inplace=True,
    )
    
    # Generate unique IDs
    aggregated["id"] = aggregated.apply(
        lambda row: gen_sha512_hash(row, ["chunk"]), axis=1
    )
    
    # Extract document IDs, text, and tokens
    try:
        # Convert chunks to DataFrame for extraction
        chunks_df = pd.DataFrame(
            aggregated["chunk"].tolist(), 
            index=aggregated.index
        )
        
        # Set column names based on length
        if len(chunks_df.columns) >= 4:
            # Has metadata column
            chunks_df.columns = ["document_ids", "text", "n_tokens", "metadata"]
            # Convert metadata to string to avoid mixed type issues
            chunk_metadata = chunks_df["metadata"].apply(
                lambda x: json.dumps(x) if isinstance(x, dict) else 
                          (x if isinstance(x, str) else None)
            )
            aggregated["chunk_metadata"] = chunk_metadata
        else:
            # Standard columns
            chunks_df.columns = ["document_ids", "text", "n_tokens"]
        
        # Add extracted columns
        aggregated["document_ids"] = chunks_df["document_ids"]
        aggregated["text"] = chunks_df["text"]
        aggregated["n_tokens"] = chunks_df["n_tokens"]
        
    except Exception as e:
        log.error(f"Error extracting chunk components: {e}")
        # Create a simple extraction using original method as fallback
        aggregated[["document_ids", "chunk", "n_tokens"]] = pd.DataFrame(
            aggregated["chunk"].tolist(), index=aggregated.index
        )
        # Rename for downstream consumption
        aggregated.rename(columns={"chunk": "text"}, inplace=True)
    
    # Ensure document_ids is always a list to prevent mixed type errors
    log.info("Normalizing document_ids format")
    aggregated["document_ids"] = aggregated["document_ids"].apply(
        lambda x: [] if x is None else (x if isinstance(x, list) else [x])
    )
    
    # Add HTML structure columns if metadata contains HTML
    if "chunk_metadata" in aggregated.columns and has_html_in_metadata:
        # Add structure columns with default None values
        aggregated["page_id"] = None
        aggregated["page_number"] = None
        aggregated["paragraph_id"] = None
        aggregated["paragraph_number"] = None
        aggregated["char_position_start"] = None
        aggregated["char_position_end"] = None
        
        # Create attributes from structured info
        log.info("Creating attributes column")
        aggregated["attributes"] = aggregated.apply(
            lambda row: create_attributes(row.get("chunk_metadata", None)), axis=1
        )
        
        # Convert attributes to JSON strings to prevent Parquet issues
        # IMPORTANT: Always store as JSON string, never as dict
        aggregated["attributes"] = aggregated["attributes"].apply(
            lambda x: json.dumps(x) if x is not None else "{}"
        )
        
        # Remove temporary column
        if "chunk_metadata" in aggregated.columns:
            aggregated = aggregated.drop(columns=["chunk_metadata"])
    else:
        # Create empty attributes column as JSON strings
        aggregated["attributes"] = ["{}"] * len(aggregated)
    
    # Filter out rows with no text and reset index
    log.info("Finalizing results")
    result = cast(
        "pd.DataFrame", aggregated[aggregated["text"].notna()].reset_index(drop=True)
    )
    
    return result


def create_attributes(metadata_str) -> dict:
    """Create a simple attributes dictionary from metadata string."""
    attributes = {}
    
    # Parse metadata if it's a string
    metadata = {}
    if isinstance(metadata_str, str) and metadata_str:
        try:
            metadata = json.loads(metadata_str)
        except:
            return attributes
    
    # Create simplified structure with HTML info
    if isinstance(metadata, dict) and "html" in metadata:
        html = metadata.get("html", {})
        if isinstance(html, dict):
            # Store basic HTML info - only primitive types
            html_props = {}
            for key, value in html.items():
                # Skip non-serializable values and large arrays
                if key not in ["pages", "paragraphs"] and isinstance(value, (str, int, float, bool)) or value is None:
                    html_props[key] = value
            
            attributes["html"] = html_props
    
    # Include page, paragraph, and char_position if present
    # Make sure we only keep primitive types that Parquet can handle
    for key in ["page", "paragraph", "char_position"]:
        if key in metadata and metadata[key]:
            # Convert dictionaries to flattened structures
            if isinstance(metadata[key], dict):
                flat_data = {}
                for subkey, subvalue in metadata[key].items():
                    # Only keep primitive types
                    if isinstance(subvalue, (str, int, float, bool)) or subvalue is None:
                        flat_data[subkey] = subvalue
                attributes[key] = flat_data
    
    return attributes