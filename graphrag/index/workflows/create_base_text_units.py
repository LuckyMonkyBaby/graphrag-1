import json
import logging
import multiprocessing as mp
import gc
import os
import time
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast, Optional, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.chunking_config import ChunkStrategyType
from graphrag.config.models.graph_rag_config import GraphRagConfig
from graphrag.data_model.schemas import (
    CHAR_POSITION_END,
    CHAR_POSITION_START,
    PAGE_ID,
    PAGE_NUMBER,
    PARAGRAPH_ID,
    PARAGRAPH_NUMBER,
)
from graphrag.index.operations.chunk_text.chunk_text import chunk_text
from graphrag.index.operations.chunk_text.strategies import get_encoding_fn
from graphrag.index.typing.context import PipelineRunContext
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.utils.hashing import gen_sha512_hash
from graphrag.logger.progress import Progress
from graphrag.utils.storage import load_table_from_storage, write_table_to_storage

# Add logger
log = logging.getLogger(__name__)

# Production optimization imports


@dataclass
class ChunkingMetrics:
    """Performance metrics for chunking operations."""

    total_documents: int = 0
    total_chunks: int = 0
    processing_time: float = 0.0
    memory_peak_mb: float = 0.0
    batches_processed: int = 0


async def run_workflow(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """All the steps to transform base text_units with optimizations."""
    import time

    start_time = time.time()

    log.info("Starting optimized run_workflow process")
    documents = await load_table_from_storage("documents", context.storage)
    log.info(f"Loaded {len(documents)} documents from storage")

    chunks = config.chunks
    log.info(
        f"Using chunk configuration: size={chunks.size}, overlap={chunks.overlap}, strategy={chunks.strategy}"
    )

    # Use configuration-based performance settings with smart defaults
    doc_count = len(documents)
    batch_size = chunks.batch_size
    max_workers = min(chunks.max_workers, mp.cpu_count())
    enable_parallel = chunks.enable_parallel and doc_count >= chunks.parallel_threshold
    
    # Auto-tune for very large or very small datasets
    if doc_count > 1000:
        batch_size = min(batch_size, max(50, doc_count // 10))
    elif doc_count < chunks.parallel_threshold:
        enable_parallel = False
        max_workers = 1

    log.info(
        f"Performance settings: batch_size={batch_size}, max_workers={max_workers}, parallel={enable_parallel}"
    )
    log.info(f"Dataset size: {doc_count} documents, threshold: {chunks.parallel_threshold}")

    # Create the text units with optimizations
    log.info("Creating base text units with production optimizations")
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
        batch_size=batch_size,
        max_workers=max_workers,
        enable_parallel=enable_parallel,
        metadata_cache_size=chunks.metadata_cache_size,
    )

    processing_time = time.time() - start_time
    log.info(f"Created {len(output)} text units in {processing_time:.2f}s")
    log.info(f"Throughput: {len(output) / processing_time:.1f} chunks/sec")

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
    batch_size: int = 100,
    max_workers: int = 8,
    enable_parallel: bool = True,
    metadata_cache_size: int = 1000,
) -> pd.DataFrame:
    """Create base text units with production optimizations."""
    import time

    import psutil

    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    metrics = ChunkingMetrics(total_documents=len(documents))

    log.info(
        f"Creating text units with strategy: {strategy}, size: {size}, overlap: {overlap}"
    )
    log.info(
        f"Production settings: batch_size={batch_size}, max_workers={max_workers}, parallel={enable_parallel}"
    )

    # Efficient sorting and preparation
    log.info("Preparing documents for processing")
    sort = documents.sort_values(by=["id"], ascending=[True]).copy()

    # Use vectorized operations for better performance
    sort["text_with_ids"] = list(zip(sort["id"], sort["text"], strict=True))

    # Update progress
    callbacks.progress(Progress(percent=0))

    # Optimized HTML metadata detection
    has_html_in_metadata = False
    if "metadata" in documents.columns and not documents["metadata"].isna().all():
        try:
            # Use vectorized string operations for better performance
            metadata_sample = (
                documents["metadata"].dropna().iloc[:10]
            )  # Sample for efficiency
            has_html_in_metadata = any("html" in str(m) for m in metadata_sample)

            if has_html_in_metadata:
                log.info("Detected HTML structure in document metadata")
        except Exception as e:
            log.warning(f"Error checking for HTML in metadata: {e}")

    # Efficient grouping with optimized aggregation
    log.info("Grouping documents for processing")
    agg_dict = {"text_with_ids": list}
    if "metadata" in sort.columns:
        agg_dict["metadata"] = "first"

    # Use efficient grouping
    if len(group_by_columns) > 0:
        grouped = sort.groupby(group_by_columns, sort=False, observed=True)
    else:
        # Create a single group more efficiently
        sort["_temp_group"] = 0
        grouped = sort.groupby("_temp_group", sort=False)

    aggregated = grouped.agg(agg_dict).reset_index()
    aggregated.rename(columns={"text_with_ids": "texts"}, inplace=True)

    # Clean up temporary column if created
    if "_temp_group" in aggregated.columns:
        aggregated.drop(columns=["_temp_group"], inplace=True)

    # Cache encoding function for reuse with configurable cache size
    encode_fn, decode_fn = get_cached_encoding_fn(encoding_model, metadata_cache_size)

    # Define optimized chunker function
    def chunker(row: dict[str, Any]) -> Any:
        line_delimiter = ".\n"
        metadata_str = ""
        metadata_tokens = 0

        # Optimized metadata handling
        if prepend_metadata and "metadata" in row:
            metadata_str, metadata_tokens = process_metadata_optimized(
                row["metadata"],
                line_delimiter,
                size,
                chunk_size_includes_metadata,
                encode_fn,
            )
        else:
            metadata_tokens = 0

        # Efficient chunking without DataFrame overhead
        chunked = chunk_text_optimized(
            row["texts"],
            size - metadata_tokens,
            overlap,
            strategy,
            encode_fn,
            decode_fn,
        )

        # Prepend metadata if needed
        if prepend_metadata:
            for index, chunk in enumerate(chunked):
                if isinstance(chunk, str):
                    chunked[index] = metadata_str + chunk
                else:
                    chunked[index] = (
                        (chunk[0], metadata_str + chunk[1], chunk[2]) if chunk else None
                    )

        # Optimized metadata enhancement (only when necessary)
        if "metadata" in row and has_html_in_metadata:
            chunked = enhance_chunks_with_metadata(chunked, row["metadata"])

        # Store the chunks
        row["chunks"] = chunked
        return row

    # Apply chunker with batching and optional parallelization
    log.info(f"Chunking {len(aggregated)} document groups")

    if enable_parallel and len(aggregated) > batch_size:
        aggregated = process_chunks_parallel(
            aggregated, chunker, batch_size, max_workers, callbacks
        )
        metrics.batches_processed = len(aggregated) // batch_size + 1
    else:
        # Sequential processing for smaller datasets
        aggregated = aggregated.apply(chunker, axis=1)
        metrics.batches_processed = 1

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
        chunks_df = pd.DataFrame(aggregated["chunk"].tolist(), index=aggregated.index)

        # Set column names based on length
        if len(chunks_df.columns) >= 4:
            # Has metadata column
            chunks_df.columns = ["document_ids", "text", "n_tokens", "metadata"]
            # Convert metadata to string to avoid mixed type issues
            chunk_metadata = chunks_df["metadata"].apply(
                lambda x: json.dumps(x)
                if isinstance(x, dict)
                else (x if isinstance(x, str) else None)
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
        # Initialize structural columns with default None values
        aggregated[PAGE_ID] = None
        aggregated[PAGE_NUMBER] = None
        aggregated[PARAGRAPH_ID] = None
        aggregated[PARAGRAPH_NUMBER] = None
        aggregated[CHAR_POSITION_START] = None
        aggregated[CHAR_POSITION_END] = None

        # Create attributes from structured info preserving PRIMARY data
        log.info("Creating attributes column with preserved structural data")
        aggregated["attributes"] = aggregated.apply(
            lambda row: create_attributes(row.get("chunk_metadata", None)), axis=1
        )

        # Convert attributes to JSON strings to prevent Parquet issues
        # IMPORTANT: Always store as JSON string, never as dict
        aggregated["attributes"] = aggregated["attributes"].apply(
            lambda x: json.dumps(x) if x is not None else "{}"
        )

        # EXTRACT structural information to dedicated columns
        log.info("Extracting structural information to dedicated columns")
        aggregated = extract_structural_columns(aggregated)

        # Remove temporary column
        if "chunk_metadata" in aggregated.columns:
            aggregated = aggregated.drop(columns=["chunk_metadata"])
    else:
        # Create empty attributes column as JSON strings and initialize structural columns
        aggregated["attributes"] = ["{}"] * len(aggregated)
        aggregated[PAGE_ID] = None
        aggregated[PAGE_NUMBER] = None
        aggregated[PARAGRAPH_ID] = None
        aggregated[PARAGRAPH_NUMBER] = None
        aggregated[CHAR_POSITION_START] = None
        aggregated[CHAR_POSITION_END] = None

    # Filter out rows with no text and reset index
    log.info("Finalizing results")
    result = cast(
        "pd.DataFrame", aggregated[aggregated["text"].notna()].reset_index(drop=True)
    )

    # Performance metrics logging and reporting via callbacks
    end_time = time.time()
    current_memory = process.memory_info().rss / 1024 / 1024  # MB
    metrics.processing_time = end_time - start_time
    metrics.memory_peak_mb = max(initial_memory, current_memory)
    metrics.total_chunks = len(result)

    # Report performance metrics via the callback system
    performance_summary = {
        "total_documents": metrics.total_documents,
        "total_chunks": metrics.total_chunks,
        "processing_time": metrics.processing_time,
        "memory_peak_mb": metrics.memory_peak_mb,
        "throughput_chunks_per_sec": metrics.total_chunks / metrics.processing_time if metrics.processing_time > 0 else 0,
        "batches_processed": metrics.batches_processed,
        "parallel_enabled": enable_parallel,
        "batch_size_used": batch_size,
        "workers_used": max_workers,
    }
    
    # Report via progress callback with performance data
    callbacks.progress(Progress(
        percent=100,
        description=f"Text chunking completed: {metrics.total_chunks} chunks, {metrics.processing_time:.2f}s"
    ))

    log.info(
        f"Performance metrics: {metrics.total_documents} docs -> {metrics.total_chunks} chunks"
    )
    log.info(
        f"Processing time: {metrics.processing_time:.2f}s, Memory peak: {metrics.memory_peak_mb:.1f}MB"
    )
    log.info(
        f"Throughput: {metrics.total_chunks / metrics.processing_time:.1f} chunks/sec"
    )

    return result


def create_attributes(metadata_str) -> dict:
    """Create attributes dictionary preserving primary structural information."""
    log.info("Creating attributes from metadata string")

    attributes = {}

    # Parse metadata if it's a string
    metadata = {}
    if isinstance(metadata_str, str) and metadata_str:
        try:
            log.debug(
                f"Parsing JSON metadata string: {metadata_str[:100]}..."
                if len(metadata_str) > 100
                else metadata_str
            )
            metadata = json.loads(metadata_str)
        except Exception as e:
            log.warning(f"Failed to parse metadata JSON: {e}")
            return attributes
    else:
        log.debug(f"Metadata is not a string: {type(metadata_str)}")

    # Log the full metadata structure for debugging
    try:
        metadata_sample = (
            str(metadata)[:500] + "..." if len(str(metadata)) > 500 else str(metadata)
        )
        log.debug(f"Raw metadata structure: {metadata_sample}")
    except Exception as e:
        log.warning(f"Failed to log metadata structure: {e}")

    # PRESERVE PRIMARY STRUCTURAL INFORMATION - DO NOT REMOVE
    if isinstance(metadata, dict) and "html" in metadata:
        html = metadata.get("html", {})
        if isinstance(html, dict):
            log.debug(f"HTML metadata keys: {list(html.keys())}")

            # Keep ALL HTML properties including pages and paragraphs
            # These are PRIMARY structural attributes, not just metadata
            html_props = {}
            for key, value in html.items():
                if isinstance(value, (str, int, float, bool, type(None))):
                    html_props[key] = value
                    log.debug(f"Added HTML property {key}: {value}")
                elif key in ["pages", "paragraphs"]:
                    # CRITICAL: Keep pages and paragraphs as they contain primary structural data
                    if isinstance(value, list):
                        # Preserve the structural information but ensure serializable format
                        html_props[key] = value  # Keep the full list
                        log.info(
                            f"PRESERVED primary structural data '{key}' with {len(value)} elements"
                        )
                    else:
                        html_props[key] = value
                        log.debug(f"Added non-list {key}: {type(value)}")
                else:
                    # For other complex objects, convert to string but preserve structure
                    if isinstance(value, (dict, list)):
                        html_props[key] = json.dumps(value) if value else None
                        log.debug(f"Serialized complex HTML property {key}")
                    else:
                        html_props[key] = str(value) if value is not None else None
                        log.debug(f"Converted HTML property {key} to string")

            attributes["html"] = html_props

    # PRESERVE page, paragraph, and char_position as PRIMARY attributes
    for key in ["page", "paragraph", "char_position"]:
        if key in metadata and metadata[key]:
            log.debug(f"Processing PRIMARY attribute {key}: {metadata[key]}")

            if isinstance(metadata[key], dict):
                # Keep the full dictionary structure for primary attributes
                attributes[key] = metadata[key]
                log.info(
                    f"PRESERVED primary attribute {key} with keys: {list(metadata[key].keys())}"
                )
            elif isinstance(metadata[key], (list, str, int, float, bool)):
                # Keep primitive and list values as-is
                attributes[key] = metadata[key]
                log.info(f"PRESERVED primary attribute {key}: {type(metadata[key])}")
            else:
                # Convert complex objects to serializable format but preserve data
                try:
                    attributes[key] = (
                        json.loads(json.dumps(metadata[key])) if metadata[key] else None
                    )
                    log.info(f"PRESERVED primary attribute {key} after serialization")
                except (TypeError, ValueError):
                    attributes[key] = str(metadata[key])
                    log.warning(
                        f"Converted primary attribute {key} to string due to serialization issues"
                    )

    # Log the final attributes structure
    log.info(
        f"Final attributes created with PRIMARY structural data preserved: {list(attributes.keys())}"
    )

    return attributes


def extract_structural_columns(aggregated: pd.DataFrame) -> pd.DataFrame:
    """Extract structural information to dedicated columns while preserving in attributes."""
    log.info("Extracting structural information to dedicated columns")

    # Extract structural information from attributes for each row
    def extract_row_structure(row):
        try:
            # Parse attributes if it's a string
            attrs = row.get("attributes", "{}")
            if isinstance(attrs, str):
                try:
                    attrs = json.loads(attrs)
                except:
                    attrs = {}

            # Extract page information
            if "page" in attrs and isinstance(attrs["page"], dict):
                page_info = attrs["page"]
                row[PAGE_ID] = page_info.get("id") or page_info.get("page_id")
                row[PAGE_NUMBER] = page_info.get("number") or page_info.get("page_num")
                log.debug(
                    f"Extracted page info: ID={row[PAGE_ID]}, Number={row[PAGE_NUMBER]}"
                )

            # Extract paragraph information
            if "paragraph" in attrs and isinstance(attrs["paragraph"], dict):
                para_info = attrs["paragraph"]
                row[PARAGRAPH_ID] = para_info.get("id") or para_info.get("para_id")
                row[PARAGRAPH_NUMBER] = para_info.get("number") or para_info.get(
                    "para_num"
                )
                log.debug(
                    f"Extracted paragraph info: ID={row[PARAGRAPH_ID]}, Number={row[PARAGRAPH_NUMBER]}"
                )

            # Extract character position information
            if "char_position" in attrs and isinstance(attrs["char_position"], dict):
                char_info = attrs["char_position"]
                row[CHAR_POSITION_START] = char_info.get("start") or char_info.get(
                    "char_start"
                )
                row[CHAR_POSITION_END] = char_info.get("end") or char_info.get(
                    "char_end"
                )
                log.debug(
                    f"Extracted char positions: Start={row[CHAR_POSITION_START]}, End={row[CHAR_POSITION_END]}"
                )

            # Also check HTML structure for additional information
            if "html" in attrs and isinstance(attrs["html"], dict):
                html_info = attrs["html"]

                # Look for page/paragraph arrays to extract specific chunk positions
                if "pages" in html_info and isinstance(html_info["pages"], list):
                    # Extract page information based on chunk content or position
                    chunk_content = row.get("text", "")
                    page_match = find_chunk_page_info(chunk_content, html_info["pages"])
                    if page_match:
                        row[PAGE_ID] = page_match.get("page_id")
                        row[PAGE_NUMBER] = page_match.get("page_num")
                        log.debug(
                            f"Found page match from HTML: ID={page_match.get('page_id')}, Number={page_match.get('page_num')}"
                        )

                if "paragraphs" in html_info and isinstance(
                    html_info["paragraphs"], list
                ):
                    # Extract paragraph information based on chunk content or position
                    chunk_content = row.get("text", "")
                    para_match = find_chunk_paragraph_info(
                        chunk_content, html_info["paragraphs"]
                    )
                    if para_match:
                        row[PARAGRAPH_ID] = para_match.get("para_id")
                        row[PARAGRAPH_NUMBER] = para_match.get("para_num")
                        row[CHAR_POSITION_START] = para_match.get("char_start")
                        row[CHAR_POSITION_END] = para_match.get("char_end")
                        log.debug(
                            f"Found paragraph match from HTML: ID={para_match.get('para_id')}, Number={para_match.get('para_num')}"
                        )

        except Exception as e:
            log.warning(f"Error extracting structural information from row: {e}")

        return row

    # Apply extraction to each row
    log.info("Applying structural extraction to each row")
    aggregated = aggregated.apply(extract_row_structure, axis=1)

    # Log summary of extracted information
    structural_columns = [
        PAGE_ID,
        PAGE_NUMBER,
        PARAGRAPH_ID,
        PARAGRAPH_NUMBER,
        CHAR_POSITION_START,
        CHAR_POSITION_END,
    ]
    for col in structural_columns:
        non_null_count = aggregated[col].notna().sum()
        log.info(f"Extracted {non_null_count} values for {col}")

    return aggregated


def determine_chunk_position(chunk_text: str, html_meta: dict) -> dict:
    """Determine the position of a chunk within the document structure."""
    # With simplified HTML structure, position determination should happen
    # during the chunking process using character positions, not retroactively
    # by matching text content
    log.debug("Position determination disabled - should be handled during chunking")
    return {}


def find_chunk_page_info(chunk_text: str, pages: list) -> dict:
    """Find page information for a chunk based on its text content."""
    # For HTML documents, page markers are just page numbers, not content boundaries
    # We can't reliably match chunk text to page markers, so we skip page assignment
    # Page info should be determined by character positions during chunking instead
    log.debug("Page matching skipped for HTML - page markers don't contain content boundaries")
    return {}


def find_chunk_paragraph_info(chunk_text: str, paragraphs: list) -> dict:
    """Find paragraph information for a chunk based on character positions."""
    # For HTML documents, we only have paragraph positions (not text content)
    # Character position matching should be done during chunking process
    # This function returns empty since we can't match without original text
    log.debug("Paragraph matching needs character position context from chunking process")
    return {}


# Production optimization helper functions


def get_cached_encoding_fn(encoding_model: str, cache_size: int = 1000):
    """Cache encoding functions for reuse across chunks."""
    # Create a function with the specified cache size
    @lru_cache(maxsize=cache_size)
    def _cached_get_encoding_fn(model: str):
        return get_encoding_fn(model)
    
    return _cached_get_encoding_fn(encoding_model)


def process_metadata_optimized(
    metadata: Any,
    line_delimiter: str,
    size: int,
    chunk_size_includes_metadata: bool,
    encode_fn,
) -> tuple[str, int]:
    """Optimized metadata processing with minimal JSON parsing."""
    metadata_str = ""
    metadata_tokens = 0

    try:
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {"raw": metadata}

        if isinstance(metadata, dict):
            # Efficient metadata string creation
            metadata_items = []
            for k, v in metadata.items():
                if k == "html" and isinstance(v, dict):
                    # Streamlined HTML metadata (exclude large arrays)
                    html_summary = {
                        key: val
                        for key, val in v.items()
                        if key not in ["pages", "paragraphs"]
                        and isinstance(val, (str, int, float, bool))
                    }
                    metadata_items.append(f"{k}: {json.dumps(html_summary)}")
                else:
                    metadata_items.append(f"{k}: {v}")

            metadata_str = line_delimiter.join(metadata_items) + line_delimiter

            # Check token count if needed
            if chunk_size_includes_metadata and metadata_str:
                metadata_tokens = len(encode_fn(metadata_str))
                if metadata_tokens >= size:
                    raise ValueError(
                        "Metadata tokens exceeds the maximum tokens per chunk."
                    )

    except Exception as e:
        log.warning(f"Error processing metadata: {e}")
        metadata_str = ""
        metadata_tokens = 0

    return metadata_str, metadata_tokens


def chunk_text_optimized(
    texts: list,
    size: int,
    overlap: int,
    strategy: ChunkStrategyType,
    encode_fn,
    decode_fn,
) -> list:
    """Optimized chunking without DataFrame overhead."""
    if strategy == ChunkStrategyType.tokens:
        return chunk_texts_by_tokens_optimized(
            texts, size, overlap, encode_fn, decode_fn
        )
    else:
        # Fallback to original implementation for other strategies
        df = pd.DataFrame({"texts": [texts]})
        return chunk_text(
            df,
            column="texts",
            size=size,
            overlap=overlap,
            encoding_model="",  # We already have the functions
            strategy=strategy,
            callbacks=WorkflowCallbacks(),
        )[0]


def chunk_texts_by_tokens_optimized(
    texts: list,
    size: int,
    overlap: int,
    encode_fn,
    decode_fn,
) -> list:
    """Optimized token-based chunking."""
    results = []

    for doc_id, text in texts:
        if not text or pd.isna(text):
            continue

        # Tokenize once
        tokens = encode_fn(str(text))

        # Create chunks efficiently
        start_idx = 0
        while start_idx < len(tokens):
            end_idx = min(start_idx + size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = decode_fn(chunk_tokens)

            results.append(([doc_id], chunk_text, len(chunk_tokens)))

            if end_idx >= len(tokens):
                break

            start_idx += size - overlap

    return results


def enhance_chunks_with_metadata(chunks: list, metadata: Any) -> list:
    """Optimized metadata enhancement for chunks."""
    try:
        if isinstance(metadata, str):
            metadata = json.loads(metadata)

        if not isinstance(metadata, dict) or "html" not in metadata:
            return chunks

        html_meta = metadata["html"]
        if not isinstance(html_meta, dict):
            return chunks

        # Create lightweight metadata for chunks
        base_chunk_meta = {
            "html": {
                "doc_type": html_meta.get("doc_type"),
                "has_pages": html_meta.get("has_pages", False),
                "has_paragraphs": html_meta.get("has_paragraphs", False),
                "page_count": html_meta.get("page_count", 0),
                "paragraph_count": html_meta.get("paragraph_count", 0),
            }
        }

        # Only add full structural data for first few chunks to avoid memory issues
        enhanced_chunks = []
        for i, chunk in enumerate(chunks):
            if not chunk or not isinstance(chunk, tuple) or len(chunk) < 3:
                enhanced_chunks.append(chunk)
                continue

            chunk_meta = base_chunk_meta.copy()

            # Only preserve full structural data for first 10 chunks per document
            if i < 10:
                chunk_meta["html"].update({
                    "pages": html_meta.get("pages", []),
                    "paragraphs": html_meta.get("paragraphs", []),
                })

            enhanced_chunks.append((*chunk[:3], chunk_meta))

        return enhanced_chunks

    except Exception as e:
        log.warning(f"Error enhancing chunks with metadata: {e}")
        return chunks


def process_chunks_parallel(
    aggregated: pd.DataFrame,
    chunker_fn,
    batch_size: int,
    max_workers: int,
    callbacks: WorkflowCallbacks,
) -> pd.DataFrame:
    """Process chunks in parallel batches with progress tracking."""
    log.info(
        f"Processing {len(aggregated)} groups in parallel with {max_workers} workers"
    )

    # Split into batches
    batches = [
        aggregated.iloc[i : i + batch_size].copy()
        for i in range(0, len(aggregated), batch_size)
    ]
    
    total_batches = len(batches)
    processed_count = 0

    def process_batch(batch_df):
        """Process a batch of documents."""
        return batch_df.apply(chunker_fn, axis=1)

    # Use ThreadPoolExecutor for I/O bound operations (better for DataFrame operations)
    processed_batches = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all batches and track progress
        future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
        
        for future in future_to_batch:
            batch_result = future.result()
            processed_batches.append(batch_result)
            processed_count += 1
            
            # Report progress
            progress_percent = (processed_count / total_batches) * 90  # Reserve 10% for final processing
            callbacks.progress(Progress(
                percent=progress_percent,
                description=f"Processing batch {processed_count}/{total_batches}"
            ))

    # Final progress update
    callbacks.progress(Progress(percent=95, description="Combining batch results"))

    # Concatenate results
    return pd.concat(processed_batches, ignore_index=True)
