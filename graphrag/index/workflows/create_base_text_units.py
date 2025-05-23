# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

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

        # Enhanced chunking that tracks character positions
        chunked = chunk_text_with_positions(
            row["texts"],
            size - metadata_tokens,
            overlap,
            strategy,
            encode_fn,
            decode_fn,
            row.get("metadata")  # Pass metadata for position tracking
        )

        # Prepend metadata if needed
        if prepend_metadata:
            for index, chunk in enumerate(chunked):
                if isinstance(chunk, dict):
                    # Handle new chunk format with positions
                    chunk["text"] = metadata_str + chunk["text"]
                elif isinstance(chunk, tuple) and len(chunk) >= 2:
                    # Handle tuple format
                    chunked[index] = (
                        chunk[0], 
                        metadata_str + chunk[1], 
                        chunk[2] if len(chunk) > 2 else len(encode_fn(chunk[1]))
                    )

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

    # Extract document IDs, text, and tokens with improved handling
    try:
        log.info("Extracting chunk components with structural information")
        
        # Process chunks that may have different formats
        processed_chunks = []
        for idx, chunk in aggregated["chunk"].items():
            if chunk is None:
                continue
                
            processed_chunk = {}
            
            if isinstance(chunk, dict):
                # New format with position tracking
                processed_chunk["document_ids"] = chunk.get("document_ids", [])
                processed_chunk["text"] = chunk.get("text", "")
                processed_chunk["n_tokens"] = chunk.get("n_tokens", 0)
                processed_chunk["char_start"] = chunk.get("char_start")
                processed_chunk["char_end"] = chunk.get("char_end")
                processed_chunk["source_metadata"] = chunk.get("source_metadata")
            elif isinstance(chunk, (list, tuple)) and len(chunk) >= 3:
                # Legacy tuple format
                processed_chunk["document_ids"] = chunk[0] if isinstance(chunk[0], list) else [chunk[0]]
                processed_chunk["text"] = chunk[1]
                processed_chunk["n_tokens"] = chunk[2]
                processed_chunk["char_start"] = None
                processed_chunk["char_end"] = None
                processed_chunk["source_metadata"] = None
            else:
                log.warning(f"Unexpected chunk format: {type(chunk)}")
                continue
                
            processed_chunks.append((idx, processed_chunk))
        
        # Create new columns from processed chunks
        for idx, chunk_data in processed_chunks:
            aggregated.loc[idx, "document_ids"] = chunk_data["document_ids"]
            aggregated.loc[idx, "text"] = chunk_data["text"]
            aggregated.loc[idx, "n_tokens"] = chunk_data["n_tokens"]
            aggregated.loc[idx, "char_start"] = chunk_data["char_start"]
            aggregated.loc[idx, "char_end"] = chunk_data["char_end"] 
            aggregated.loc[idx, "source_metadata"] = chunk_data["source_metadata"]

    except Exception as e:
        log.error(f"Error extracting chunk components: {e}")
        # Fallback to original method
        try:
            chunks_df = pd.DataFrame(aggregated["chunk"].tolist(), index=aggregated.index)
            if len(chunks_df.columns) >= 3:
                chunks_df.columns = ["document_ids", "text", "n_tokens"]
                aggregated["document_ids"] = chunks_df["document_ids"]
                aggregated["text"] = chunks_df["text"]
                aggregated["n_tokens"] = chunks_df["n_tokens"]
        except Exception as e2:
            log.error(f"Fallback extraction also failed: {e2}")
            raise

    # Ensure document_ids is always a list to prevent mixed type errors
    log.info("Normalizing document_ids format")
    aggregated["document_ids"] = aggregated["document_ids"].apply(
        lambda x: [] if x is None else (x if isinstance(x, list) else [x])
    )

    # Initialize structural columns
    log.info("Initializing structural columns")
    aggregated[PAGE_ID] = None
    aggregated[PAGE_NUMBER] = None
    aggregated[PARAGRAPH_ID] = None
    aggregated[PARAGRAPH_NUMBER] = None
    aggregated[CHAR_POSITION_START] = None
    aggregated[CHAR_POSITION_END] = None

    # Extract structural information if available
    if has_html_in_metadata:
        log.info("Extracting structural information from metadata and positions")
        aggregated = extract_structural_columns_improved(aggregated)
    
    # Create attributes from structural metadata
    log.info("Creating attributes column from structural metadata")
    aggregated["attributes"] = aggregated.apply(
        lambda row: create_attributes_from_metadata(row.get("source_metadata")), axis=1
    )

    # Convert attributes to JSON strings to prevent Parquet issues
    aggregated["attributes"] = aggregated["attributes"].apply(
        lambda x: json.dumps(x) if x is not None else "{}"
    )

    # Clean up temporary columns
    if "source_metadata" in aggregated.columns:
        aggregated = aggregated.drop(columns=["source_metadata"])
    if "char_start" in aggregated.columns:
        aggregated = aggregated.drop(columns=["char_start"])
    if "char_end" in aggregated.columns:
        aggregated = aggregated.drop(columns=["char_end"])

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


def chunk_text_with_positions(
    texts: list,
    size: int,
    overlap: int,
    strategy: ChunkStrategyType,
    encode_fn,
    decode_fn,
    metadata: Any = None
) -> list:
    """Enhanced chunking that tracks character positions and extracts structural info."""
    if strategy == ChunkStrategyType.tokens:
        return chunk_texts_by_tokens_with_positions(
            texts, size, overlap, encode_fn, decode_fn, metadata
        )
    else:
        # Fallback to original implementation for other strategies
        df = pd.DataFrame({"texts": [texts]})
        chunks = chunk_text(
            df,
            column="texts",
            size=size,
            overlap=overlap,
            encoding_model="",  # We already have the functions
            strategy=strategy,
            callbacks=WorkflowCallbacks(),
        )[0]
        
        # Convert to enhanced format
        enhanced_chunks = []
        for chunk in chunks:
            if isinstance(chunk, (list, tuple)) and len(chunk) >= 3:
                enhanced_chunks.append({
                    "document_ids": chunk[0],
                    "text": chunk[1],
                    "n_tokens": chunk[2],
                    "char_start": None,
                    "char_end": None,
                    "source_metadata": metadata
                })
        return enhanced_chunks


def chunk_texts_by_tokens_with_positions(
    texts: list,
    size: int,  
    overlap: int,
    encode_fn,
    decode_fn,
    metadata: Any = None
) -> list:
    """Enhanced token-based chunking that tracks positions and extracts structural info."""
    results = []
    
    # Parse metadata once if available
    parsed_metadata = None
    if metadata:
        if isinstance(metadata, str):
            try:
                parsed_metadata = json.loads(metadata)
            except:
                pass
        elif isinstance(metadata, dict):
            parsed_metadata = metadata

    for doc_id, text in texts:
        if not text or pd.isna(text):
            continue

        text_str = str(text)
        
        # Tokenize once
        tokens = encode_fn(text_str)
        
        # Create chunks efficiently with position tracking
        start_idx = 0
        char_offset = 0
        
        while start_idx < len(tokens):
            end_idx = min(start_idx + size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = decode_fn(chunk_tokens)
            
            # Calculate character positions in original text
            char_start = None
            char_end = None
            
            try:
                # Find the chunk text in the original text starting from char_offset
                chunk_start_in_text = text_str.find(chunk_text.strip(), char_offset)
                if chunk_start_in_text >= 0:
                    char_start = chunk_start_in_text
                    char_end = char_start + len(chunk_text.strip())
                    char_offset = char_end
            except:
                pass  # Position tracking failed, continue without positions
            
            # Extract structural information for this chunk
            structural_info = extract_chunk_structural_info(
                chunk_text, char_start, char_end, parsed_metadata
            )
            
            chunk_data = {
                "document_ids": [doc_id],
                "text": chunk_text,
                "n_tokens": len(chunk_tokens),
                "char_start": char_start,
                "char_end": char_end,
                "source_metadata": parsed_metadata,
                **structural_info  # Add any extracted structural info
            }
            
            results.append(chunk_data)

            if end_idx >= len(tokens):
                break

            start_idx += size - overlap

    return results


def extract_chunk_structural_info(chunk_text: str, char_start: int, char_end: int, metadata: dict) -> dict:
    """Extract structural information for a specific chunk based on its position and content."""
    structural_info = {}
    
    if not metadata or not isinstance(metadata, dict):
        return structural_info
        
    html_data = metadata.get("html", {})
    if not isinstance(html_data, dict):
        return structural_info
    
    # Extract page information based on character position
    pages = html_data.get("pages", [])
    if isinstance(pages, list) and char_start is not None:
        # Find the page this chunk belongs to
        # For now, we'll use a simple heuristic based on position
        # This could be improved with more sophisticated page boundary detection
        total_pages = len(pages)
        if total_pages > 0:
            # Estimate page based on character position
            # This is a rough approximation - in a real implementation,
            # you'd want to track actual page boundaries during parsing
            estimated_page_idx = min(int(char_start / 10000), total_pages - 1)  # Rough estimate
            if 0 <= estimated_page_idx < len(pages):
                page_info = pages[estimated_page_idx]
                if isinstance(page_info, dict):
                    structural_info["page_id"] = page_info.get("page_id")
                    structural_info["page_number"] = page_info.get("page_num")
    
    # Extract paragraph information based on character position
    paragraphs = html_data.get("paragraphs", [])
    if isinstance(paragraphs, list) and char_start is not None and char_end is not None:
        # Find paragraphs that overlap with this chunk
        for para in paragraphs:
            if isinstance(para, dict):
                para_start = para.get("char_start")
                para_end = para.get("char_end")
                
                if (para_start is not None and para_end is not None and
                    char_start < para_end and char_end > para_start):
                    # This paragraph overlaps with the chunk
                    structural_info["paragraph_id"] = para.get("para_id")
                    structural_info["paragraph_number"] = para.get("para_num")
                    structural_info["char_position_start"] = char_start
                    structural_info["char_position_end"] = char_end
                    break  # Use the first matching paragraph
    
    return structural_info


def extract_structural_columns_improved(aggregated: pd.DataFrame) -> pd.DataFrame:
    """Extract structural information to dedicated columns from chunk data."""
    log.info("Extracting structural information to dedicated columns")

    def extract_row_structure(row):
        try:
            # First, try to get structural info from the chunk data itself
            chunk = row.get("chunk")
            if isinstance(chunk, dict):
                # New chunk format may already have structural info
                if "page_id" in chunk:
                    row[PAGE_ID] = chunk["page_id"]
                if "page_number" in chunk:
                    row[PAGE_NUMBER] = chunk["page_number"]
                if "paragraph_id" in chunk:
                    row[PARAGRAPH_ID] = chunk["paragraph_id"]
                if "paragraph_number" in chunk:
                    row[PARAGRAPH_NUMBER] = chunk["paragraph_number"]
                if "char_position_start" in chunk:
                    row[CHAR_POSITION_START] = chunk["char_position_start"]
                if "char_position_end" in chunk:
                    row[CHAR_POSITION_END] = chunk["char_position_end"]
            
            # Also try to extract from source metadata if available
            source_metadata = row.get("source_metadata")
            char_start = row.get("char_start")
            char_end = row.get("char_end")
            chunk_text = row.get("text", "")
            
            if source_metadata and isinstance(source_metadata, dict):
                structural_info = extract_chunk_structural_info(
                    chunk_text, char_start, char_end, source_metadata
                )
                
                # Only override if we don't already have values
                for key, value in structural_info.items():
                    if value is not None:
                        if key == "page_id" and row.get(PAGE_ID) is None:
                            row[PAGE_ID] = value
                        elif key == "page_number" and row.get(PAGE_NUMBER) is None:
                            row[PAGE_NUMBER] = value
                        elif key == "paragraph_id" and row.get(PARAGRAPH_ID) is None:
                            row[PARAGRAPH_ID] = value
                        elif key == "paragraph_number" and row.get(PARAGRAPH_NUMBER) is None:
                            row[PARAGRAPH_NUMBER] = value
                        elif key == "char_position_start" and row.get(CHAR_POSITION_START) is None:
                            row[CHAR_POSITION_START] = value
                        elif key == "char_position_end" and row.get(CHAR_POSITION_END) is None:
                            row[CHAR_POSITION_END] = value
                            
        except Exception as e:
            log.warning(f"Error extracting structural information from row: {e}")

        return row

    # Apply extraction to each row
    log.info("Applying improved structural extraction to each row")
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


def create_attributes_from_metadata(metadata: Any) -> dict:
    """Create attributes dictionary from source metadata."""
    if not metadata or not isinstance(metadata, dict):
        return {}
        
    # Create a simplified attributes structure
    attributes = {}
    
    # Preserve HTML structure info
    if "html" in metadata:
        html_data = metadata["html"]
        if isinstance(html_data, dict):
            # Create a lightweight version for attributes
            html_attrs = {
                "doc_type": html_data.get("doc_type"),
                "filename": html_data.get("filename"),
                "has_pages": bool(html_data.get("pages")),
                "has_paragraphs": bool(html_data.get("paragraphs")),
                "page_count": len(html_data.get("pages", [])),
                "paragraph_count": len(html_data.get("paragraphs", []))
            }
            attributes["html"] = html_attrs
    
    return attributes


# Keep the rest of the helper functions unchanged
def get_cached_encoding_fn(encoding_model: str, cache_size: int = 1000):
    """Cache encoding functions for reuse across chunks."""
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