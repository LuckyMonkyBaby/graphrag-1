# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing run_workflow method definition."""

import json
import logging
import multiprocessing as mp
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast, Optional

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

log = logging.getLogger(__name__)


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
    start_time = time.time()

    log.info("Starting optimized run_workflow process")
    documents = await load_table_from_storage("documents", context.storage)
    log.info(f"Loaded {len(documents)} documents from storage")

    chunks = config.chunks
    log.info(f"Using chunk configuration: size={chunks.size}, overlap={chunks.overlap}, strategy={chunks.strategy}")

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

    log.info(f"Performance settings: batch_size={batch_size}, max_workers={max_workers}, parallel={enable_parallel}")

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

    # Write the output
    log.info("Writing output to storage")
    await write_table_to_storage(output, "text_units", context.storage)
    log.info("Successfully wrote text_units to storage")

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
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024

    metrics = ChunkingMetrics(total_documents=len(documents))

    log.info(f"Creating text units with strategy: {strategy}, size: {size}, overlap: {overlap}")

    # Efficient sorting and preparation
    sort = documents.sort_values(by=["id"], ascending=[True]).copy()
    sort["text_with_ids"] = list(zip(sort["id"], sort["text"], strict=True))

    callbacks.progress(Progress(percent=0))

    # Check for HTML metadata
    has_html_in_metadata = False
    if "metadata" in documents.columns and not documents["metadata"].isna().all():
        metadata_sample = documents["metadata"].dropna().iloc[:10]
        has_html_in_metadata = any("html" in str(m) for m in metadata_sample)
        if has_html_in_metadata:
            log.info("Detected HTML structure in document metadata")

    # Efficient grouping
    agg_dict = {"text_with_ids": list}
    if "metadata" in sort.columns:
        agg_dict["metadata"] = "first"

    if len(group_by_columns) > 0:
        grouped = sort.groupby(group_by_columns, sort=False, observed=True)
    else:
        sort["_temp_group"] = 0
        grouped = sort.groupby("_temp_group", sort=False)

    aggregated = grouped.agg(agg_dict).reset_index()
    aggregated.rename(columns={"text_with_ids": "texts"}, inplace=True)

    if "_temp_group" in aggregated.columns:
        aggregated.drop(columns=["_temp_group"], inplace=True)

    # Cache encoding function
    encode_fn, decode_fn = get_cached_encoding_fn(encoding_model, metadata_cache_size)

    # Define chunker function
    def chunker(row: dict[str, Any]) -> Any:
        line_delimiter = ".\n"
        metadata_str = ""
        metadata_tokens = 0

        if prepend_metadata and "metadata" in row:
            metadata_str, metadata_tokens = process_metadata_optimized(
                row["metadata"], line_delimiter, size, chunk_size_includes_metadata, encode_fn
            )

        # Enhanced chunking that tracks character positions
        chunked = chunk_text_with_positions(
            row["texts"],
            size - metadata_tokens,
            overlap,
            strategy,
            encode_fn,
            decode_fn,
            row.get("metadata")
        )

        # Prepend metadata if needed
        if prepend_metadata and metadata_str:
            for chunk in chunked:
                if isinstance(chunk, dict):
                    chunk["text"] = metadata_str + chunk["text"]

        row["chunks"] = chunked
        return row

    # Apply chunker
    log.info(f"Chunking {len(aggregated)} document groups")
    if enable_parallel and len(aggregated) > batch_size:
        aggregated = process_chunks_parallel(aggregated, chunker, batch_size, max_workers, callbacks)
        metrics.batches_processed = len(aggregated) // batch_size + 1
    else:
        aggregated = aggregated.apply(chunker, axis=1)
        metrics.batches_processed = 1

    # Keep only necessary columns and explode
    aggregated = cast("pd.DataFrame", aggregated[[*group_by_columns, "chunks"]])
    aggregated = aggregated.explode("chunks").reset_index(drop=True)  # CRITICAL: reset_index to avoid duplicates
    aggregated.rename(columns={"chunks": "chunk"}, inplace=True)

    # Generate unique IDs
    aggregated["id"] = aggregated.apply(lambda row: gen_sha512_hash(row, ["chunk"]), axis=1)

    # Extract chunk components - FIXED VERSION
    log.info("Extracting chunk components")
    
    # Initialize all columns first
    aggregated["document_ids"] = None
    aggregated["text"] = None
    aggregated["n_tokens"] = None
    
    # Process each chunk using iloc to avoid index issues
    for i in range(len(aggregated)):
        chunk = aggregated.iloc[i]["chunk"]
        
        if chunk is None:
            continue
            
        if isinstance(chunk, dict):
            # Extract from dictionary format
            aggregated.iloc[i, aggregated.columns.get_loc("document_ids")] = chunk.get("document_ids", [])
            aggregated.iloc[i, aggregated.columns.get_loc("text")] = chunk.get("text", "")
            aggregated.iloc[i, aggregated.columns.get_loc("n_tokens")] = chunk.get("n_tokens", 0)
        elif isinstance(chunk, (list, tuple)) and len(chunk) >= 3:
            # Extract from tuple format
            aggregated.iloc[i, aggregated.columns.get_loc("document_ids")] = chunk[0] if isinstance(chunk[0], list) else [chunk[0]]
            aggregated.iloc[i, aggregated.columns.get_loc("text")] = chunk[1]
            aggregated.iloc[i, aggregated.columns.get_loc("n_tokens")] = chunk[2]

    # Ensure document_ids is always a list
    aggregated["document_ids"] = aggregated["document_ids"].apply(
        lambda x: [] if x is None else (x if isinstance(x, list) else [x])
    )

    # Initialize structural columns
    aggregated[PAGE_ID] = None
    aggregated[PAGE_NUMBER] = None
    aggregated[PARAGRAPH_ID] = None
    aggregated[PARAGRAPH_NUMBER] = None
    aggregated[CHAR_POSITION_START] = None
    aggregated[CHAR_POSITION_END] = None

    # Extract structural information if available
    if has_html_in_metadata:
        log.info("Extracting structural information")
        for i in range(len(aggregated)):
            chunk = aggregated.iloc[i]["chunk"]
            if isinstance(chunk, dict):
                # Map structural info from chunk to schema columns
                if "paragraph_id" in chunk:
                    aggregated.iloc[i, aggregated.columns.get_loc(PARAGRAPH_ID)] = chunk["paragraph_id"]
                if "paragraph_number" in chunk:
                    aggregated.iloc[i, aggregated.columns.get_loc(PARAGRAPH_NUMBER)] = chunk["paragraph_number"]
                if "char_position_start" in chunk:
                    aggregated.iloc[i, aggregated.columns.get_loc(CHAR_POSITION_START)] = chunk["char_position_start"]
                if "char_position_end" in chunk:
                    aggregated.iloc[i, aggregated.columns.get_loc(CHAR_POSITION_END)] = chunk["char_position_end"]

    # Create simple attributes
    aggregated["attributes"] = aggregated.apply(
        lambda row: json.dumps({
            "paragraph": {"id": row.get(PARAGRAPH_ID), "number": row.get(PARAGRAPH_NUMBER)} 
            if row.get(PARAGRAPH_ID) else {}
        }) if row.get(PARAGRAPH_ID) else "{}",
        axis=1
    )

    # Filter out rows with no text and reset index
    result = aggregated[aggregated["text"].notna()].reset_index(drop=True)

    # Performance metrics
    end_time = time.time()
    current_memory = process.memory_info().rss / 1024 / 1024
    metrics.processing_time = end_time - start_time
    metrics.memory_peak_mb = max(initial_memory, current_memory)
    metrics.total_chunks = len(result)

    callbacks.progress(Progress(
        percent=100,
        description=f"Text chunking completed: {metrics.total_chunks} chunks, {metrics.processing_time:.2f}s"
    ))

    log.info(f"Performance: {metrics.total_documents} docs -> {metrics.total_chunks} chunks in {metrics.processing_time:.2f}s")

    return result


def chunk_text_with_positions(texts: list, size: int, overlap: int, strategy: ChunkStrategyType, 
                            encode_fn, decode_fn, metadata: Any = None) -> list:
    """Enhanced chunking that tracks character positions."""
    if strategy == ChunkStrategyType.tokens:
        return chunk_texts_by_tokens_with_positions(texts, size, overlap, encode_fn, decode_fn, metadata)
    else:
        # Fallback to original implementation
        df = pd.DataFrame({"texts": [texts]})
        chunks = chunk_text(df, column="texts", size=size, overlap=overlap, 
                          encoding_model="", strategy=strategy, callbacks=WorkflowCallbacks())[0]
        
        # Convert to enhanced format
        enhanced_chunks = []
        for chunk in chunks:
            if isinstance(chunk, (list, tuple)) and len(chunk) >= 3:
                enhanced_chunks.append({
                    "document_ids": chunk[0],
                    "text": chunk[1],
                    "n_tokens": chunk[2],
                })
        return enhanced_chunks


def chunk_texts_by_tokens_with_positions(texts: list, size: int, overlap: int, 
                                       encode_fn, decode_fn, metadata: Any = None) -> list:
    """Token-based chunking that uses pre-computed HTML paragraph positions."""
    results = []
    
    # Parse metadata once if available
    parsed_metadata = None
    paragraph_positions = []
    if metadata:
        if isinstance(metadata, str):
            try:
                parsed_metadata = json.loads(metadata)
            except:
                pass
        elif isinstance(metadata, dict):
            parsed_metadata = metadata
            
        # Extract paragraph positions from HTML metadata (already computed!)
        if parsed_metadata and "html" in parsed_metadata:
            html_data = parsed_metadata["html"]
            if isinstance(html_data, dict) and "paragraphs" in html_data:
                paragraph_positions = html_data["paragraphs"]

    for doc_id, text in texts:
        if not text or pd.isna(text):
            continue

        text_str = str(text)
        tokens = encode_fn(text_str)
        
        start_idx = 0
        
        while start_idx < len(tokens):
            end_idx = min(start_idx + size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = decode_fn(chunk_tokens)
            
            # Estimate chunk character positions based on token ratios
            # This is much simpler and more reliable than text searching
            total_tokens = len(tokens)
            total_chars = len(text_str)
            
            char_start = int((start_idx / total_tokens) * total_chars) if total_tokens > 0 else 0
            char_end = int((end_idx / total_tokens) * total_chars) if total_tokens > 0 else len(chunk_text)
            
            # Find overlapping paragraphs using the pre-computed HTML positions
            structural_info = {}
            if paragraph_positions:
                for para in paragraph_positions:
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
            
            chunk_data = {
                "document_ids": [doc_id],
                "text": chunk_text,
                "n_tokens": len(chunk_tokens),
                **structural_info
            }
            
            results.append(chunk_data)

            if end_idx >= len(tokens):
                break
            start_idx += size - overlap

    return results


def extract_chunk_structural_info(chunk_text: str, char_start: int, char_end: int, metadata: dict) -> dict:
    """Simplified structural extraction - most work already done by HTML parser."""
    # This function is now much simpler since we do the extraction directly in chunk_texts_by_tokens_with_positions
    # Keep it for backward compatibility but it's mostly redundant now
    return {}


def get_cached_encoding_fn(encoding_model: str, cache_size: int = 1000):
    """Cache encoding functions for reuse."""
    @lru_cache(maxsize=cache_size)
    def _cached_get_encoding_fn(model: str):
        return get_encoding_fn(model)
    return _cached_get_encoding_fn(encoding_model)


def process_metadata_optimized(metadata: Any, line_delimiter: str, size: int, 
                             chunk_size_includes_metadata: bool, encode_fn) -> tuple[str, int]:
    """Optimized metadata processing."""
    metadata_str = ""
    metadata_tokens = 0

    try:
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {"raw": metadata}

        if isinstance(metadata, dict):
            metadata_items = []
            for k, v in metadata.items():
                if k == "html" and isinstance(v, dict):
                    # Streamlined HTML metadata (exclude large arrays)
                    html_summary = {
                        key: val for key, val in v.items()
                        if key not in ["pages", "paragraphs"] and isinstance(val, (str, int, float, bool))
                    }
                    metadata_items.append(f"{k}: {json.dumps(html_summary)}")
                else:
                    metadata_items.append(f"{k}: {v}")

            metadata_str = line_delimiter.join(metadata_items) + line_delimiter

            if chunk_size_includes_metadata and metadata_str:
                metadata_tokens = len(encode_fn(metadata_str))
                if metadata_tokens >= size:
                    raise ValueError("Metadata tokens exceeds the maximum tokens per chunk.")

    except Exception as e:
        log.warning(f"Error processing metadata: {e}")
        metadata_str = ""
        metadata_tokens = 0

    return metadata_str, metadata_tokens


def process_chunks_parallel(aggregated: pd.DataFrame, chunker_fn, batch_size: int, 
                          max_workers: int, callbacks: WorkflowCallbacks) -> pd.DataFrame:
    """Process chunks in parallel batches."""
    log.info(f"Processing {len(aggregated)} groups in parallel with {max_workers} workers")

    batches = [aggregated.iloc[i:i + batch_size].copy() for i in range(0, len(aggregated), batch_size)]
    total_batches = len(batches)
    processed_count = 0

    def process_batch(batch_df):
        return batch_df.apply(chunker_fn, axis=1)

    processed_batches = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(process_batch, batch): i for i, batch in enumerate(batches)}
        
        for future in future_to_batch:
            batch_result = future.result()
            processed_batches.append(batch_result)
            processed_count += 1
            
            progress_percent = (processed_count / total_batches) * 90
            callbacks.progress(Progress(
                percent=progress_percent,
                description=f"Processing batch {processed_count}/{total_batches}"
            ))

    callbacks.progress(Progress(percent=95, description="Combining batch results"))
    return pd.concat(processed_batches, ignore_index=True)