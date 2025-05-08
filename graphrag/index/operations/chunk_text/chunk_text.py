# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing _get_num_total, chunk, run_strategy and load_strategy methods definitions."""

from typing import Any, cast

import pandas as pd

from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
from graphrag.config.models.chunking_config import ChunkingConfig, ChunkStrategyType
from graphrag.index.operations.chunk_text.typing import (
    ChunkInput,
    ChunkStrategy,
)
from graphrag.logger.progress import ProgressTicker, progress_ticker


def chunk_text(
    input: pd.DataFrame,
    column: str,
    size: int,
    overlap: int,
    encoding_model: str,
    strategy: ChunkStrategyType,
    callbacks: WorkflowCallbacks,
    file_path_column: str | None = None,  # New parameter for source tracking
) -> pd.Series:
    """
    Chunk a piece of text into smaller pieces with optional source tracking.

    ## Usage
    ```yaml
    args:
        column: <column name> # The name of the column containing the text to chunk
        file_path_column: <column name> # Optional: column name containing source file paths
        strategy: <strategy config> # The strategy to use to chunk the text
    ```
    """
    strategy_exec = load_strategy(strategy)

    num_total = _get_num_total(input, column)
    tick = progress_ticker(callbacks.progress, num_total)

    # Get file paths if a column is specified
    has_file_paths = file_path_column is not None and file_path_column in input.columns
    
    # Config object
    config = ChunkingConfig(size=size, overlap=overlap, encoding_model=encoding_model)

    # Apply chunking with or without source tracking
    if has_file_paths:
        # With source tracking
        return cast(
            "pd.Series",
            input.apply(
                cast(
                    "Any",
                    lambda x: run_strategy_with_source(
                        strategy_exec,
                        x[column],
                        x[file_path_column],  # Pass file path for this row
                        config,
                        tick,
                    ),
                ),
                axis=1,
            ),
        )
    else:
        # Without source tracking (original behavior)
        return cast(
            "pd.Series",
            input.apply(
                cast(
                    "Any",
                    lambda x: run_strategy(
                        strategy_exec,
                        x[column],
                        config,
                        tick,
                    ),
                ),
                axis=1,
            ),
        )


def run_strategy(
    strategy_exec: ChunkStrategy,
    input: ChunkInput,
    config: ChunkingConfig,
    tick: ProgressTicker,
) -> list[str | tuple[list[str] | None, str, int]]:
    """Run strategy method definition (original without source tracking)."""
    if isinstance(input, str):
        # Use a placeholder file path when not tracking sources
        file_paths = ["unknown"]
        chunks = strategy_exec([input], file_paths, config, tick)
        return [chunk.text_chunk for chunk in chunks]

    # We can work with both just a list of text content
    # or a list of tuples of (document_id, text content)
    texts = [item if isinstance(item, str) else item[1] for item in input]
    file_paths = ["unknown"] * len(texts)  # Placeholder file paths
    
    strategy_results = strategy_exec(texts, file_paths, config, tick)

    results = []
    for strategy_result in strategy_results:
        doc_indices = strategy_result.source_doc_indices
        if isinstance(input[doc_indices[0]], str):
            results.append(strategy_result.text_chunk)
        else:
            doc_ids = [input[doc_idx][0] for doc_idx in doc_indices]
            results.append((
                doc_ids,
                strategy_result.text_chunk,
                strategy_result.n_tokens,
            ))
    return results


def run_strategy_with_source(
    strategy_exec: ChunkStrategy,
    input: ChunkInput,
    file_path: str,
    config: ChunkingConfig,
    tick: ProgressTicker,
) -> list[tuple[list[str] | None, str, int, Any]]:  # Added source_location to tuple
    """Run chunking strategy with source tracking."""
    if isinstance(input, str):
        chunks = strategy_exec([input], [file_path], config, tick)
        return [(None, chunk.text_chunk, chunk.n_tokens or 0, chunk.source_location) 
                for chunk in chunks]

    # Handle lists of text or tuples
    texts = [item if isinstance(item, str) else item[1] for item in input]
    
    # Create file paths for each text (using the provided path as a base)
    if len(texts) == 1:
        paths = [file_path]
    else:
        paths = [f"{file_path}_{i}" for i in range(len(texts))]
    
    chunks = strategy_exec(texts, paths, config, tick)
    
    results = []
    for chunk in chunks:
        doc_indices = chunk.source_doc_indices
        if isinstance(input[doc_indices[0]], str):
            # Just text
            results.append((
                None, 
                chunk.text_chunk, 
                chunk.n_tokens or 0,
                chunk.source_location
            ))
        else:
            # With document IDs
            doc_ids = [input[doc_idx][0] for doc_idx in doc_indices]
            results.append((
                doc_ids,
                chunk.text_chunk,
                chunk.n_tokens or 0,
                chunk.source_location
            ))
    
    return results


def load_strategy(strategy: ChunkStrategyType) -> ChunkStrategy:
    """Load strategy method definition."""
    match strategy:
        case ChunkStrategyType.tokens:
            from graphrag.index.operations.chunk_text.strategies import run_tokens

            return run_tokens
        case ChunkStrategyType.sentence:
            # NLTK
            from graphrag.index.operations.chunk_text.bootstrap import bootstrap
            from graphrag.index.operations.chunk_text.strategies import run_sentences

            bootstrap()
            return run_sentences
        case _:
            msg = f"Unknown strategy: {strategy}"
            raise ValueError(msg)


def _get_num_total(output: pd.DataFrame, column: str) -> int:
    num_total = 0
    for row in output[column]:
        if isinstance(row, str):
            num_total += 1
        else:
            num_total += len(row)
    return num_total