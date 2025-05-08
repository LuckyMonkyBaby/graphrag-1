# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing 'TextChunk' model with source tracking."""
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Optional

from graphrag.config.models.chunking_config import ChunkingConfig
from graphrag.logger.progress import ProgressTicker
from graphrag.index.operations.chunk_text.source_location import SourceLocation


@dataclass
class TextChunk:
    """Text chunk class definition with source tracking."""
    text_chunk: str
    source_doc_indices: list[int]
    n_tokens: int | None = None
    source_location: Optional[SourceLocation] = None


ChunkInput = str | list[str] | list[tuple[str, str]]
"""Input to a chunking strategy. Can be a string, a list of strings, or a list of tuples of (id, text)."""

ChunkStrategy = Callable[
    [list[str], list[str], ChunkingConfig, ProgressTicker], Iterable[TextChunk]
]
"""A function that chunks text, now with file path information."""