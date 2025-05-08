# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module containing chunk strategies with source tracking."""
from collections.abc import Iterable
import nltk
import tiktoken

from graphrag.config.models.chunking_config import ChunkingConfig
from graphrag.index.operations.chunk_text.typing import TextChunk, SourceLocation
from graphrag.index.text_splitting.text_splitting import (
    Tokenizer,
    split_multiple_texts_on_tokens,
)
from graphrag.logger.progress import ProgressTicker


def get_encoding_fn(encoding_name):
    """Get the encoding model."""
    enc = tiktoken.get_encoding(encoding_name)
    
    def encode(text: str) -> list[int]:
        if not isinstance(text, str):
            text = f"{text}"
        return enc.encode(text)
    
    def decode(tokens: list[int]) -> str:
        return enc.decode(tokens)
    
    return encode, decode


def track_source_position(original_text: str, chunk_text: str) -> SourceLocation:
    """Calculate source position information for a chunk."""
    start_pos = original_text.find(chunk_text)
    if start_pos == -1:
        # Fallback if exact match not found
        return SourceLocation(
            file_path="unknown",
            start_line=0,
            end_line=0
        )
    
    end_pos = start_pos + len(chunk_text)
    
    # Count lines up to start_pos
    start_line = original_text[:start_pos].count('\n') + 1
    # Count lines within the chunk
    end_line = start_line + chunk_text.count('\n')
    
    return SourceLocation(
        file_path="",  # Will be filled in later
        start_line=start_line,
        end_line=end_line,
        start_char=start_pos,
        end_char=end_pos
    )


def split_multiple_texts_on_tokens_with_source(
    texts: list[str],
    file_paths: list[str],
    tokenizer: Tokenizer,
    tick: ProgressTicker
) -> list[TextChunk]:
    """Split texts and track source locations."""
    result = []
    
    for doc_idx, (text, file_path) in enumerate(zip(texts, file_paths)):
        input_ids = tokenizer.encode(text)
        if tick:
            tick(1)
        
        start_idx = 0
        while start_idx < len(input_ids):
            cur_idx = min(start_idx + tokenizer.tokens_per_chunk, len(input_ids))
            chunk_ids = input_ids[start_idx:cur_idx]
            chunk_text = tokenizer.decode(chunk_ids)
            
            # Track source location
            source_loc = track_source_position(text, chunk_text)
            source_loc.file_path = file_path
            
            result.append(TextChunk(
                text_chunk=chunk_text,
                source_doc_indices=[doc_idx],
                n_tokens=len(chunk_ids),
                source_location=source_loc
            ))
            
            if cur_idx == len(input_ids):
                break
                
            start_idx += tokenizer.tokens_per_chunk - tokenizer.chunk_overlap
    
    return result


def run_tokens(
    input: list[str],
    file_paths: list[str],
    config: ChunkingConfig,
    tick: ProgressTicker,
) -> Iterable[TextChunk]:
    """Chunks text with source tracking."""
    encode, decode = get_encoding_fn(config.encoding_model)
    
    return split_multiple_texts_on_tokens_with_source(
        input,
        file_paths,
        Tokenizer(
            chunk_overlap=config.overlap,
            tokens_per_chunk=config.size,
            encode=encode,
            decode=decode,
        ),
        tick,
    )


def run_sentences(
    input: list[str],
    file_paths: list[str],
    config: ChunkingConfig,
    tick: ProgressTicker
) -> Iterable[TextChunk]:
    """Chunks text into sentences with source tracking."""
    result = []
    
    for doc_idx, (text, file_path) in enumerate(zip(input, file_paths)):
        sentences = nltk.sent_tokenize(text)
        
        for sentence in sentences:
            # Calculate source position
            source_loc = track_source_position(text, sentence)
            source_loc.file_path = file_path
            
            result.append(TextChunk(
                text_chunk=sentence,
                source_doc_indices=[doc_idx],
                n_tokens=None,
                source_location=source_loc
            ))
        
        tick(1)
    
    return result