# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Parameterization settings for the default configuration."""

import multiprocessing as mp
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator

from graphrag.config.defaults import graphrag_config_defaults
from graphrag.config.enums import ChunkStrategyType


class ChunkingConfig(BaseModel):
    """Configuration section for chunking."""

    size: int = Field(
        description="The chunk size to use.",
        default=graphrag_config_defaults.chunks.size,
    )
    overlap: int = Field(
        description="The chunk overlap to use.",
        default=graphrag_config_defaults.chunks.overlap,
    )
    group_by_columns: list[str] = Field(
        description="The chunk by columns to use.",
        default=graphrag_config_defaults.chunks.group_by_columns,
    )
    strategy: ChunkStrategyType = Field(
        description="The chunking strategy to use.",
        default=graphrag_config_defaults.chunks.strategy,
    )
    encoding_model: str = Field(
        description="The encoding model to use.",
        default=graphrag_config_defaults.chunks.encoding_model,
    )
    prepend_metadata: bool = Field(
        description="Prepend metadata into each chunk.",
        default=graphrag_config_defaults.chunks.prepend_metadata,
    )
    chunk_size_includes_metadata: bool = Field(
        description="Count metadata in max tokens.",
        default=graphrag_config_defaults.chunks.chunk_size_includes_metadata,
    )
    
    # Performance optimization settings
    batch_size: int = Field(
        description="Batch size for parallel processing of documents.",
        default=100,
        ge=1,
        le=1000,
    )
    max_workers: int = Field(
        description="Maximum number of worker threads for parallel processing.",
        default=min(8, mp.cpu_count()),
        ge=1,
        le=32,
    )
    enable_parallel: bool = Field(
        description="Enable parallel processing for large document sets.",
        default=True,
    )
    parallel_threshold: int = Field(
        description="Minimum number of documents to trigger parallel processing.",
        default=20,
        ge=1,
    )
    metadata_cache_size: int = Field(
        description="LRU cache size for encoding functions.",
        default=1000,
        ge=10,
        le=10000,
    )
    
    # Enterprise resilience settings
    enable_checkpointing: bool = Field(
        description="Enable checkpointing for resumable processing.",
        default=True,
    )
    checkpoint_frequency: int = Field(
        description="Checkpoint frequency (number of processed batches).",
        default=10,
        ge=1,
        le=100,
    )
    enable_progress_persistence: bool = Field(
        description="Persist progress to survive restarts.",
        default=True,
    )
    
    # Quality assurance settings
    enable_quality_checks: bool = Field(
        description="Enable quality checks during chunking.",
        default=True,
    )
    min_chunk_size: int = Field(
        description="Minimum acceptable chunk size in characters.",
        default=50,
        ge=1,
        le=1000,
    )
    max_empty_chunks_ratio: float = Field(
        description="Maximum ratio of empty chunks before failing.",
        default=0.1,
        ge=0.0,
        le=1.0,
    )
    
    # Memory management
    memory_limit_gb: Optional[float] = Field(
        description="Memory limit in GB for chunking process.",
        default=None,
        ge=0.5,
        le=1024,
    )
    enable_memory_monitoring: bool = Field(
        description="Enable memory usage monitoring.",
        default=True,
    )
    gc_frequency: int = Field(
        description="Garbage collection frequency (batches).",
        default=50,
        ge=1,
        le=1000,
    )
    
    # Error handling
    max_errors_per_batch: int = Field(
        description="Maximum errors per batch before failing.",
        default=5,
        ge=0,
        le=100,
    )
    error_recovery_strategy: str = Field(
        description="Error recovery strategy (fail_fast, skip_errors, retry).",
        default="retry",
    )
    
    # Performance monitoring
    enable_performance_metrics: bool = Field(
        description="Enable detailed performance metrics collection.",
        default=True,
    )
    metrics_collection_interval: int = Field(
        description="Metrics collection interval in seconds.",
        default=30,
        ge=5,
        le=300,
    )
    
    @validator('batch_size')
    def validate_batch_size(cls, v, values):
        """Validate batch size based on available resources."""
        max_workers = values.get('max_workers', 8)
        # Ensure batch size is reasonable for the number of workers
        if v < max_workers:
            return max_workers
        return v
    
    @validator('memory_limit_gb')
    def validate_memory_limit(cls, v, values):
        """Validate memory limit is reasonable."""
        if v is not None:
            batch_size = values.get('batch_size', 100)
            # Rough estimation: each document might need ~1MB in memory
            estimated_memory_gb = (batch_size * 1) / 1024  # Convert MB to GB
            if v < estimated_memory_gb:
                raise ValueError(f"Memory limit {v}GB may be too low for batch size {batch_size}")
        return v
