# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Smart configuration presets for enterprise deployments."""

import multiprocessing as mp
import psutil
from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum


class DeploymentScale(Enum):
    """Deployment scale classifications."""
    DEVELOPMENT = "development"      # < 1K documents
    SMALL = "small"                 # 1K - 10K documents  
    MEDIUM = "medium"               # 10K - 100K documents
    LARGE = "large"                 # 100K - 1M documents
    ENTERPRISE = "enterprise"       # 1M - 10M documents
    HYPERSCALE = "hyperscale"      # > 10M documents


@dataclass
class SystemResources:
    """Detected system resources."""
    cpu_cores: int
    memory_gb: float
    storage_type: str  # "ssd", "hdd", "cloud"
    network_speed: str  # "standard", "high", "enterprise"


def detect_system_resources() -> SystemResources:
    """Auto-detect available system resources."""
    cpu_cores = mp.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Simple storage detection
    storage_type = "ssd"  # Default assumption for modern systems
    network_speed = "standard"  # Conservative default
    
    return SystemResources(
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        storage_type=storage_type,
        network_speed=network_speed
    )


def estimate_deployment_scale(document_count: int) -> DeploymentScale:
    """Estimate deployment scale based on document count."""
    if document_count < 1000:
        return DeploymentScale.DEVELOPMENT
    elif document_count < 10000:
        return DeploymentScale.SMALL
    elif document_count < 100000:
        return DeploymentScale.MEDIUM
    elif document_count < 1000000:
        return DeploymentScale.LARGE
    elif document_count < 10000000:
        return DeploymentScale.ENTERPRISE
    else:
        return DeploymentScale.HYPERSCALE


class ConfigurationPresets:
    """Smart configuration presets that auto-tune based on environment."""
    
    @staticmethod
    def get_chunking_config(
        scale: DeploymentScale,
        resources: SystemResources,
        daily_update: bool = True
    ) -> Dict[str, Any]:
        """Get optimized chunking configuration."""
        
        # Base configuration
        config = {
            "enable_parallel": True,
            "enable_performance_metrics": True,
            "enable_checkpointing": daily_update,  # Enable for daily updates
            "error_recovery_strategy": "retry",
        }
        
        # Scale-specific optimizations
        if scale == DeploymentScale.HYPERSCALE:
            config.update({
                "batch_size": min(1000, resources.cpu_cores * 50),
                "max_workers": min(32, resources.cpu_cores),
                "parallel_threshold": 5,
                "metadata_cache_size": 5000,
                "checkpoint_frequency": 5,  # More frequent checkpoints
                "memory_limit_gb": resources.memory_gb * 0.8,
            })
        elif scale == DeploymentScale.ENTERPRISE:
            config.update({
                "batch_size": min(500, resources.cpu_cores * 25),
                "max_workers": min(16, resources.cpu_cores),
                "parallel_threshold": 10,
                "metadata_cache_size": 2000,
                "checkpoint_frequency": 10,
                "memory_limit_gb": resources.memory_gb * 0.7,
            })
        elif scale == DeploymentScale.LARGE:
            config.update({
                "batch_size": min(200, resources.cpu_cores * 10),
                "max_workers": min(8, resources.cpu_cores),
                "parallel_threshold": 20,
                "metadata_cache_size": 1000,
                "checkpoint_frequency": 20,
            })
        else:  # DEVELOPMENT, SMALL, MEDIUM
            config.update({
                "batch_size": min(100, resources.cpu_cores * 5),
                "max_workers": min(4, resources.cpu_cores),
                "parallel_threshold": 50,
                "metadata_cache_size": 500,
                "enable_checkpointing": False,  # Not needed for smaller datasets
            })
        
        return config
    
    @staticmethod
    def get_daily_update_config(
        scale: DeploymentScale,
        resources: SystemResources
    ) -> Dict[str, Any]:
        """Get configuration optimized for daily incremental updates."""
        
        base_config = ConfigurationPresets.get_chunking_config(
            scale, resources, daily_update=True
        )
        
        # Daily update specific optimizations
        daily_optimizations = {
            # More aggressive parallel processing for updates
            "parallel_threshold": max(1, base_config["parallel_threshold"] // 4),
            
            # Smaller batches for better progress tracking
            "batch_size": max(10, base_config["batch_size"] // 2),
            
            # More frequent checkpoints
            "checkpoint_frequency": max(1, base_config.get("checkpoint_frequency", 10) // 2),
            
            # Enhanced error handling for production
            "max_errors_per_batch": 2,
            "enable_progress_persistence": True,
            
            # Memory management for long-running updates
            "gc_frequency": 25,
            "enable_memory_monitoring": True,
        }
        
        return {**base_config, **daily_optimizations}
    
    @staticmethod
    def get_performance_config(
        scale: DeploymentScale,
        resources: SystemResources,
        target_latency_ms: int = 1000
    ) -> Dict[str, Any]:
        """Get performance-optimized configuration."""
        
        # Calculate optimal settings based on target latency
        if target_latency_ms <= 100:  # Ultra-low latency
            workers_multiplier = 2
            cache_multiplier = 3
        elif target_latency_ms <= 500:  # Low latency
            workers_multiplier = 1.5
            cache_multiplier = 2
        else:  # Standard latency
            workers_multiplier = 1
            cache_multiplier = 1
        
        base_config = ConfigurationPresets.get_chunking_config(scale, resources)
        
        # Performance optimizations
        perf_config = {
            "max_workers": min(64, int(base_config["max_workers"] * workers_multiplier)),
            "metadata_cache_size": int(base_config["metadata_cache_size"] * cache_multiplier),
            "enable_performance_metrics": True,
            "metrics_collection_interval": 15,  # More frequent metrics
        }
        
        return {**base_config, **perf_config}


def create_enterprise_daily_config(
    document_count_estimate: int = 100000,
    target_update_hours: int = 6,
    high_availability: bool = True
) -> Dict[str, Any]:
    """Create complete configuration for enterprise daily updates."""
    
    resources = detect_system_resources()
    scale = estimate_deployment_scale(document_count_estimate)
    
    # Get base daily update configuration
    config = ConfigurationPresets.get_daily_update_config(scale, resources)
    
    # Add enterprise-specific settings
    enterprise_settings = {
        # High availability settings
        "enable_checkpointing": high_availability,
        "enable_progress_persistence": high_availability,
        "checkpoint_frequency": 5 if high_availability else 20,
        
        # Quality assurance
        "enable_quality_checks": True,
        "min_chunk_size": 100,
        "max_empty_chunks_ratio": 0.05,
        
        # Resource management based on update window
        "memory_limit_gb": resources.memory_gb * 0.8,
        "gc_frequency": 20,
        
        # Error handling for production
        "max_errors_per_batch": 3,
        "error_recovery_strategy": "retry",
    }
    
    # Adjust batch size based on target update time
    documents_per_hour = document_count_estimate / target_update_hours
    if documents_per_hour > 10000:  # High throughput needed
        enterprise_settings["batch_size"] = min(1000, config["batch_size"] * 2)
        enterprise_settings["max_workers"] = min(32, config["max_workers"] * 2)
    
    return {**config, **enterprise_settings}


# Pre-defined configurations for common scenarios
PRESET_CONFIGS = {
    "dev_fast": {
        "chunks": {
            "batch_size": 20,
            "max_workers": 2,
            "enable_parallel": True,
            "parallel_threshold": 5,
            "metadata_cache_size": 100,
            "enable_checkpointing": False,
            "enable_performance_metrics": False,
        }
    },
    
    "enterprise_daily": {
        "chunks": create_enterprise_daily_config(
            document_count_estimate=1000000,
            target_update_hours=6,
            high_availability=True
        )
    },
    
    "hyperscale_batch": {
        "chunks": ConfigurationPresets.get_performance_config(
            DeploymentScale.HYPERSCALE,
            detect_system_resources(),
            target_latency_ms=2000
        )
    },
    
    "cost_optimized": {
        "chunks": {
            "batch_size": 50,
            "max_workers": 4,
            "enable_parallel": True,
            "parallel_threshold": 100,
            "metadata_cache_size": 500,
            "enable_checkpointing": False,
            "enable_performance_metrics": False,
            "memory_limit_gb": 4,  # Conservative memory usage
        }
    }
}


def get_preset_config(preset_name: str) -> Dict[str, Any]:
    """Get a pre-defined configuration preset."""
    if preset_name not in PRESET_CONFIGS:
        available = ", ".join(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
    
    return PRESET_CONFIGS[preset_name]


def validate_and_optimize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and auto-optimize configuration based on system resources."""
    
    resources = detect_system_resources()
    chunks_config = config.get("chunks", {})
    
    # Auto-correct common misconfigurations
    optimized = chunks_config.copy()
    
    # Ensure max_workers doesn't exceed CPU cores
    if optimized.get("max_workers", 0) > resources.cpu_cores:
        optimized["max_workers"] = resources.cpu_cores
        
    # Ensure batch_size is reasonable for memory
    batch_size = optimized.get("batch_size", 100)
    memory_limit = optimized.get("memory_limit_gb")
    if memory_limit and batch_size > memory_limit * 100:  # Rough heuristic
        optimized["batch_size"] = int(memory_limit * 100)
        
    # Ensure parallel_threshold makes sense
    if optimized.get("parallel_threshold", 20) > batch_size:
        optimized["parallel_threshold"] = max(1, batch_size // 2)
    
    return {**config, "chunks": optimized}