# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for enterprise configuration features."""

import os
import pytest
from unittest.mock import patch

from graphrag.config.presets import (
    DeploymentScale,
    SystemResources,
    ConfigurationPresets,
    create_enterprise_daily_config,
    get_preset_config,
    detect_system_resources,
    estimate_deployment_scale,
)
from graphrag.config.auto_config import (
    auto_generate_config,
    detect_deployment_context,
    validate_and_optimize_config,
)


class TestDeploymentScale:
    """Test deployment scale estimation."""
    
    def test_estimate_deployment_scale(self):
        """Test deployment scale estimation based on document count."""
        assert estimate_deployment_scale(500) == DeploymentScale.DEVELOPMENT
        assert estimate_deployment_scale(5000) == DeploymentScale.SMALL
        assert estimate_deployment_scale(50000) == DeploymentScale.MEDIUM
        assert estimate_deployment_scale(500000) == DeploymentScale.LARGE
        assert estimate_deployment_scale(5000000) == DeploymentScale.ENTERPRISE
        assert estimate_deployment_scale(50000000) == DeploymentScale.HYPERSCALE


class TestSystemResources:
    """Test system resource detection."""
    
    def test_detect_system_resources(self):
        """Test system resource detection."""
        resources = detect_system_resources()
        
        assert isinstance(resources.cpu_cores, int)
        assert resources.cpu_cores > 0
        assert isinstance(resources.memory_gb, float)
        assert resources.memory_gb > 0
        assert resources.storage_type in ["ssd", "hdd", "cloud"]
        assert resources.network_speed in ["standard", "high", "enterprise"]


class TestConfigurationPresets:
    """Test configuration presets."""
    
    def test_get_chunking_config_development(self):
        """Test chunking config for development scale."""
        resources = SystemResources(
            cpu_cores=8, memory_gb=16, storage_type="ssd", network_speed="standard"
        )
        
        config = ConfigurationPresets.get_chunking_config(
            DeploymentScale.DEVELOPMENT, resources
        )
        
        assert config["enable_parallel"] is True
        assert config["batch_size"] <= 100
        assert config["max_workers"] <= 4
        assert config["enable_checkpointing"] is False  # Not needed for dev
    
    def test_get_chunking_config_enterprise(self):
        """Test chunking config for enterprise scale."""
        resources = SystemResources(
            cpu_cores=32, memory_gb=128, storage_type="ssd", network_speed="enterprise"
        )
        
        config = ConfigurationPresets.get_chunking_config(
            DeploymentScale.ENTERPRISE, resources, daily_update=True
        )
        
        assert config["enable_parallel"] is True
        assert config["batch_size"] >= 200
        assert config["max_workers"] >= 8
        assert config["enable_checkpointing"] is True  # Enabled for daily updates
        assert config["metadata_cache_size"] >= 2000
    
    def test_get_daily_update_config(self):
        """Test daily update configuration."""
        resources = SystemResources(
            cpu_cores=16, memory_gb=64, storage_type="ssd", network_speed="high"
        )
        
        config = ConfigurationPresets.get_daily_update_config(
            DeploymentScale.LARGE, resources
        )
        
        # Should have daily update optimizations
        assert config["enable_progress_persistence"] is True
        assert config["enable_memory_monitoring"] is True
        assert config["max_errors_per_batch"] == 2
        assert config["gc_frequency"] == 25
    
    def test_get_performance_config(self):
        """Test performance-optimized configuration."""
        resources = SystemResources(
            cpu_cores=16, memory_gb=64, storage_type="ssd", network_speed="high"
        )
        
        config = ConfigurationPresets.get_performance_config(
            DeploymentScale.LARGE, resources, target_latency_ms=100  # Ultra-low latency
        )
        
        # Should have performance optimizations
        assert config["enable_performance_metrics"] is True
        assert config["metrics_collection_interval"] == 15
        # Workers should be increased for low latency
        base_config = ConfigurationPresets.get_chunking_config(DeploymentScale.LARGE, resources)
        assert config["max_workers"] >= base_config["max_workers"]


class TestEnterpriseConfiguration:
    """Test enterprise configuration generation."""
    
    def test_create_enterprise_daily_config(self):
        """Test enterprise daily configuration creation."""
        config = create_enterprise_daily_config(
            document_count_estimate=1000000,
            target_update_hours=6,
            high_availability=True
        )
        
        # Should have enterprise features
        assert config["enable_checkpointing"] is True
        assert config["enable_progress_persistence"] is True
        assert config["enable_quality_checks"] is True
        assert config["checkpoint_frequency"] == 5  # Frequent for HA
        assert config["min_chunk_size"] == 100  # Higher threshold
        assert config["max_empty_chunks_ratio"] == 0.05  # Strict quality
        assert config["max_errors_per_batch"] == 3  # Low tolerance
    
    def test_create_enterprise_daily_config_high_throughput(self):
        """Test enterprise config for high throughput scenarios."""
        config = create_enterprise_daily_config(
            document_count_estimate=10000000,  # 10M documents
            target_update_hours=4,  # Short window = high throughput needed
            high_availability=True
        )
        
        # Should optimize for high throughput
        assert config["batch_size"] >= 200  # Larger batches
        assert config["max_workers"] >= 8  # More workers


class TestPresetConfigs:
    """Test preset configuration loading."""
    
    def test_get_preset_config_dev_fast(self):
        """Test dev_fast preset."""
        config = get_preset_config("dev_fast")
        
        chunks = config["chunks"]
        assert chunks["batch_size"] <= 50
        assert chunks["max_workers"] <= 4
        assert chunks["enable_checkpointing"] is False
        assert chunks["enable_performance_metrics"] is False
    
    def test_get_preset_config_enterprise_daily(self):
        """Test enterprise_daily preset."""
        config = get_preset_config("enterprise_daily")
        
        chunks = config["chunks"]
        assert chunks["enable_checkpointing"] is True
        assert chunks["enable_progress_persistence"] is True
        assert chunks["enable_quality_checks"] is True
        assert chunks["batch_size"] >= 100
    
    def test_get_preset_config_invalid(self):
        """Test invalid preset name."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset_config("invalid_preset")


class TestAutoConfiguration:
    """Test automatic configuration generation."""
    
    @patch.dict(os.environ, {}, clear=True)
    def test_detect_deployment_context_defaults(self):
        """Test deployment context detection with defaults."""
        context = detect_deployment_context()
        
        assert context["environment"] == "development"
        assert context["scale"] == "small"
        assert context["daily_updates"] is False
        assert context["high_availability"] is False
    
    @patch.dict(os.environ, {
        "GRAPHRAG_ENV": "enterprise",
        "GRAPHRAG_DAILY_UPDATES": "true",
        "GRAPHRAG_HIGH_AVAILABILITY": "yes"
    })
    def test_detect_deployment_context_enterprise(self):
        """Test deployment context detection for enterprise."""
        context = detect_deployment_context()
        
        assert context["environment"] == "enterprise"
        assert context["daily_updates"] is True
        assert context["high_availability"] is True
        assert context["document_count_estimate"] == 1000000  # Enterprise default
    
    def test_auto_generate_config_development(self):
        """Test auto-generation for development environment."""
        with patch("graphrag.config.auto_config.detect_deployment_context") as mock_context:
            mock_context.return_value = {
                "environment": "development",
                "scale": "small",
                "daily_updates": False,
                "high_availability": False,
                "document_count_estimate": 1000,
            }
            
            config = auto_generate_config()
            
            # Should use development preset
            chunks = config["chunks"]
            assert chunks["batch_size"] <= 100
            assert chunks["enable_checkpointing"] is False
    
    def test_auto_generate_config_enterprise(self):
        """Test auto-generation for enterprise environment."""
        with patch("graphrag.config.auto_config.detect_deployment_context") as mock_context:
            mock_context.return_value = {
                "environment": "enterprise",
                "scale": "enterprise",
                "daily_updates": True,
                "high_availability": True,
                "document_count_estimate": 1000000,
            }
            
            config = auto_generate_config()
            
            # Should use enterprise configuration
            chunks = config["chunks"]
            assert chunks["enable_checkpointing"] is True
            assert chunks["enable_progress_persistence"] is True
            assert chunks["batch_size"] >= 100
    
    def test_auto_generate_config_with_existing(self):
        """Test auto-generation with existing configuration."""
        existing_config = {
            "chunks": {
                "size": 2000,  # Custom setting
                "overlap": 300,
            },
            "llm": {
                "model": "gpt-4o",
            }
        }
        
        config = auto_generate_config(existing_config=existing_config)
        
        # Should preserve existing settings
        assert config["chunks"]["size"] == 2000
        assert config["chunks"]["overlap"] == 300
        assert config["llm"]["model"] == "gpt-4o"
        
        # Should add optimized settings
        assert "batch_size" in config["chunks"]
        assert "max_workers" in config["chunks"]


class TestConfigurationValidation:
    """Test configuration validation and optimization."""
    
    def test_validate_and_optimize_config_basic(self):
        """Test basic configuration validation."""
        config = {
            "chunks": {
                "batch_size": 100,
                "max_workers": 4,
                "parallel_threshold": 20,
            }
        }
        
        validated = validate_and_optimize_config(config)
        
        # Should return optimized configuration
        assert "chunks" in validated
        chunks = validated["chunks"]
        assert chunks["batch_size"] == 100
        assert chunks["max_workers"] <= 32  # Should be reasonable
    
    @patch("graphrag.config.presets.detect_system_resources")
    def test_validate_and_optimize_config_corrections(self, mock_resources):
        """Test configuration validation with corrections."""
        mock_resources.return_value = SystemResources(
            cpu_cores=4, memory_gb=8, storage_type="ssd", network_speed="standard"
        )
        
        config = {
            "chunks": {
                "batch_size": 1000,
                "max_workers": 16,  # Too many for 4 cores
                "parallel_threshold": 2000,  # Higher than batch size
                "memory_limit_gb": 0.1,  # Too low for batch size
            }
        }
        
        validated = validate_and_optimize_config(config)
        
        chunks = validated["chunks"]
        # Should correct max_workers to match CPU cores
        assert chunks["max_workers"] == 4
        # Should adjust parallel_threshold (may be corrected to a reasonable value)
        assert chunks["parallel_threshold"] <= 1000  # Should be reasonable
    
    def test_validate_and_optimize_config_no_chunks(self):
        """Test validation with config that has no chunks section."""
        config = {
            "llm": {
                "model": "gpt-4o",
            }
        }
        
        validated = validate_and_optimize_config(config)
        
        # Should preserve other sections
        assert config["llm"]["model"] == "gpt-4o"
        # Should add empty chunks section
        assert "chunks" in validated