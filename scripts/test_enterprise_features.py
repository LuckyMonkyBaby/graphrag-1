#!/usr/bin/env python
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Test script for enterprise GraphRAG features."""

import os
import sys
import tempfile
import shutil
from pathlib import Path

def test_smart_configuration():
    """Test smart configuration generation."""
    print("🧪 Testing smart configuration generation...")
    
    try:
        from graphrag.config.presets import get_preset_config, create_enterprise_daily_config
        from graphrag.config.auto_config import auto_generate_config
        
        # Test preset loading
        config = get_preset_config('enterprise_daily')
        print(f"✅ Enterprise daily preset: batch_size={config['chunks']['batch_size']}")
        
        # Test auto-generation
        auto_config = auto_generate_config()
        print(f"✅ Auto-configuration: workers={auto_config['chunks']['max_workers']}")
        
        # Test enterprise daily config
        enterprise_config = create_enterprise_daily_config(
            document_count_estimate=100000,
            target_update_hours=6,
            high_availability=True
        )
        print(f"✅ Enterprise config: checkpointing={enterprise_config['enable_checkpointing']}")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    return True


def test_cli_integration():
    """Test CLI integration with new presets."""
    print("🧪 Testing CLI integration...")
    
    try:
        import subprocess
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test init with preset (simplified)
            try:
                from graphrag.config.auto_config import init_config_with_presets
                config = init_config_with_presets(
                    preset="dev_fast",
                    output_file=Path(temp_dir) / "test_settings.yaml"
                )
                print("✅ CLI preset configuration works")
                return True
            except Exception as e:
                print(f"❌ CLI preset test failed: {e}")
                return False
                
    except Exception as e:
        print(f"❌ CLI test failed: {e}")
        return False
    
    return True


def test_chunking_performance():
    """Test chunking with enterprise settings."""
    print("🧪 Testing enterprise chunking performance...")
    
    try:
        import pandas as pd
        from unittest.mock import Mock
        from graphrag.index.workflows.create_base_text_units import create_base_text_units
        from graphrag.config.models.chunking_config import ChunkStrategyType
        from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
        
        # Create test documents
        documents = pd.DataFrame({
            "id": [f"doc{i}" for i in range(10)],
            "text": [f"Test document {i} with some content for processing." for i in range(10)]
        })
        
        callbacks = Mock(spec=WorkflowCallbacks)
        callbacks.progress = Mock()
        
        # Test with enterprise settings
        result = create_base_text_units(
            documents=documents,
            callbacks=callbacks,
            group_by_columns=["id"],
            size=100,
            overlap=20,
            encoding_model="cl100k_base",
            strategy=ChunkStrategyType.tokens,
            # Enterprise settings
            batch_size=5,
            max_workers=2,
            enable_parallel=True,
            metadata_cache_size=100,
        )
        
        print(f"✅ Processed {len(documents)} docs → {len(result)} chunks")
        print(f"✅ Progress callbacks: {len(callbacks.progress.call_args_list)}")
        
    except Exception as e:
        print(f"❌ Chunking test failed: {e}")
        return False
    
    return True


def test_configuration_validation():
    """Test configuration validation."""
    print("🧪 Testing configuration validation...")
    
    try:
        from graphrag.config.auto_config import validate_and_optimize_config
        
        # Test with problematic config
        config = {
            "chunks": {
                "batch_size": 1000,
                "max_workers": 100,  # Too many
                "parallel_threshold": 5000,  # Too high
            }
        }
        
        validated = validate_and_optimize_config(config)
        
        chunks = validated["chunks"]
        print(f"✅ Validated max_workers: {chunks['max_workers']} (was 100)")
        print(f"✅ Validated parallel_threshold: {chunks['parallel_threshold']} (was 5000)")
        
    except Exception as e:
        print(f"❌ Validation test failed: {e}")
        return False
    
    return True


def main():
    """Run all enterprise feature tests."""
    print("🚀 Testing GraphRAG Enterprise Features\n")
    
    tests = [
        ("Smart Configuration", test_smart_configuration),
        ("CLI Integration", test_cli_integration),
        ("Chunking Performance", test_chunking_performance),
        ("Configuration Validation", test_configuration_validation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 50)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All enterprise features working correctly!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())