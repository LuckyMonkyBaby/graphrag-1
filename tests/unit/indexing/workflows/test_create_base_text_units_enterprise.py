# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Tests for enterprise-enhanced create_base_text_units workflow."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from graphrag.index.workflows.create_base_text_units import (
    create_base_text_units,
    get_cached_encoding_fn,
    process_metadata_optimized,
    validate_and_optimize_config,
)
from graphrag.config.models.chunking_config import ChunkStrategyType
from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks


class TestEnterpriseChunking:
    """Test enterprise features in create_base_text_units."""
    
    def test_create_base_text_units_with_enterprise_settings(self):
        """Test chunking with enterprise settings."""
        # Sample documents
        documents = pd.DataFrame({
            "id": ["doc1", "doc2", "doc3"],
            "text": [
                "This is a test document with some content.",
                "Another document with different content for testing.",
                "Third document to test enterprise chunking features."
            ]
        })
        
        callbacks = Mock(spec=WorkflowCallbacks)
        callbacks.progress = Mock()
        
        # Enterprise settings
        result = create_base_text_units(
            documents=documents,
            callbacks=callbacks,
            group_by_columns=["id"],
            size=100,
            overlap=20,
            encoding_model="cl100k_base",
            strategy=ChunkStrategyType.tokens,
            # Enterprise parameters
            batch_size=50,
            max_workers=4,
            enable_parallel=True,
            metadata_cache_size=500,
        )
        
        # Verify results
        assert len(result) > 0
        assert "text" in result.columns
        assert "n_tokens" in result.columns
        assert "id" in result.columns
        
        # Verify progress callbacks were called
        assert callbacks.progress.called
    
    def test_create_base_text_units_parallel_processing(self):
        """Test parallel processing with larger dataset."""
        # Create larger dataset to trigger parallel processing
        documents = pd.DataFrame({
            "id": [f"doc{i}" for i in range(50)],
            "text": [f"Document {i} content for testing parallel processing." for i in range(50)]
        })
        
        callbacks = Mock(spec=WorkflowCallbacks)
        callbacks.progress = Mock()
        
        result = create_base_text_units(
            documents=documents,
            callbacks=callbacks,
            group_by_columns=["id"],
            size=100,
            overlap=20,
            encoding_model="cl100k_base",
            strategy=ChunkStrategyType.tokens,
            # Settings to trigger parallel processing
            batch_size=10,
            max_workers=2,
            enable_parallel=True,
            metadata_cache_size=100,
        )
        
        # Verify parallel processing worked
        assert len(result) >= len(documents)  # Should have at least one chunk per doc
        
        # Verify progress was reported during parallel processing
        progress_calls = [call[0][0] for call in callbacks.progress.call_args_list]
        # Should have progress updates during processing
        assert len(progress_calls) > 1
    
    def test_create_base_text_units_memory_management(self):
        """Test memory management features."""
        documents = pd.DataFrame({
            "id": ["doc1", "doc2"],
            "text": [
                "Test document for memory management testing.",
                "Another document to test memory optimization."
            ]
        })
        
        callbacks = Mock(spec=WorkflowCallbacks)
        callbacks.progress = Mock()
        
        result = create_base_text_units(
            documents=documents,
            callbacks=callbacks,
            group_by_columns=["id"],
            size=100,
            overlap=20,
            encoding_model="cl100k_base",
            strategy=ChunkStrategyType.tokens,
            # Memory management settings
            batch_size=1,  # Small batches
            max_workers=1,
            enable_parallel=False,
            metadata_cache_size=10,  # Small cache
        )
        
        # Should complete successfully even with tight memory settings
        assert len(result) > 0
    
    def test_performance_metrics_collection(self):
        """Test performance metrics collection."""
        documents = pd.DataFrame({
            "id": ["doc1"],
            "text": ["Test document for performance metrics."]
        })
        
        callbacks = Mock(spec=WorkflowCallbacks)
        callbacks.progress = Mock()
        
        result = create_base_text_units(
            documents=documents,
            callbacks=callbacks,
            group_by_columns=["id"],
            size=100,
            overlap=20,
            encoding_model="cl100k_base",
            strategy=ChunkStrategyType.tokens,
        )
        
        # Check that final progress callback includes performance data
        final_call = callbacks.progress.call_args_list[-1]
        progress_obj = final_call[0][0]
        
        # Should have performance details
        assert hasattr(progress_obj, 'details')
        if progress_obj.details:
            assert 'total_documents' in progress_obj.details
            assert 'total_chunks' in progress_obj.details
            assert 'processing_time' in progress_obj.details


class TestCachingOptimizations:
    """Test caching optimizations."""
    
    def test_get_cached_encoding_fn(self):
        """Test encoding function caching."""
        # Test with different cache sizes
        encode_fn1, decode_fn1 = get_cached_encoding_fn("cl100k_base", cache_size=100)
        encode_fn2, decode_fn2 = get_cached_encoding_fn("cl100k_base", cache_size=100)
        
        # Should return functions
        assert callable(encode_fn1)
        assert callable(decode_fn1)
        assert callable(encode_fn2)
        assert callable(decode_fn2)
        
        # Test encoding
        test_text = "Hello world"
        tokens1 = encode_fn1(test_text)
        tokens2 = encode_fn2(test_text)
        
        assert tokens1 == tokens2  # Should produce same results
        assert len(tokens1) > 0
        
        # Test decoding
        decoded1 = decode_fn1(tokens1)
        decoded2 = decode_fn2(tokens2)
        
        assert decoded1 == decoded2 == test_text
    
    def test_process_metadata_optimized(self):
        """Test optimized metadata processing."""
        # Test with simple metadata
        simple_metadata = {"source": "test", "category": "document"}
        
        metadata_str, tokens = process_metadata_optimized(
            metadata=simple_metadata,
            line_delimiter=".\n",
            size=1000,
            chunk_size_includes_metadata=True,
            encode_fn=lambda x: [1] * len(x.split())  # Mock encoding
        )
        
        assert "source: test" in metadata_str
        assert "category: document" in metadata_str
        assert tokens > 0
        
        # Test with complex HTML metadata
        html_metadata = {
            "html": {
                "pages": [{"page": 1}, {"page": 2}],  # Should be excluded
                "paragraphs": [{"para": 1}],  # Should be excluded
                "doc_type": "pdf",  # Should be included
                "page_count": 2,  # Should be included
            }
        }
        
        metadata_str, tokens = process_metadata_optimized(
            metadata=html_metadata,
            line_delimiter=".\n",
            size=1000,
            chunk_size_includes_metadata=True,
            encode_fn=lambda x: [1] * len(x.split())  # Mock encoding
        )
        
        # Should include streamlined HTML metadata
        assert "doc_type" in metadata_str
        assert "page_count" in metadata_str
        # Should exclude large arrays
        assert "pages" not in metadata_str
        assert "paragraphs" not in metadata_str
        
        # Test with metadata too large
        with pytest.raises(ValueError, match="Metadata tokens exceeds"):
            process_metadata_optimized(
                metadata={"large": "x" * 1000},
                line_delimiter=".\n",
                size=10,  # Very small limit
                chunk_size_includes_metadata=True,
                encode_fn=lambda x: [1] * len(x)  # Each char = 1 token
            )


class TestErrorHandling:
    """Test error handling and recovery."""
    
    def test_create_base_text_units_with_empty_documents(self):
        """Test handling of empty documents."""
        documents = pd.DataFrame({
            "id": ["doc1", "doc2", "doc3"],
            "text": ["Valid content", "", None]  # Mix of valid, empty, and null
        })
        
        callbacks = Mock(spec=WorkflowCallbacks)
        callbacks.progress = Mock()
        
        result = create_base_text_units(
            documents=documents,
            callbacks=callbacks,
            group_by_columns=["id"],
            size=100,
            overlap=20,
            encoding_model="cl100k_base",
            strategy=ChunkStrategyType.tokens,
        )
        
        # Should handle empty/null documents gracefully
        assert len(result) > 0
        # Should filter out empty chunks
        valid_chunks = result[result["text"].notna() & (result["text"] != "")]
        assert len(valid_chunks) > 0
    
    def test_create_base_text_units_with_malformed_metadata(self):
        """Test handling of malformed metadata."""
        documents = pd.DataFrame({
            "id": ["doc1", "doc2"],
            "text": ["Document with good metadata", "Document with bad metadata"],
            "metadata": ['{"valid": "json"}', '{"invalid": json}']  # One valid, one invalid JSON
        })
        
        callbacks = Mock(spec=WorkflowCallbacks)
        callbacks.progress = Mock()
        
        # Should not raise exception even with malformed metadata
        result = create_base_text_units(
            documents=documents,
            callbacks=callbacks,
            group_by_columns=["id"],
            size=100,
            overlap=20,
            encoding_model="cl100k_base",
            strategy=ChunkStrategyType.tokens,
            prepend_metadata=True,
        )
        
        assert len(result) > 0


class TestConfigurationValidation:
    """Test configuration validation functionality."""
    
    def test_validate_and_optimize_config_from_workflow(self):
        """Test configuration validation integrated with workflow."""
        # This tests the validation function we imported
        config = {
            "chunks": {
                "batch_size": 1000,
                "max_workers": 100,  # Unrealistic
                "parallel_threshold": 5000,  # Too high
            }
        }
        
        optimized = validate_and_optimize_config(config)
        
        # Should optimize unrealistic settings
        chunks = optimized["chunks"]
        assert chunks["max_workers"] <= 32  # Should be reasonable
        assert chunks["parallel_threshold"] <= chunks["batch_size"]


class TestIntegrationWithExistingCode:
    """Test integration with existing codebase."""
    
    def test_backward_compatibility(self):
        """Test that new features don't break existing functionality."""
        documents = pd.DataFrame({
            "id": ["doc1"],
            "text": ["Simple test document"]
        })
        
        callbacks = Mock(spec=WorkflowCallbacks)
        callbacks.progress = Mock()
        
        # Call with minimal parameters (existing API)
        result = create_base_text_units(
            documents=documents,
            callbacks=callbacks,
            group_by_columns=["id"],
            size=100,
            overlap=20,
            encoding_model="cl100k_base",
            strategy=ChunkStrategyType.tokens,
        )
        
        # Should work with default enterprise parameters
        assert len(result) > 0
        assert "text" in result.columns
        assert "n_tokens" in result.columns
    
    def test_existing_column_structure(self):
        """Test that output maintains expected column structure."""
        documents = pd.DataFrame({
            "id": ["doc1"],
            "text": ["Test document content"]
        })
        
        callbacks = Mock(spec=WorkflowCallbacks)
        callbacks.progress = Mock()
        
        result = create_base_text_units(
            documents=documents,
            callbacks=callbacks,
            group_by_columns=["id"],
            size=100,
            overlap=20,
            encoding_model="cl100k_base",
            strategy=ChunkStrategyType.tokens,
        )
        
        # Verify expected columns exist
        expected_columns = {
            "id", "text", "n_tokens", "document_ids",
            "attributes", "page_id", "page_number", 
            "paragraph_id", "paragraph_number",
            "char_position_start", "char_position_end"
        }
        
        result_columns = set(result.columns)
        assert expected_columns.issubset(result_columns)