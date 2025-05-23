# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Document registry for tracking file sources and metadata in enterprise GraphRAG."""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from graphrag.storage.pipeline_storage import PipelineStorage

log = logging.getLogger(__name__)


class DocumentRegistry:
    """Registry for tracking document sources and metadata for citation purposes."""

    def __init__(self, storage: PipelineStorage):
        self.storage = storage
        self._document_cache: Dict[str, Dict[str, Any]] = {}

    async def load_document_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load document metadata from storage for citation resolution."""
        try:
            # Load documents table
            documents_df = await self.storage.get("documents.parquet")
            if isinstance(documents_df, bytes):
                # If bytes, need to read as parquet
                import io

                import pyarrow.parquet as pq

                documents_df = pq.read_table(io.BytesIO(documents_df)).to_pandas()

            # Create document metadata lookup
            metadata_dict = {}
            for _, doc in documents_df.iterrows():
                doc_id = str(doc.get("id", ""))
                metadata_dict[doc_id] = {
                    "document_id": doc_id,
                    "title": doc.get("title", ""),
                    "file_path": doc.get("file_path", ""),
                    "filename": self._extract_filename(doc.get("file_path", "")),
                    "file_size": doc.get("file_size"),
                    "file_type": doc.get("file_type", ""),
                    "source_url": doc.get("source_url", ""),
                    "creation_date": doc.get("creation_date"),
                    "doc_type": self._extract_doc_type(doc.get("metadata", {})),
                }

            # Cache for future use
            self._document_cache = metadata_dict
            log.info(f"Loaded metadata for {len(metadata_dict)} documents")
            return metadata_dict

        except Exception as e:
            log.warning(f"Failed to load document metadata: {e}")
            return {}

    def get_document_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific document ID."""
        return self._document_cache.get(doc_id)

    def get_documents_metadata(self, doc_ids: list[str]) -> Dict[str, Dict[str, Any]]:
        """Get metadata for multiple document IDs."""
        return {
            doc_id: self._document_cache[doc_id]
            for doc_id in doc_ids
            if doc_id in self._document_cache
        }

    def _extract_filename(self, file_path: str) -> str:
        """Extract filename from file path."""
        if not file_path:
            return ""
        return Path(file_path).name

    def _extract_doc_type(self, metadata: Any) -> str:
        """Extract document type from metadata."""
        if isinstance(metadata, dict):
            html_meta = metadata.get("html", {})
            if isinstance(html_meta, dict):
                return html_meta.get("doc_type", "")
        elif isinstance(metadata, str):
            try:
                import json

                parsed_meta = json.loads(metadata)
                html_meta = parsed_meta.get("html", {})
                if isinstance(html_meta, dict):
                    return html_meta.get("doc_type", "")
            except (json.JSONDecodeError, KeyError):
                pass
        return ""


class EnterpriseDocumentRegistry(DocumentRegistry):
    """Enhanced document registry with enterprise features."""

    def __init__(self, storage: PipelineStorage, database_url: Optional[str] = None):
        super().__init__(storage)
        self.database_url = database_url
        self._db_connection = None

    async def register_document_with_provenance(
        self,
        file_path: str,
        doc_id: str,
        metadata: Dict[str, Any],
        source_system: Optional[str] = None,
        processing_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register document with full enterprise provenance tracking."""

        # Compute file metadata
        file_stats = self._compute_file_stats(file_path)

        doc_record = {
            "document_id": doc_id,
            "file_path": file_path,
            "filename": Path(file_path).name,
            "file_hash": file_stats.get("hash"),
            "file_size": file_stats.get("size"),
            "mime_type": file_stats.get("mime_type"),
            "file_extension": Path(file_path).suffix.lower(),
            "creation_date": metadata.get("creation_date"),
            "last_modified": file_stats.get("last_modified"),
            "index_timestamp": datetime.utcnow().isoformat(),
            "processing_version": processing_version or "1.0.0",
            "source_system": source_system,
            "access_permissions": metadata.get("permissions"),
            "retention_policy": metadata.get("retention"),
            "doc_type": metadata.get("doc_type"),
            "title": metadata.get("title", Path(file_path).stem),
        }

        # Store in cache
        self._document_cache[doc_id] = doc_record

        # Store in database if available
        if self.database_url:
            await self._store_in_database(doc_record)

        log.info(f"Registered document {doc_id} with full provenance")
        return doc_record

    async def get_document_lineage(self, doc_id: str) -> Dict[str, Any]:
        """Get full processing lineage for a document."""
        if not self.database_url:
            return self.get_document_metadata(doc_id) or {}

        # Query database for lineage information
        try:
            query = """
            SELECT d.*, 
                   COUNT(DISTINCT t.id) as chunk_count,
                   COUNT(DISTINCT e.id) as entity_count,
                   COUNT(DISTINCT r.id) as relationship_count,
                   ARRAY_AGG(DISTINCT t.id) as text_unit_ids
            FROM documents d
            LEFT JOIN text_units t ON t.document_ids @> ARRAY[d.document_id]
            LEFT JOIN entities e ON e.text_unit_ids && ARRAY(
                SELECT t.id FROM text_units t WHERE t.document_ids @> ARRAY[d.document_id]
            )
            LEFT JOIN relationships r ON r.text_unit_ids && ARRAY(
                SELECT t.id FROM text_units t WHERE t.document_ids @> ARRAY[d.document_id]
            )
            WHERE d.document_id = %s
            GROUP BY d.document_id
            """

            # Execute query (implementation depends on database type)
            result = await self._execute_query(query, [doc_id])
            return result[0] if result else {}

        except Exception as e:
            log.warning(f"Failed to get document lineage for {doc_id}: {e}")
            return self.get_document_metadata(doc_id) or {}

    def _compute_file_stats(self, file_path: str) -> Dict[str, Any]:
        """Compute file statistics for provenance tracking."""
        try:
            stat = os.stat(file_path)

            # Compute file hash for integrity checking
            import hashlib

            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)

            # Detect MIME type
            import mimetypes

            mime_type, _ = mimetypes.guess_type(file_path)

            return {
                "size": stat.st_size,
                "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "hash": hash_md5.hexdigest(),
                "mime_type": mime_type or "application/octet-stream",
            }
        except Exception as e:
            log.warning(f"Failed to compute file stats for {file_path}: {e}")
            return {}

    async def _store_in_database(self, doc_record: Dict[str, Any]):
        """Store document record in enterprise database."""
        # Implementation depends on database type (PostgreSQL, SQL Server, etc.)
        # This is a placeholder for enterprise database integration
        log.debug(
            f"Would store document record in database: {doc_record['document_id']}"
        )

    async def _execute_query(self, query: str, params: list) -> list:
        """Execute database query."""
        # Implementation depends on database type
        # This is a placeholder for enterprise database integration
        log.debug(f"Would execute query: {query} with params: {params}")
        return []


async def create_document_registry(
    storage: PipelineStorage,
    enterprise_mode: bool = False,
    database_url: Optional[str] = None,
) -> DocumentRegistry:
    """Factory function to create appropriate document registry."""
    if enterprise_mode:
        registry = EnterpriseDocumentRegistry(storage, database_url)
    else:
        registry = DocumentRegistry(storage)

    # Load existing document metadata
    await registry.load_document_metadata()
    return registry
