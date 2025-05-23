# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Enhanced LanceDB vector store with AWS S3 backend support and enterprise features."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from graphrag.vector_stores.base import BaseVectorStore

log = logging.getLogger(__name__)


class EnhancedLanceDBVectorStore(BaseVectorStore):
    """Enhanced LanceDB vector store with cloud storage and enterprise features."""
    
    def __init__(
        self,
        storage_uri: str = "./lancedb",
        storage_options: Optional[Dict[str, Any]] = None,
        vector_column: str = "vector",
        index_type: str = "IVF_PQ",
        num_partitions: int = 256,
        num_sub_vectors: int = 96,
        tables_config: Optional[Dict[str, str]] = None,
        enable_versioning: bool = False,
        enable_compression: bool = True,
        connection_timeout: int = 30,
    ):
        """Initialize Enhanced LanceDB vector store.
        
        Args:
            storage_uri: Storage URI (local path or s3://bucket-name)
            storage_options: Storage-specific options (e.g., AWS credentials)
            vector_column: Name of the vector column
            index_type: Vector index type (IVF_PQ, IVF_FLAT, HNSW)
            num_partitions: Number of IVF partitions
            num_sub_vectors: Number of PQ sub-vectors
            tables_config: Mapping of logical tables to physical table names
            enable_versioning: Enable table versioning for updates
            enable_compression: Enable data compression
            connection_timeout: Connection timeout in seconds
        """
        self.storage_uri = storage_uri
        self.storage_options = storage_options or {}
        self.vector_column = vector_column
        self.index_type = index_type
        self.num_partitions = num_partitions
        self.num_sub_vectors = num_sub_vectors
        self.enable_versioning = enable_versioning
        self.enable_compression = enable_compression
        self.connection_timeout = connection_timeout
        
        # Table configuration
        self.tables_config = tables_config or {
            "entities": "entities",
            "relationships": "relationships", 
            "text_units": "text_units",
            "embeddings": "embeddings",
            "documents": "documents",
        }
        
        self._db = None
        self._tables = {}
        
    async def connect(self):
        """Connect to LanceDB with appropriate backend."""
        try:
            import lancedb
            
            # Connect based on storage URI
            if self.storage_uri.startswith("s3://"):
                log.info(f"Connecting to LanceDB with S3 backend: {self.storage_uri}")
                self._db = await asyncio.get_event_loop().run_in_executor(
                    None, self._connect_s3, lancedb
                )
            else:
                log.info(f"Connecting to LanceDB with local storage: {self.storage_uri}")
                # Ensure directory exists for local storage
                Path(self.storage_uri).mkdir(parents=True, exist_ok=True)
                self._db = await asyncio.get_event_loop().run_in_executor(
                    None, lancedb.connect, self.storage_uri
                )
            
            # Initialize tables
            await self._initialize_tables()
            
            log.info("Successfully connected to Enhanced LanceDB")
            
        except ImportError:
            raise ImportError("LanceDB not installed. Install with: pip install lancedb")
        except Exception as e:
            log.error(f"Failed to connect to LanceDB: {e}")
            raise
    
    def _connect_s3(self, lancedb):
        """Connect to LanceDB with S3 backend (sync method for executor)."""
        try:
            # For S3, we need to configure the storage options
            s3_options = self.storage_options.copy()
            
            # Add default S3 options if not specified
            if "allow_http" not in s3_options:
                s3_options["allow_http"] = False
            
            return lancedb.connect(
                uri=self.storage_uri,
                storage_options=s3_options
            )
        except Exception as e:
            log.error(f"Failed to connect to S3 backend: {e}")
            raise
    
    async def _initialize_tables(self):
        """Initialize all required tables."""
        for logical_name, physical_name in self.tables_config.items():
            try:
                # Check if table exists
                table_names = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self._db.table_names()
                )
                
                if physical_name not in table_names:
                    log.info(f"Creating table: {physical_name}")
                    await self._create_table(logical_name, physical_name)
                else:
                    log.info(f"Using existing table: {physical_name}")
                    self._tables[logical_name] = await asyncio.get_event_loop().run_in_executor(
                        None, self._db.open_table, physical_name
                    )
                    
            except Exception as e:
                log.warning(f"Failed to initialize table {physical_name}: {e}")
    
    async def _create_table(self, logical_name: str, physical_name: str):
        """Create a table with appropriate schema."""
        # Define schemas for different table types
        schemas = {
            "entities": pa.schema([
                pa.field("id", pa.string()),
                pa.field("name", pa.string()),
                pa.field("type", pa.string()),
                pa.field("description", pa.string()),
                pa.field(self.vector_column, pa.list_(pa.float32())),
                pa.field("text_unit_ids", pa.list_(pa.string())),
                pa.field("degree", pa.int64()),
                pa.field("creation_date", pa.timestamp("us")),
            ]),
            "relationships": pa.schema([
                pa.field("id", pa.string()),
                pa.field("source", pa.string()),
                pa.field("target", pa.string()),
                pa.field("description", pa.string()),
                pa.field(self.vector_column, pa.list_(pa.float32())),
                pa.field("weight", pa.float64()),
                pa.field("text_unit_ids", pa.list_(pa.string())),
                pa.field("creation_date", pa.timestamp("us")),
            ]),
            "text_units": pa.schema([
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field(self.vector_column, pa.list_(pa.float32())),
                pa.field("document_ids", pa.list_(pa.string())),
                pa.field("entity_ids", pa.list_(pa.string())),
                pa.field("relationship_ids", pa.list_(pa.string())),
                pa.field("page_id", pa.string()),
                pa.field("page_number", pa.int64()),
                pa.field("paragraph_id", pa.string()),
                pa.field("paragraph_number", pa.int64()),
                pa.field("char_position_start", pa.int64()),
                pa.field("char_position_end", pa.int64()),
                pa.field("attributes", pa.string()),  # JSON string
                pa.field("creation_date", pa.timestamp("us")),
            ]),
            "embeddings": pa.schema([
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field(self.vector_column, pa.list_(pa.float32())),
                pa.field("model", pa.string()),
                pa.field("dimensions", pa.int32()),
                pa.field("creation_date", pa.timestamp("us")),
            ]),
            "documents": pa.schema([
                pa.field("id", pa.string()),
                pa.field("title", pa.string()),
                pa.field("file_path", pa.string()),
                pa.field("file_size", pa.int64()),
                pa.field("file_type", pa.string()),
                pa.field("source_url", pa.string()),
                pa.field("text_unit_ids", pa.list_(pa.string())),
                pa.field("creation_date", pa.timestamp("us")),
                pa.field("metadata", pa.string()),  # JSON string
            ]),
        }
        
        schema = schemas.get(logical_name)
        if not schema:
            raise ValueError(f"Unknown table type: {logical_name}")
        
        # Create empty table with schema
        empty_data = pa.table([], schema=schema)
        
        self._tables[logical_name] = await asyncio.get_event_loop().run_in_executor(
            None, self._db.create_table, physical_name, empty_data
        )
        
        log.info(f"Created table {physical_name} with schema: {[field.name for field in schema]}")
    
    async def add_entities(self, entities_df: pd.DataFrame):
        """Add entities with embeddings to the vector store."""
        try:
            if "entities" not in self._tables:
                raise ValueError("Entities table not initialized")
            
            # Ensure required columns exist
            required_columns = ["id", "name", "type", "description", self.vector_column]
            missing_columns = [col for col in required_columns if col not in entities_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert to PyArrow table
            table_data = pa.Table.from_pandas(entities_df, preserve_index=False)
            
            # Add to LanceDB table
            await asyncio.get_event_loop().run_in_executor(
                None, self._tables["entities"].add, table_data
            )
            
            # Create vector index if enough data
            await self._maybe_create_index("entities", len(entities_df))
            
            log.info(f"Added {len(entities_df)} entities to vector store")
            
        except Exception as e:
            log.error(f"Failed to add entities: {e}")
            raise
    
    async def add_text_units(self, text_units_df: pd.DataFrame):
        """Add text units with embeddings and citation metadata."""
        try:
            if "text_units" not in self._tables:
                raise ValueError("Text units table not initialized")
            
            # Ensure required columns exist
            required_columns = ["id", "text", self.vector_column]
            missing_columns = [col for col in required_columns if col not in text_units_df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert to PyArrow table
            table_data = pa.Table.from_pandas(text_units_df, preserve_index=False)
            
            # Add to LanceDB table
            await asyncio.get_event_loop().run_in_executor(
                None, self._tables["text_units"].add, table_data
            )
            
            # Create vector index if enough data
            await self._maybe_create_index("text_units", len(text_units_df))
            
            log.info(f"Added {len(text_units_df)} text units to vector store")
            
        except Exception as e:
            log.error(f"Failed to add text units: {e}")
            raise
    
    async def similarity_search(
        self,
        query_vector: List[float],
        table_name: str = "text_units",
        limit: int = 10,
        filter_conditions: Optional[str] = None,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        """Perform similarity search with optional filtering."""
        try:
            if table_name not in self._tables:
                raise ValueError(f"Table {table_name} not found")
            
            table = self._tables[table_name]
            
            # Prepare search
            search_query = table.search(query_vector, vector_column_name=self.vector_column)
            
            # Apply filters if provided
            if filter_conditions:
                search_query = search_query.where(filter_conditions)
            
            # Execute search
            results = await asyncio.get_event_loop().run_in_executor(
                None, lambda: search_query.limit(limit).to_pandas()
            )
            
            # Convert to list of dicts
            search_results = []
            for _, row in results.iterrows():
                result = {
                    "id": row["id"],
                    "text": row.get("text", ""),
                    "score": row.get("_distance", 0.0),  # LanceDB uses _distance for similarity
                }
                
                # Add citation metadata for text units
                if table_name == "text_units" and include_metadata:
                    result.update({
                        "document_ids": row.get("document_ids", []),
                        "page_id": row.get("page_id"),
                        "page_number": row.get("page_number"),
                        "paragraph_id": row.get("paragraph_id"),
                        "paragraph_number": row.get("paragraph_number"),
                        "char_position_start": row.get("char_position_start"),
                        "char_position_end": row.get("char_position_end"),
                        "attributes": row.get("attributes"),
                    })
                
                search_results.append(result)
            
            log.debug(f"Similarity search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            log.error(f"Similarity search failed: {e}")
            raise
    
    async def get_by_ids(
        self,
        ids: List[str],
        table_name: str = "text_units",
        include_vectors: bool = False,
    ) -> pd.DataFrame:
        """Retrieve records by IDs."""
        try:
            if table_name not in self._tables:
                raise ValueError(f"Table {table_name} not found")
            
            table = self._tables[table_name]
            
            # Build filter condition for IDs
            id_filter = " OR ".join([f"id = '{id_val}'" for id_val in ids])
            
            # Query table
            results = await asyncio.get_event_loop().run_in_executor(
                None, lambda: table.search().where(id_filter).to_pandas()
            )
            
            # Optionally exclude vector columns to save bandwidth
            if not include_vectors and self.vector_column in results.columns:
                results = results.drop(columns=[self.vector_column])
            
            return results
            
        except Exception as e:
            log.error(f"Failed to get records by IDs: {e}")
            raise
    
    async def _maybe_create_index(self, table_name: str, num_records: int):
        """Create vector index if we have enough records."""
        try:
            # Only create index if we have enough records
            min_records_for_index = 1000
            if num_records < min_records_for_index:
                return
            
            table = self._tables[table_name]
            
            # Check if index already exists
            existing_indices = await asyncio.get_event_loop().run_in_executor(
                None, lambda: table.list_indices()
            )
            
            # Create index if it doesn't exist
            if not any(idx.get("column") == self.vector_column for idx in existing_indices):
                log.info(f"Creating {self.index_type} index on {table_name}.{self.vector_column}")
                
                if self.index_type == "IVF_PQ":
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        table.create_index,
                        self.vector_column,
                        index_type="IVF_PQ",
                        num_partitions=self.num_partitions,
                        num_sub_vectors=self.num_sub_vectors,
                    )
                elif self.index_type == "IVF_FLAT":
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        table.create_index,
                        self.vector_column,
                        index_type="IVF_FLAT",
                        num_partitions=self.num_partitions,
                    )
                else:
                    log.warning(f"Unsupported index type: {self.index_type}")
                    
        except Exception as e:
            log.warning(f"Failed to create index on {table_name}: {e}")
    
    async def get_table_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tables."""
        stats = {}
        
        for logical_name, table in self._tables.items():
            try:
                # Get table stats
                table_stats = await asyncio.get_event_loop().run_in_executor(
                    None, lambda t=table: {
                        "num_rows": t.count_rows(),
                        "schema": str(t.schema),
                        "indices": t.list_indices(),
                    }
                )
                stats[logical_name] = table_stats
                
            except Exception as e:
                log.warning(f"Failed to get stats for table {logical_name}: {e}")
                stats[logical_name] = {"error": str(e)}
        
        return stats
    
    async def compact_tables(self):
        """Compact all tables to optimize storage and query performance."""
        for logical_name, table in self._tables.items():
            try:
                log.info(f"Compacting table: {logical_name}")
                await asyncio.get_event_loop().run_in_executor(
                    None, table.compact_files
                )
                log.info(f"Compacted table: {logical_name}")
                
            except Exception as e:
                log.warning(f"Failed to compact table {logical_name}: {e}")
    
    async def close(self):
        """Close the connection to LanceDB."""
        self._tables.clear()
        self._db = None
        log.info("Closed Enhanced LanceDB connection")