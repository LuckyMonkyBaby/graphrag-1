# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Cloud storage factory for multi-cloud GraphRAG deployments."""

import logging
from typing import Any, Dict, Optional

from graphrag.config.models.cloud_config import (
    CloudProvider,
    CloudStorageConfig,
    StorageBackend,
    VectorStoreBackend,
    GraphStoreBackend,
)
from graphrag.storage.pipeline_storage import PipelineStorage

log = logging.getLogger(__name__)


class CloudStorageFactory:
    """Factory for creating cloud storage instances based on configuration."""
    
    @staticmethod
    async def create_document_storage(config: CloudStorageConfig) -> PipelineStorage:
        """Create document storage backend based on configuration."""
        backend = config.document_storage
        
        if backend == StorageBackend.S3:
            return await CloudStorageFactory._create_s3_storage(config)
        elif backend == StorageBackend.AZURE_BLOB:
            return await CloudStorageFactory._create_azure_blob_storage(config)
        elif backend == StorageBackend.FILE_SYSTEM:
            return await CloudStorageFactory._create_file_storage(config)
        else:
            raise ValueError(f"Unsupported document storage backend: {backend}")
    
    @staticmethod
    async def create_vector_store(config: CloudStorageConfig) -> Any:
        """Create vector store backend based on configuration."""
        backend = config.vector_storage
        
        if backend == VectorStoreBackend.LANCEDB:
            return await CloudStorageFactory._create_lancedb_store(config)
        elif backend == VectorStoreBackend.AWS_OPENSEARCH:
            return await CloudStorageFactory._create_opensearch_store(config)
        elif backend == VectorStoreBackend.AZURE_AI_SEARCH:
            return await CloudStorageFactory._create_azure_search_store(config)
        elif backend == VectorStoreBackend.PINECONE:
            return await CloudStorageFactory._create_pinecone_store(config)
        else:
            raise ValueError(f"Unsupported vector storage backend: {backend}")
    
    @staticmethod
    async def create_graph_store(config: CloudStorageConfig) -> Any:
        """Create graph store backend based on configuration."""
        backend = config.graph_storage
        
        if backend == GraphStoreBackend.NEO4J:
            return await CloudStorageFactory._create_neo4j_store(config)
        elif backend == GraphStoreBackend.AWS_NEPTUNE:
            return await CloudStorageFactory._create_neptune_store(config)
        elif backend == GraphStoreBackend.AZURE_COSMOS_GREMLIN:
            return await CloudStorageFactory._create_cosmos_gremlin_store(config)
        elif backend == GraphStoreBackend.NETWORKX:
            return await CloudStorageFactory._create_networkx_store(config)
        else:
            raise ValueError(f"Unsupported graph storage backend: {backend}")
    
    @staticmethod
    async def create_metadata_store(config: CloudStorageConfig) -> Any:
        """Create metadata store backend based on configuration."""
        backend = config.metadata_storage
        
        if backend == StorageBackend.SQLITE:
            return await CloudStorageFactory._create_sqlite_store(config)
        elif backend == StorageBackend.POSTGRESQL:
            return await CloudStorageFactory._create_postgresql_store(config)
        elif backend == StorageBackend.RDS:
            return await CloudStorageFactory._create_rds_store(config)
        elif backend == StorageBackend.COSMOS_DB:
            return await CloudStorageFactory._create_cosmosdb_store(config)
        else:
            raise ValueError(f"Unsupported metadata storage backend: {backend}")
    
    # AWS Implementations
    @staticmethod
    async def _create_s3_storage(config: CloudStorageConfig) -> PipelineStorage:
        """Create S3-based document storage."""
        try:
            from graphrag.storage.aws_s3_pipeline_storage import S3PipelineStorage
            
            aws_config = config.aws
            if not aws_config or not aws_config.s3_bucket:
                raise ValueError("S3 bucket configuration required for S3 storage")
            
            return S3PipelineStorage(
                bucket=aws_config.s3_bucket,
                prefix=aws_config.s3_prefix,
                region=aws_config.region,
                access_key_id=aws_config.access_key_id,
                secret_access_key=aws_config.secret_access_key,
                session_token=aws_config.session_token,
            )
        except ImportError:
            raise ImportError("AWS dependencies not installed. Install with: pip install boto3")
    
    @staticmethod
    async def _create_opensearch_store(config: CloudStorageConfig) -> Any:
        """Create AWS OpenSearch vector store."""
        try:
            from graphrag.vector_stores.aws_opensearch import AWSOpenSearchVectorStore
            
            aws_config = config.aws
            if not aws_config or not aws_config.opensearch_domain:
                raise ValueError("OpenSearch domain configuration required")
            
            return AWSOpenSearchVectorStore(
                domain_endpoint=aws_config.opensearch_domain,
                region=aws_config.region,
                index_prefix=aws_config.opensearch_index_prefix,
                access_key_id=aws_config.access_key_id,
                secret_access_key=aws_config.secret_access_key,
            )
        except ImportError:
            raise ImportError("AWS OpenSearch dependencies not installed. Install with: pip install opensearch-py boto3")
    
    @staticmethod
    async def _create_neptune_store(config: CloudStorageConfig) -> Any:
        """Create AWS Neptune graph store."""
        try:
            from graphrag.graph_stores.aws_neptune import AWSNeptuneGraphStore
            
            aws_config = config.aws
            if not aws_config or not aws_config.neptune_cluster_endpoint:
                raise ValueError("Neptune cluster endpoint configuration required")
            
            return AWSNeptuneGraphStore(
                cluster_endpoint=aws_config.neptune_cluster_endpoint,
                reader_endpoint=aws_config.neptune_reader_endpoint,
                region=aws_config.region,
                access_key_id=aws_config.access_key_id,
                secret_access_key=aws_config.secret_access_key,
            )
        except ImportError:
            raise ImportError("AWS Neptune dependencies not installed. Install with: pip install gremlinpython boto3")
    
    @staticmethod
    async def _create_rds_store(config: CloudStorageConfig) -> Any:
        """Create AWS RDS metadata store."""
        try:
            from graphrag.metadata_stores.aws_rds import AWSRDSMetadataStore
            
            aws_config = config.aws
            if not aws_config or not aws_config.rds_endpoint:
                raise ValueError("RDS endpoint configuration required")
            
            return AWSRDSMetadataStore(
                endpoint=aws_config.rds_endpoint,
                database=aws_config.rds_database,
                username=aws_config.rds_username,
                password=aws_config.rds_password,
                region=aws_config.region,
            )
        except ImportError:
            raise ImportError("AWS RDS dependencies not installed. Install with: pip install psycopg2-binary boto3")
    
    # Azure Implementations
    @staticmethod
    async def _create_azure_blob_storage(config: CloudStorageConfig) -> PipelineStorage:
        """Create Azure Blob storage."""
        try:
            from graphrag.storage.blob_pipeline_storage import BlobPipelineStorage
            
            azure_config = config.azure
            if not azure_config or not azure_config.storage_account_name:
                raise ValueError("Azure storage account configuration required")
            
            return BlobPipelineStorage(
                connection_string=f"DefaultEndpointsProtocol=https;AccountName={azure_config.storage_account_name}",
                container_name=azure_config.storage_container,
            )
        except ImportError:
            raise ImportError("Azure dependencies not installed. Install with: pip install azure-storage-blob")
    
    @staticmethod
    async def _create_azure_search_store(config: CloudStorageConfig) -> Any:
        """Create Azure AI Search vector store."""
        try:
            from graphrag.vector_stores.azure_ai_search import AzureAISearchVectorStore
            
            azure_config = config.azure
            if not azure_config or not azure_config.search_service_name:
                raise ValueError("Azure AI Search service configuration required")
            
            return AzureAISearchVectorStore(
                service_name=azure_config.search_service_name,
                index_prefix=azure_config.search_index_prefix,
            )
        except ImportError:
            raise ImportError("Azure AI Search dependencies not installed. Install with: pip install azure-search-documents")
    
    @staticmethod
    async def _create_cosmos_gremlin_store(config: CloudStorageConfig) -> Any:
        """Create Azure Cosmos DB Gremlin graph store."""
        try:
            from graphrag.graph_stores.azure_cosmos_gremlin import AzureCosmosGremlinStore
            
            azure_config = config.azure
            if not azure_config or not azure_config.cosmos_account_name:
                raise ValueError("Azure Cosmos DB account configuration required")
            
            return AzureCosmosGremlinStore(
                account_name=azure_config.cosmos_account_name,
                database_name=azure_config.cosmos_database,
            )
        except ImportError:
            raise ImportError("Azure Cosmos DB dependencies not installed. Install with: pip install gremlinpython azure-cosmos")
    
    @staticmethod
    async def _create_cosmosdb_store(config: CloudStorageConfig) -> Any:
        """Create Azure Cosmos DB metadata store."""
        try:
            from graphrag.storage.cosmosdb_pipeline_storage import CosmosDBPipelineStorage
            
            azure_config = config.azure
            if not azure_config or not azure_config.cosmos_account_name:
                raise ValueError("Azure Cosmos DB account configuration required")
            
            return CosmosDBPipelineStorage(
                cosmosdb_account_url=f"https://{azure_config.cosmos_account_name}.documents.azure.com:443/",
                cosmosdb_database_name=azure_config.cosmos_database,
            )
        except ImportError:
            raise ImportError("Azure Cosmos DB dependencies not installed. Install with: pip install azure-cosmos")
    
    # LanceDB Implementations
    @staticmethod
    async def _create_lancedb_store(config: CloudStorageConfig) -> Any:
        """Create LanceDB vector store with optional cloud storage."""
        try:
            from graphrag.vector_stores.enhanced_lancedb import EnhancedLanceDBVectorStore
            
            lancedb_config = config.lancedb
            
            # Determine storage URI based on configuration
            if lancedb_config.s3_bucket:
                # Use S3 backend for LanceDB
                storage_uri = f"s3://{lancedb_config.s3_bucket}"
                if lancedb_config.s3_region:
                    storage_options = {"region": lancedb_config.s3_region}
                else:
                    storage_options = {}
            else:
                # Use local filesystem
                storage_uri = lancedb_config.storage_path
                storage_options = {}
            
            return EnhancedLanceDBVectorStore(
                storage_uri=storage_uri,
                storage_options=storage_options,
                vector_column=lancedb_config.vector_column_name,
                index_type=lancedb_config.index_type,
                num_partitions=lancedb_config.num_partitions,
                num_sub_vectors=lancedb_config.num_sub_vectors,
                tables_config={
                    "entities": lancedb_config.entities_table,
                    "relationships": lancedb_config.relationships_table,
                    "text_units": lancedb_config.text_units_table,
                    "embeddings": lancedb_config.embeddings_table,
                },
            )
        except ImportError:
            raise ImportError("LanceDB dependencies not installed. Install with: pip install lancedb")
    
    # Local/Self-hosted Implementations
    @staticmethod
    async def _create_file_storage(config: CloudStorageConfig) -> PipelineStorage:
        """Create local file storage."""
        from graphrag.storage.file_pipeline_storage import FilePipelineStorage
        return FilePipelineStorage(root_dir="./output")
    
    @staticmethod
    async def _create_sqlite_store(config: CloudStorageConfig) -> Any:
        """Create SQLite metadata store."""
        try:
            from graphrag.metadata_stores.sqlite import SQLiteMetadataStore
            return SQLiteMetadataStore(database_path="./graphrag.db")
        except ImportError:
            raise ImportError("SQLite dependencies not available")
    
    @staticmethod
    async def _create_postgresql_store(config: CloudStorageConfig) -> Any:
        """Create PostgreSQL metadata store."""
        try:
            from graphrag.metadata_stores.postgresql import PostgreSQLMetadataStore
            return PostgreSQLMetadataStore(connection_string="postgresql://localhost/graphrag")
        except ImportError:
            raise ImportError("PostgreSQL dependencies not installed. Install with: pip install psycopg2-binary")
    
    @staticmethod
    async def _create_neo4j_store(config: CloudStorageConfig) -> Any:
        """Create Neo4j graph store."""
        try:
            from graphrag.graph_stores.neo4j import Neo4jGraphStore
            return Neo4jGraphStore(uri="bolt://localhost:7687", user="neo4j", password="password")
        except ImportError:
            raise ImportError("Neo4j dependencies not installed. Install with: pip install neo4j")
    
    @staticmethod
    async def _create_networkx_store(config: CloudStorageConfig) -> Any:
        """Create NetworkX in-memory graph store."""
        from graphrag.graph_stores.networkx import NetworkXGraphStore
        return NetworkXGraphStore()
    
    @staticmethod
    async def _create_pinecone_store(config: CloudStorageConfig) -> Any:
        """Create Pinecone vector store."""
        try:
            from graphrag.vector_stores.pinecone import PineconeVectorStore
            return PineconeVectorStore()
        except ImportError:
            raise ImportError("Pinecone dependencies not installed. Install with: pip install pinecone-client")


class MultiCloudOrchestrator:
    """Orchestrator for multi-cloud deployments with failover and load balancing."""
    
    def __init__(self, config: CloudStorageConfig):
        self.config = config
        self.primary_stores = {}
        self.failover_stores = {}
        
    async def initialize(self):
        """Initialize all storage backends."""
        factory = CloudStorageFactory()
        
        # Initialize primary stores
        self.primary_stores = {
            "documents": await factory.create_document_storage(self.config),
            "vectors": await factory.create_vector_store(self.config),
            "graph": await factory.create_graph_store(self.config),
            "metadata": await factory.create_metadata_store(self.config),
        }
        
        log.info(f"Initialized {self.config.provider} cloud storage with backends:")
        log.info(f"  Documents: {self.config.document_storage}")
        log.info(f"  Vectors: {self.config.vector_storage}")
        log.info(f"  Graph: {self.config.graph_storage}")
        log.info(f"  Metadata: {self.config.metadata_storage}")
    
    async def get_storage(self, storage_type: str) -> Any:
        """Get storage backend with automatic failover."""
        try:
            return self.primary_stores[storage_type]
        except Exception as e:
            log.warning(f"Primary {storage_type} storage failed: {e}")
            
            # Attempt failover if configured
            if storage_type in self.failover_stores:
                log.info(f"Failing over to backup {storage_type} storage")
                return self.failover_stores[storage_type]
            
            raise
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all storage backends."""
        health_status = {}
        
        for store_type, store in self.primary_stores.items():
            try:
                # Attempt a simple operation to check health
                await self._check_store_health(store)
                health_status[store_type] = True
            except Exception as e:
                log.warning(f"{store_type} storage health check failed: {e}")
                health_status[store_type] = False
        
        return health_status
    
    async def _check_store_health(self, store: Any):
        """Perform health check on individual store."""
        # Implementation depends on store type
        # This is a placeholder for actual health checks
        pass