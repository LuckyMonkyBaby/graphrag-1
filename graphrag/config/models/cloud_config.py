# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Cloud provider configuration models for multi-cloud GraphRAG deployments."""

from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class CloudProvider(str, Enum):
    """Supported cloud providers."""
    
    AZURE = "azure"
    AWS = "aws"
    GCP = "gcp"
    LOCAL = "local"


class StorageBackend(str, Enum):
    """Supported storage backends."""
    
    # Azure
    AZURE_BLOB = "azure_blob"
    AZURE_DATA_LAKE = "azure_data_lake"
    COSMOS_DB = "cosmos_db"
    
    # AWS
    S3 = "s3"
    DYNAMODB = "dynamodb"
    RDS = "rds"
    REDSHIFT = "redshift"
    
    # Local/Self-hosted
    FILE_SYSTEM = "file_system"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"


class VectorStoreBackend(str, Enum):
    """Supported vector store backends."""
    
    # Cloud-native
    AZURE_AI_SEARCH = "azure_ai_search"
    AWS_OPENSEARCH = "aws_opensearch"
    PINECONE = "pinecone"
    
    # Self-hosted/Local
    LANCEDB = "lancedb"
    WEAVIATE = "weaviate"
    QDRANT = "qdrant"
    CHROMA = "chroma"
    FAISS = "faiss"


class GraphStoreBackend(str, Enum):
    """Supported graph store backends."""
    
    # Cloud-managed
    AZURE_COSMOS_GREMLIN = "azure_cosmos_gremlin"
    AWS_NEPTUNE = "aws_neptune"
    
    # Self-hosted
    NEO4J = "neo4j"
    ARANGODB = "arangodb"
    NETWORKX = "networkx"  # For small graphs


class AWSConfig(BaseModel):
    """AWS-specific configuration."""
    
    region: str = Field(default="us-east-1", description="AWS region")
    access_key_id: Optional[str] = Field(default=None, description="AWS access key ID")
    secret_access_key: Optional[str] = Field(default=None, description="AWS secret access key")
    session_token: Optional[str] = Field(default=None, description="AWS session token")
    
    # S3 Configuration
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket for document storage")
    s3_prefix: str = Field(default="graphrag/", description="S3 key prefix")
    
    # DynamoDB Configuration
    dynamodb_table_prefix: str = Field(default="graphrag_", description="DynamoDB table prefix")
    
    # OpenSearch Configuration
    opensearch_domain: Optional[str] = Field(default=None, description="OpenSearch domain endpoint")
    opensearch_index_prefix: str = Field(default="graphrag_", description="OpenSearch index prefix")
    
    # Neptune Configuration (for graph storage)
    neptune_cluster_endpoint: Optional[str] = Field(default=None, description="Neptune cluster endpoint")
    neptune_reader_endpoint: Optional[str] = Field(default=None, description="Neptune reader endpoint")
    
    # RDS Configuration (for metadata)
    rds_endpoint: Optional[str] = Field(default=None, description="RDS endpoint for metadata")
    rds_database: str = Field(default="graphrag", description="RDS database name")
    rds_username: Optional[str] = Field(default=None, description="RDS username")
    rds_password: Optional[str] = Field(default=None, description="RDS password")


class AzureConfig(BaseModel):
    """Azure-specific configuration."""
    
    subscription_id: Optional[str] = Field(default=None, description="Azure subscription ID")
    resource_group: Optional[str] = Field(default=None, description="Azure resource group")
    
    # Storage Account Configuration
    storage_account_name: Optional[str] = Field(default=None, description="Azure storage account")
    storage_container: str = Field(default="graphrag", description="Blob container name")
    
    # Cosmos DB Configuration
    cosmos_account_name: Optional[str] = Field(default=None, description="Cosmos DB account name")
    cosmos_database: str = Field(default="graphrag", description="Cosmos DB database name")
    
    # AI Search Configuration
    search_service_name: Optional[str] = Field(default=None, description="Azure AI Search service")
    search_index_prefix: str = Field(default="graphrag_", description="Search index prefix")


class LanceDBConfig(BaseModel):
    """LanceDB-specific configuration for local/self-hosted deployments."""
    
    # Storage configuration
    storage_path: str = Field(default="./lancedb", description="Local path for LanceDB storage")
    s3_bucket: Optional[str] = Field(default=None, description="S3 bucket for LanceDB cloud storage")
    s3_region: Optional[str] = Field(default=None, description="S3 region for LanceDB")
    
    # Performance configuration
    vector_column_name: str = Field(default="vector", description="Name of vector column")
    index_type: str = Field(default="IVF_PQ", description="Vector index type")
    num_partitions: int = Field(default=256, description="Number of IVF partitions")
    num_sub_vectors: int = Field(default=96, description="Number of PQ sub-vectors")
    
    # Table configuration
    entities_table: str = Field(default="entities", description="Entities table name")
    relationships_table: str = Field(default="relationships", description="Relationships table name")
    text_units_table: str = Field(default="text_units", description="Text units table name")
    embeddings_table: str = Field(default="embeddings", description="Embeddings table name")
    
    # Connection configuration
    read_consistency_level: str = Field(default="eventual", description="Read consistency level")
    write_mode: str = Field(default="append", description="Write mode for updates")


class CloudStorageConfig(BaseModel):
    """Cloud storage configuration."""
    
    provider: CloudProvider = Field(default=CloudProvider.LOCAL, description="Cloud provider")
    
    # Storage backends
    document_storage: StorageBackend = Field(default=StorageBackend.FILE_SYSTEM, description="Document storage backend")
    metadata_storage: StorageBackend = Field(default=StorageBackend.SQLITE, description="Metadata storage backend")
    vector_storage: VectorStoreBackend = Field(default=VectorStoreBackend.LANCEDB, description="Vector storage backend")
    graph_storage: GraphStoreBackend = Field(default=GraphStoreBackend.NETWORKX, description="Graph storage backend")
    
    # Provider-specific configs
    aws: Optional[AWSConfig] = Field(default=None, description="AWS configuration")
    azure: Optional[AzureConfig] = Field(default=None, description="Azure configuration")
    lancedb: LanceDBConfig = Field(default_factory=LanceDBConfig, description="LanceDB configuration")
    
    # General settings
    enable_compression: bool = Field(default=True, description="Enable data compression")
    encryption_at_rest: bool = Field(default=True, description="Enable encryption at rest")
    backup_enabled: bool = Field(default=True, description="Enable automatic backups")
    
    # Performance settings
    connection_pool_size: int = Field(default=10, description="Connection pool size")
    timeout_seconds: int = Field(default=30, description="Operation timeout")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")


class MultiCloudConfig(BaseModel):
    """Multi-cloud deployment configuration."""
    
    # Primary and failover providers
    primary_provider: CloudProvider = Field(default=CloudProvider.LOCAL, description="Primary cloud provider")
    failover_provider: Optional[CloudProvider] = Field(default=None, description="Failover cloud provider")
    
    # Cross-cloud replication
    enable_cross_cloud_backup: bool = Field(default=False, description="Enable cross-cloud backup")
    backup_providers: list[CloudProvider] = Field(default_factory=list, description="Backup cloud providers")
    
    # Load balancing
    enable_load_balancing: bool = Field(default=False, description="Enable multi-cloud load balancing")
    load_balancing_strategy: str = Field(default="round_robin", description="Load balancing strategy")
    
    # Data locality
    prefer_local_storage: bool = Field(default=True, description="Prefer storage close to compute")
    max_cross_region_latency_ms: int = Field(default=100, description="Max acceptable cross-region latency")


# Pre-configured deployment profiles
AWS_PROFILE = CloudStorageConfig(
    provider=CloudProvider.AWS,
    document_storage=StorageBackend.S3,
    metadata_storage=StorageBackend.RDS,
    vector_storage=VectorStoreBackend.AWS_OPENSEARCH,
    graph_storage=GraphStoreBackend.AWS_NEPTUNE,
    aws=AWSConfig()
)

AZURE_PROFILE = CloudStorageConfig(
    provider=CloudProvider.AZURE,
    document_storage=StorageBackend.AZURE_BLOB,
    metadata_storage=StorageBackend.COSMOS_DB,
    vector_storage=VectorStoreBackend.AZURE_AI_SEARCH,
    graph_storage=GraphStoreBackend.AZURE_COSMOS_GREMLIN,
    azure=AzureConfig()
)

LOCAL_LANCEDB_PROFILE = CloudStorageConfig(
    provider=CloudProvider.LOCAL,
    document_storage=StorageBackend.FILE_SYSTEM,
    metadata_storage=StorageBackend.SQLITE,
    vector_storage=VectorStoreBackend.LANCEDB,
    graph_storage=GraphStoreBackend.NETWORKX,
    lancedb=LanceDBConfig()
)

HYBRID_AWS_LANCEDB_PROFILE = CloudStorageConfig(
    provider=CloudProvider.AWS,
    document_storage=StorageBackend.S3,
    metadata_storage=StorageBackend.RDS,
    vector_storage=VectorStoreBackend.LANCEDB,  # LanceDB with S3 backend
    graph_storage=GraphStoreBackend.NEO4J,
    aws=AWSConfig(),
    lancedb=LanceDBConfig(s3_bucket="my-lancedb-bucket", s3_region="us-east-1")
)