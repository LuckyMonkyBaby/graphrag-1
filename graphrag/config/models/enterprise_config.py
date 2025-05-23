# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Enterprise-grade configuration models for production deployments."""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
import multiprocessing as mp


class PerformanceProfile(str, Enum):
    """Performance profiles for different deployment scenarios."""
    
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    HIGH_THROUGHPUT = "high_throughput"
    ENTERPRISE = "enterprise"


class ResilienceLevel(str, Enum):
    """Resilience levels for fault tolerance."""
    
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    MISSION_CRITICAL = "mission_critical"


class DataGovernanceConfig(BaseModel):
    """Data governance and compliance configuration."""
    
    enable_audit_logging: bool = Field(
        default=True,
        description="Enable comprehensive audit logging for compliance"
    )
    enable_data_lineage: bool = Field(
        default=True,
        description="Track data lineage for governance"
    )
    enable_pii_detection: bool = Field(
        default=True,
        description="Automatically detect and flag PII"
    )
    enable_sensitive_data_masking: bool = Field(
        default=False,
        description="Mask sensitive data in logs and outputs"
    )
    retention_policy_days: int = Field(
        default=2555,  # 7 years for enterprise compliance
        ge=30,
        le=3650,
        description="Data retention policy in days"
    )
    compliance_frameworks: List[str] = Field(
        default=["SOC2"],
        description="Compliance frameworks to adhere to (SOC2, GDPR, HIPAA, etc.)"
    )
    access_control_enabled: bool = Field(
        default=True,
        description="Enable role-based access control"
    )
    encryption_at_rest: bool = Field(
        default=True,
        description="Enable encryption at rest for all data"
    )
    encryption_in_transit: bool = Field(
        default=True,
        description="Enable encryption in transit"
    )


class PerformanceConfig(BaseModel):
    """Enterprise performance configuration."""
    
    profile: PerformanceProfile = Field(
        default=PerformanceProfile.PRODUCTION,
        description="Performance profile to use"
    )
    
    # Resource allocation
    max_cpu_cores: int = Field(
        default=min(16, mp.cpu_count()),
        ge=1,
        le=128,
        description="Maximum CPU cores to use"
    )
    max_memory_gb: int = Field(
        default=32,
        ge=1,
        le=1024,
        description="Maximum memory allocation in GB"
    )
    
    # Concurrency settings
    max_concurrent_operations: int = Field(
        default=50,
        ge=1,
        le=1000,
        description="Maximum concurrent operations"
    )
    max_concurrent_requests: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Maximum concurrent API requests"
    )
    
    # Throughput optimization
    target_documents_per_hour: int = Field(
        default=10000,
        ge=100,
        le=1000000,
        description="Target documents processed per hour"
    )
    target_queries_per_second: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Target queries per second"
    )
    
    # Resource scaling
    enable_auto_scaling: bool = Field(
        default=True,
        description="Enable automatic resource scaling"
    )
    scale_up_threshold: float = Field(
        default=0.8,
        ge=0.1,
        le=0.99,
        description="CPU/Memory threshold to scale up"
    )
    scale_down_threshold: float = Field(
        default=0.3,
        ge=0.01,
        le=0.9,
        description="CPU/Memory threshold to scale down"
    )


class ResilienceConfig(BaseModel):
    """Enterprise resilience and fault tolerance configuration."""
    
    level: ResilienceLevel = Field(
        default=ResilienceLevel.HIGH,
        description="Resilience level for fault tolerance"
    )
    
    # Retry configuration
    max_retries: int = Field(
        default=5,
        ge=0,
        le=20,
        description="Maximum retry attempts for failed operations"
    )
    retry_backoff_multiplier: float = Field(
        default=2.0,
        ge=1.0,
        le=10.0,
        description="Exponential backoff multiplier"
    )
    retry_max_wait_seconds: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="Maximum wait time between retries"
    )
    
    # Circuit breaker configuration
    enable_circuit_breaker: bool = Field(
        default=True,
        description="Enable circuit breaker pattern"
    )
    circuit_breaker_failure_threshold: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Failures before circuit breaker opens"
    )
    circuit_breaker_timeout_seconds: int = Field(
        default=60,
        ge=5,
        le=600,
        description="Circuit breaker timeout"
    )
    
    # Health checks
    enable_health_checks: bool = Field(
        default=True,
        description="Enable comprehensive health checks"
    )
    health_check_interval_seconds: int = Field(
        default=30,
        ge=5,
        le=300,
        description="Health check interval"
    )
    
    # Failover configuration
    enable_multi_region_failover: bool = Field(
        default=False,
        description="Enable multi-region failover"
    )
    failover_timeout_seconds: int = Field(
        default=300,
        ge=30,
        le=1800,
        description="Maximum time for failover"
    )
    
    # Backup and recovery
    enable_continuous_backup: bool = Field(
        default=True,
        description="Enable continuous backup"
    )
    backup_frequency_hours: int = Field(
        default=6,
        ge=1,
        le=24,
        description="Backup frequency in hours"
    )
    backup_retention_days: int = Field(
        default=90,
        ge=7,
        le=365,
        description="Backup retention in days"
    )


class MonitoringConfig(BaseModel):
    """Enterprise monitoring and observability configuration."""
    
    # Metrics collection
    enable_metrics: bool = Field(
        default=True,
        description="Enable metrics collection"
    )
    metrics_port: int = Field(
        default=9090,
        ge=1024,
        le=65535,
        description="Metrics endpoint port"
    )
    metrics_interval_seconds: int = Field(
        default=15,
        ge=1,
        le=300,
        description="Metrics collection interval"
    )
    
    # Distributed tracing
    enable_tracing: bool = Field(
        default=True,
        description="Enable distributed tracing"
    )
    trace_sampling_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Trace sampling rate (0.0-1.0)"
    )
    
    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    enable_structured_logging: bool = Field(
        default=True,
        description="Enable structured JSON logging"
    )
    log_rotation_size_mb: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Log file rotation size in MB"
    )
    log_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Log retention in days"
    )
    
    # Alerting
    enable_alerting: bool = Field(
        default=True,
        description="Enable alerting"
    )
    alert_channels: List[str] = Field(
        default=["email", "slack"],
        description="Alert channels (email, slack, webhook, etc.)"
    )
    
    # Performance thresholds for alerting
    indexing_failure_rate_threshold: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Indexing failure rate threshold for alerts"
    )
    search_latency_threshold_ms: int = Field(
        default=1000,
        ge=100,
        le=30000,
        description="Search latency threshold for alerts"
    )
    memory_usage_threshold: float = Field(
        default=0.85,
        ge=0.1,
        le=0.99,
        description="Memory usage threshold for alerts"
    )
    storage_usage_threshold: float = Field(
        default=0.80,
        ge=0.1,
        le=0.99,
        description="Storage usage threshold for alerts"
    )


class IncrementalUpdateConfig(BaseModel):
    """Configuration for incremental graph updates."""
    
    # Update strategy
    enable_incremental_updates: bool = Field(
        default=True,
        description="Enable incremental updates instead of full rebuilds"
    )
    update_frequency_hours: int = Field(
        default=6,
        ge=1,
        le=168,  # 1 week max
        description="Frequency of incremental updates in hours"
    )
    
    # Change detection
    enable_change_detection: bool = Field(
        default=True,
        description="Enable automatic change detection"
    )
    change_detection_method: str = Field(
        default="content_hash",
        description="Method for detecting changes (content_hash, timestamp, checksum)"
    )
    
    # Batch processing for updates
    update_batch_size: int = Field(
        default=1000,
        ge=10,
        le=10000,
        description="Batch size for processing updates"
    )
    max_update_workers: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Maximum workers for update processing"
    )
    
    # Delta management
    enable_delta_compression: bool = Field(
        default=True,
        description="Enable compression of delta files"
    )
    delta_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Retention period for delta files"
    )
    
    # Conflict resolution
    conflict_resolution_strategy: str = Field(
        default="latest_wins",
        description="Strategy for resolving update conflicts (latest_wins, merge, manual)"
    )
    
    # Validation
    enable_update_validation: bool = Field(
        default=True,
        description="Enable validation of incremental updates"
    )
    validation_sample_rate: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Sample rate for update validation"
    )


class EnterpriseConfig(BaseModel):
    """Complete enterprise configuration combining all aspects."""
    
    # Core enterprise settings
    deployment_name: str = Field(
        default="graphrag-enterprise",
        description="Deployment name for identification"
    )
    environment: str = Field(
        default="production",
        description="Environment name (development, staging, production)"
    )
    version: str = Field(
        default="1.0.0",
        description="Configuration version for tracking"
    )
    
    # Enterprise components
    data_governance: DataGovernanceConfig = Field(
        default_factory=DataGovernanceConfig,
        description="Data governance and compliance configuration"
    )
    performance: PerformanceConfig = Field(
        default_factory=PerformanceConfig,
        description="Performance optimization configuration"
    )
    resilience: ResilienceConfig = Field(
        default_factory=ResilienceConfig,
        description="Resilience and fault tolerance configuration"
    )
    monitoring: MonitoringConfig = Field(
        default_factory=MonitoringConfig,
        description="Monitoring and observability configuration"
    )
    incremental_updates: IncrementalUpdateConfig = Field(
        default_factory=IncrementalUpdateConfig,
        description="Incremental update configuration"
    )
    
    # SLA targets
    sla_targets: Dict[str, Any] = Field(
        default_factory=lambda: {
            "indexing_throughput_docs_per_hour": 50000,
            "indexing_latency_p99_seconds": 300,
            "search_latency_p95_ms": 500,
            "search_latency_p99_ms": 1000,
            "uptime_percentage": 99.9,
            "recovery_time_objective_minutes": 15,
            "recovery_point_objective_minutes": 5,
        },
        description="Service level agreement targets"
    )
    
    @validator('performance')
    def validate_performance_profile(cls, v, values):
        """Validate performance configuration based on profile."""
        if v.profile == PerformanceProfile.ENTERPRISE:
            # Ensure enterprise-grade settings
            if v.max_concurrent_operations < 50:
                v.max_concurrent_operations = 50
            if v.target_documents_per_hour < 10000:
                v.target_documents_per_hour = 10000
        return v
    
    @validator('resilience')
    def validate_resilience_level(cls, v, values):
        """Validate resilience configuration based on level."""
        if v.level == ResilienceLevel.MISSION_CRITICAL:
            # Ensure mission-critical settings
            v.enable_circuit_breaker = True
            v.enable_health_checks = True
            v.enable_continuous_backup = True
            if v.max_retries < 5:
                v.max_retries = 5
        return v
    
    def get_optimized_chunking_config(self) -> Dict[str, Any]:
        """Get optimized chunking configuration based on enterprise settings."""
        profile = self.performance.profile
        
        base_config = {
            "enable_parallel": True,
            "metadata_cache_size": 2000,
        }
        
        if profile == PerformanceProfile.ENTERPRISE:
            return {
                **base_config,
                "batch_size": 200,
                "max_workers": min(16, self.performance.max_cpu_cores),
                "parallel_threshold": 10,
            }
        elif profile == PerformanceProfile.HIGH_THROUGHPUT:
            return {
                **base_config,
                "batch_size": 500,
                "max_workers": min(32, self.performance.max_cpu_cores),
                "parallel_threshold": 5,
            }
        else:
            return {
                **base_config,
                "batch_size": 100,
                "max_workers": min(8, self.performance.max_cpu_cores),
                "parallel_threshold": 20,
            }