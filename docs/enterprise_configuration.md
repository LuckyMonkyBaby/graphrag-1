# Enterprise Configuration Guide

This guide covers advanced configuration options for enterprise-scale GraphRAG deployments with daily graph updates.

## Quick Start with Smart Presets

### 1. Auto-Optimized Configuration (Recommended)
```bash
# Auto-detect optimal settings based on your environment
poetry run graphrag init ./my-project --preset auto --environment auto
```

### 2. Enterprise Daily Updates
```bash
# Optimized for daily graph updates at enterprise scale
poetry run graphrag init ./my-project --preset enterprise_daily --environment enterprise
```

### 3. Available Presets

| Preset | Best For | Document Count | Update Frequency |
|--------|----------|----------------|------------------|
| `dev_fast` | Development & testing | < 1K | As needed |
| `enterprise_daily` | Daily production updates | 100K - 10M | Daily |
| `hyperscale_batch` | Massive batch processing | > 10M | Weekly/monthly |
| `cost_optimized` | Budget-conscious deployments | Any | Variable |

## Configuration Files

### Enterprise Daily Update Template
The system provides a pre-built configuration at `config/enterprise-daily-update.yaml`:

```yaml
chunks:
  # Optimized for daily updates
  batch_size: 200               # High throughput
  max_workers: 16               # Scale with CPU cores
  enable_checkpointing: true    # Resumable processing
  checkpoint_frequency: 5       # Frequent safety checkpoints
  
  # Quality assurance
  enable_quality_checks: true
  min_chunk_size: 100
  max_empty_chunks_ratio: 0.05
  
  # Resource management
  memory_limit_gb: 32
  enable_memory_monitoring: true
  error_recovery_strategy: "retry"
```

## Environment-Based Auto-Configuration

The system automatically detects and optimizes configuration based on:

### Environment Variables
```bash
# Set deployment context
export GRAPHRAG_ENV=enterprise          # or production, development
export GRAPHRAG_DAILY_UPDATES=true     # Enable daily update optimizations
export GRAPHRAG_HIGH_AVAILABILITY=true # Enable enterprise features
export GRAPHRAG_INPUT_DIR=/path/to/docs # Auto-estimate document count
```

### Auto-Detection Logic

1. **Document Count**: Scans input directory to estimate scale
2. **System Resources**: Detects CPU cores, memory, storage type
3. **Environment**: Determines deployment context from env vars
4. **Update Pattern**: Optimizes for batch vs incremental processing

## Performance Optimization

### Chunking Performance

The system automatically configures chunking parameters based on scale:

| Scale | Batch Size | Workers | Parallel Threshold | Cache Size |
|-------|------------|---------|-------------------|------------|
| Development | 20-50 | 2-4 | 50+ | 500 |
| Small (1K-10K) | 50-100 | 4-8 | 20-50 | 1000 |
| Large (100K-1M) | 100-200 | 8-16 | 10-20 | 2000 |
| Enterprise (1M+) | 200-500 | 16-32 | 5-10 | 5000 |

### Memory Management

Enterprise configurations include automatic memory management:
- **Memory Limits**: Automatically set based on available RAM
- **Garbage Collection**: Frequent cleanup for long-running processes
- **Memory Monitoring**: Real-time usage tracking with alerts

### Error Handling

Production-grade error handling includes:
- **Circuit Breakers**: Fail-fast on persistent errors
- **Retry Logic**: Exponential backoff with configurable limits
- **Checkpointing**: Resume from last successful point
- **Quality Checks**: Validate output quality during processing

## Daily Update Optimizations

### Incremental Processing
```yaml
# Optimized for daily updates
chunks:
  enable_checkpointing: true
  checkpoint_frequency: 5        # Checkpoint every 5 batches
  enable_progress_persistence: true
  
  # Smaller batches for better progress tracking
  batch_size: 100               # vs 500 for batch processing
  
  # More aggressive parallelization
  parallel_threshold: 10        # vs 50 for development
```

### Resource Allocation
Daily update configurations automatically:
- Use smaller batch sizes for better progress tracking
- Enable more frequent checkpointing
- Optimize memory usage for long-running processes
- Include comprehensive error recovery

## Google-Scale Enterprise Features

### Reliability
- **Multi-level checkpointing**: Process, batch, and document level
- **Automatic retry logic**: Exponential backoff with circuit breakers
- **Progress persistence**: Survives restarts and failures
- **Quality validation**: Continuous monitoring of output quality

### Performance
- **Auto-scaling**: Adapts resource usage based on load
- **Memory optimization**: Automatic garbage collection and limits
- **Cache optimization**: Multi-tier caching for hot data
- **Parallel processing**: Optimized worker allocation

### Monitoring
- **Performance metrics**: Real-time throughput and latency
- **Resource monitoring**: CPU, memory, and storage usage
- **Error tracking**: Comprehensive error categorization
- **Progress reporting**: Detailed status for long-running operations

## Configuration Validation

The system includes automatic validation:

```python
from graphrag.config.auto_config import validate_and_optimize_config

# Auto-correct common misconfigurations
config = validate_and_optimize_config(your_config)
```

### Validation Rules
- **Resource limits**: Ensures settings don't exceed system capacity
- **Batch sizing**: Validates batch size vs available memory
- **Worker allocation**: Prevents over-subscription of CPU cores
- **Cache sizing**: Optimizes cache based on available memory

## Usage Examples

### Development Setup
```bash
# Quick setup for development
poetry run graphrag init ./dev-project --preset dev_fast
```

### Production Deployment
```bash
# Enterprise production setup
export GRAPHRAG_ENV=enterprise
export GRAPHRAG_DAILY_UPDATES=true
poetry run graphrag init ./production --preset auto --environment enterprise
```

### Custom Configuration
```python
from graphrag.config.presets import create_enterprise_daily_config

# Generate custom enterprise config
config = create_enterprise_daily_config(
    document_count_estimate=5000000,    # 5M documents
    target_update_hours=4,              # 4-hour update window
    high_availability=True              # Enable HA features
)
```

### Performance Tuning
```python
from graphrag.config.presets import ConfigurationPresets
from graphrag.config.presets import DeploymentScale, detect_system_resources

# Get performance-optimized config
resources = detect_system_resources()
config = ConfigurationPresets.get_performance_config(
    scale=DeploymentScale.ENTERPRISE,
    resources=resources,
    target_latency_ms=500  # Target 500ms latency
)
```

## Best Practices

### For Daily Updates
1. **Use checkpointing**: Enable for resumable processing
2. **Monitor resources**: Set memory limits and monitoring
3. **Quality validation**: Enable quality checks for production
4. **Error handling**: Use retry strategy with circuit breakers

### For Enterprise Scale
1. **Auto-configuration**: Let the system detect optimal settings
2. **Resource allocation**: Set appropriate memory and CPU limits
3. **Monitoring**: Enable comprehensive metrics collection
4. **Validation**: Use automatic configuration validation

### For Cost Optimization
1. **Right-sizing**: Use auto-detection to avoid over-provisioning
2. **Batch processing**: Larger batches for better efficiency
3. **Cache optimization**: Appropriate cache sizes for your dataset
4. **Resource limits**: Set conservative memory and CPU limits

## Troubleshooting

### Performance Issues
- Check system resource utilization
- Validate batch size vs available memory
- Ensure parallel threshold is appropriate
- Monitor cache hit rates

### Memory Issues
- Enable memory monitoring and limits
- Increase garbage collection frequency
- Reduce batch size if needed
- Check for memory leaks in custom code

### Error Recovery
- Review error logs for patterns
- Adjust retry settings if needed
- Use checkpointing for long-running processes
- Validate input data quality

## Migration Guide

### From Basic to Enterprise Configuration
1. **Backup existing config**: Save your current settings
2. **Generate enterprise config**: Use the auto-configuration
3. **Merge settings**: Combine your customizations
4. **Test thoroughly**: Validate with a subset of your data
5. **Deploy gradually**: Roll out in stages

This enterprise configuration system provides Google-scale reliability while maintaining simplicity and ease of use.