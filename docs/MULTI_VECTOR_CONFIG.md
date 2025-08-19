# Multi-Vector Configuration Guide

This document describes the enhanced configuration settings for DFS Prophet's multi-vector architecture.

## Overview

The multi-vector configuration allows you to customize:
- Vector type definitions and dimensions
- Embedding model settings per vector type
- Search weight configurations for vector fusion
- Performance thresholds per vector type
- Collection naming conventions

## Environment Variables

### Vector Type Configuration

```bash
# Vector type names
VECTOR_TYPE_STATS_VECTOR_NAME=stats
VECTOR_TYPE_CONTEXT_VECTOR_NAME=context
VECTOR_TYPE_VALUE_VECTOR_NAME=value
VECTOR_TYPE_COMBINED_VECTOR_NAME=combined

# Vector dimensions per type
VECTOR_TYPE_STATS_VECTOR_DIMENSIONS=768
VECTOR_TYPE_CONTEXT_VECTOR_DIMENSIONS=768
VECTOR_TYPE_VALUE_VECTOR_DIMENSIONS=768
VECTOR_TYPE_COMBINED_VECTOR_DIMENSIONS=768

# Collection naming conventions
VECTOR_TYPE_MULTI_VECTOR_REGULAR_COLLECTION=dfs_players_multi_regular
VECTOR_TYPE_MULTI_VECTOR_QUANTIZED_COLLECTION=dfs_players_multi_quantized
```

### Multi-Vector Search Settings

```bash
# Vector fusion weights (must sum to approximately 1.0)
MULTI_VECTOR_VECTOR_WEIGHT_STATS=0.4
MULTI_VECTOR_VECTOR_WEIGHT_CONTEXT=0.3
MULTI_VECTOR_VECTOR_WEIGHT_VALUE=0.3

# Search configuration
MULTI_VECTOR_ENABLE_VECTOR_FUSION=true
MULTI_VECTOR_MAX_VECTORS_PER_SEARCH=3

# Fusion strategies
MULTI_VECTOR_FUSION_STRATEGY=weighted_average
MULTI_VECTOR_MIN_FUSION_SCORE=0.1

# Performance thresholds per vector type
MULTI_VECTOR_STATS_PERFORMANCE_THRESHOLD=0.7
MULTI_VECTOR_CONTEXT_PERFORMANCE_THRESHOLD=0.6
MULTI_VECTOR_VALUE_PERFORMANCE_THRESHOLD=0.5
```

### Embedding Model Per Type Settings

```bash
# Model settings for different vector types
EMBEDDING_MODEL_STATS_MODEL_NAME=BAAI/bge-base-en-v1.5
EMBEDDING_MODEL_CONTEXT_MODEL_NAME=BAAI/bge-base-en-v1.5
EMBEDDING_MODEL_VALUE_MODEL_NAME=BAAI/bge-base-en-v1.5
EMBEDDING_MODEL_COMBINED_MODEL_NAME=BAAI/bge-base-en-v1.5

# Model-specific parameters
EMBEDDING_MODEL_STATS_MODEL_MAX_LENGTH=512
EMBEDDING_MODEL_CONTEXT_MODEL_MAX_LENGTH=512
EMBEDDING_MODEL_VALUE_MODEL_MAX_LENGTH=512
EMBEDDING_MODEL_COMBINED_MODEL_MAX_LENGTH=512

# Cache settings per model
EMBEDDING_MODEL_ENABLE_MODEL_CACHING=true
EMBEDDING_MODEL_CACHE_TTL_HOURS=24
```

## Configuration Validation

The system automatically validates:

1. **Vector Dimensions**: Must be divisible by 64 for optimal performance
2. **Weight Sum**: Vector weights should sum to approximately 1.0
3. **Performance Thresholds**: Must be between 0.0 and 1.0
4. **Model Configurations**: Model names must be valid and max_length within range

## Usage Examples

### Accessing Configuration in Code

```python
from dfs_prophet.config import get_settings

settings = get_settings()

# Get vector names
vector_names = settings.get_vector_names()
# {'stats': 'stats', 'context': 'context', 'value': 'value', 'combined': 'combined'}

# Get vector dimensions
dimensions = settings.get_vector_dimensions()
# {'stats': 768, 'context': 768, 'value': 768, 'combined': 768}

# Get fusion weights
weights = settings.get_vector_weights()
# {'stats': 0.4, 'context': 0.3, 'value': 0.3}

# Get performance thresholds
thresholds = settings.get_performance_thresholds()
# {'stats': 0.7, 'context': 0.6, 'value': 0.5}

# Get model configurations
model_configs = settings.get_model_configs()
# {'stats': {'model_name': 'BAAI/bge-base-en-v1.5', 'max_length': 512}, ...}

# Validate configuration
is_valid = settings.validate_multi_vector_config()
```

### Collection Name Helpers

```python
# Get multi-vector collection names
regular_collection = settings.get_multi_vector_collection_name("players", quantized=False)
# Returns: "dfs_players_multi_regular"

quantized_collection = settings.get_multi_vector_collection_name("players", quantized=True)
# Returns: "dfs_players_multi_quantized"
```

## Fusion Strategies

Available fusion strategies for combining multiple vectors:

- `weighted_average`: Weighted combination of vector scores
- `max_score`: Take the maximum score across all vectors
- `min_score`: Take the minimum score across all vectors
- `product`: Multiply scores from all vectors

## Performance Considerations

1. **Vector Dimensions**: Keep dimensions divisible by 64 for optimal performance
2. **Model Caching**: Enable model caching for repeated embedding generation
3. **Batch Processing**: Use appropriate batch sizes for your hardware
4. **Quantization**: Use binary quantization for memory efficiency in production

## Production Recommendations

```bash
# Production settings
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# Optimize for performance
VECTOR_TYPE_STATS_VECTOR_DIMENSIONS=768
VECTOR_TYPE_CONTEXT_VECTOR_DIMENSIONS=768
VECTOR_TYPE_VALUE_VECTOR_DIMENSIONS=768

# Balance weights for your use case
MULTI_VECTOR_VECTOR_WEIGHT_STATS=0.5
MULTI_VECTOR_VECTOR_WEIGHT_CONTEXT=0.3
MULTI_VECTOR_VECTOR_WEIGHT_VALUE=0.2

# Enable caching and quantization
EMBEDDING_MODEL_ENABLE_MODEL_CACHING=true
BINARY_QUANTIZATION_ENABLED=true
```
