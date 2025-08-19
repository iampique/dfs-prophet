# Multi-Vector Health Checks

DFS Prophet's enhanced health check system provides comprehensive monitoring of the multi-vector system, including collection status, performance metrics, data quality, and automated recommendations.

## Overview

The multi-vector health check system extends the basic health monitoring with specialized endpoints for vector-specific monitoring, performance tracking, and automated optimization recommendations.

## Endpoints

### 1. Multi-Vector System Health

**Endpoint:** `GET /health/vectors`

Comprehensive health check for the entire multi-vector system.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "response_time_ms": 245.6,
  "health_score": 87.5,
  "vector_system": {
    "vector_collections": {
      "name": "vector_collections",
      "healthy": true,
      "health_percentage": 75.0,
      "collections": {
        "stats_regular": {
          "exists": true,
          "points_count": 1000,
          "vectors_count": 1000,
          "status": "healthy"
        },
        "stats_quantized": {
          "exists": true,
          "points_count": 1000,
          "vectors_count": 1000,
          "status": "healthy"
        }
      },
      "summary": {
        "total_collections": 8,
        "healthy_collections": 6,
        "empty_collections": 2,
        "error_collections": 0
      }
    },
    "cross_vector_consistency": {
      "name": "cross_vector_consistency",
      "healthy": true,
      "consistency_score": 95.2,
      "regular_points": 1000,
      "quantized_points": 1000,
      "consistency_issues": []
    }
  },
  "recommendations": [
    "Vector system is healthy. Continue monitoring for optimal performance."
  ]
}
```

### 2. Specific Vector Type Health

**Endpoint:** `GET /health/vectors/{vector_type}`

Health check for a specific vector type (stats, context, value, combined).

**Parameters:**
- `vector_type`: The vector type to check

**Response:**
```json
{
  "vector_type": "stats",
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "response_time_ms": 45.2,
  "details": {
    "name": "vector_type_stats",
    "healthy": true,
    "vector_type": "stats",
    "checks": {
      "collection_stats": {
        "regular": {
          "points_count": 1000,
          "vectors_count": 1000
        },
        "quantized": {
          "points_count": 1000,
          "vectors_count": 1000
        }
      },
      "points_count": 1000,
      "vectors_count": 1000,
      "status": "healthy"
    }
  }
}
```

### 3. Performance Health Check

**Endpoint:** `GET /health/performance`

Multi-vector performance metrics and optimization recommendations.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "response_time_ms": 123.4,
  "performance_metrics": {
    "search_analytics": {
      "total_searches": 1500,
      "average_latency": 45.2,
      "strategy_performance": {
        "stats": {
          "latency": 42.1,
          "count": 500
        },
        "context": {
          "latency": 38.5,
          "count": 400
        }
      }
    }
  },
  "memory_breakdown": {
    "total_mb": 16384,
    "used_mb": 8192,
    "percentage": 50.0,
    "vector_breakdown": {
      "stats": 512,
      "context": 384,
      "value": 256,
      "combined": 128
    }
  },
  "performance_trends": {
    "latency_trend": "stable",
    "memory_trend": "stable",
    "search_volume_trend": "increasing"
  },
  "optimization_recommendations": [
    "Performance is within acceptable ranges. Continue monitoring."
  ]
}
```

## Health Check Components

### 1. Vector Collections

Checks the status of all vector collections:

- **Regular Collections**: Standard vector storage
- **Quantized Collections**: Binary quantized storage for performance
- **Collection Health**: Points count, vectors count, existence status
- **Health Percentage**: Overall collection health score

### 2. Cross-Vector Consistency

Validates consistency between different vector types:

- **Point Count Consistency**: Ensures regular and quantized collections have similar point counts
- **Dimension Consistency**: Validates vector dimensions across collections
- **Consistency Score**: Percentage-based consistency measurement
- **Issues Detection**: Identifies specific consistency problems

### 3. Vector Performance

Monitors search performance per vector type:

- **Latency Tracking**: Response time per vector type
- **Search Count**: Number of searches performed
- **Performance Thresholds**: 200ms latency threshold for "healthy" status
- **Performance Trends**: Historical performance analysis

### 4. Data Quality Metrics

Assesses data quality for each vector type:

- **Completeness**: Percentage of complete data records
- **Accuracy**: Data accuracy measurements
- **Consistency**: Data consistency across vector types
- **Quality Score**: Overall quality assessment (85% threshold)

### 5. Memory Usage Breakdown

Monitors memory usage by vector type:

- **System Memory**: Total, used, and available memory
- **Vector Memory**: Estimated memory usage per vector type
- **Memory Thresholds**: 50% total memory usage threshold
- **Memory Trends**: Historical memory usage patterns

### 6. Embedding Models

Checks embedding model status:

- **Model Loading**: Verifies models are loaded and accessible
- **Model Names**: Tracks which models are being used
- **Model Health**: Overall model availability status

### 7. Search Functionality

Tests actual search functionality:

- **Regular Search**: Tests search on regular collections
- **Quantized Search**: Tests search on quantized collections
- **Search Success Rate**: Percentage of successful searches
- **Error Detection**: Identifies search failures

### 8. Vector Analytics

Validates analytics functionality:

- **Profile Analyzer**: Player profile analysis capabilities
- **Archetype Classification**: Player archetype classification
- **Similarity Analysis**: Multi-dimensional similarity analysis

## Health Scoring

### Calculation Method

The health score is calculated as a percentage of healthy checks:

```python
health_score = (healthy_checks / total_checks) * 100
```

### Score Interpretation

- **90-100%**: Excellent - System is performing optimally
- **75-89%**: Good - Minor issues detected, monitor closely
- **50-74%**: Warning - Some components degraded, attention needed
- **0-49%**: Critical - Multiple failures, immediate action required

## Recommendations System

### Automated Recommendations

The system generates recommendations based on health check results:

#### Critical Issues (Health Score < 50%)
- "Critical: Multiple vector system components are failing. Immediate attention required."

#### Warning Issues (Health Score < 75%)
- "Warning: Some vector system components are degraded. Review and optimize."

#### Specific Component Issues
- **Vector Collections**: "Fix vector collections: Some collections are empty or have errors."
- **Cross-Vector Consistency**: "Address cross-vector consistency issues."
- **Performance**: "Optimize vector search performance."
- **Data Quality**: "Improve vector data quality metrics."

#### Performance Recommendations
- **High Memory Usage**: "High memory usage detected. Consider optimizing vector storage."
- **High Latency**: "High search latency detected. Consider using quantized collections."
- **Increasing Latency**: "Latency trend is increasing. Monitor and optimize search algorithms."

## Usage Examples

### Basic Health Monitoring

```python
import aiohttp
import asyncio

async def check_vector_health():
    async with aiohttp.ClientSession() as session:
        # Check overall vector health
        async with session.get("http://localhost:8000/health/vectors") as response:
            health_data = await response.json()
            
        print(f"Health Score: {health_data['health_score']}%")
        print(f"Status: {health_data['status']}")
        
        # Check recommendations
        for rec in health_data['recommendations']:
            print(f"Recommendation: {rec}")

# Run health check
asyncio.run(check_vector_health())
```

### Specific Vector Type Monitoring

```python
async def monitor_vector_type(vector_type: str):
    async with aiohttp.ClientSession() as session:
        async with session.get(f"http://localhost:8000/health/vectors/{vector_type}") as response:
            vector_health = await response.json()
            
        print(f"{vector_type} vector status: {vector_health['status']}")
        print(f"Points count: {vector_health['details']['checks']['points_count']}")

# Monitor specific vector types
for vector_type in ["stats", "context", "value", "combined"]:
    await monitor_vector_type(vector_type)
```

### Performance Monitoring

```python
async def check_performance():
    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/health/performance") as response:
            performance = await response.json()
            
        # Check memory usage
        memory = performance['memory_breakdown']
        print(f"Memory usage: {memory['percentage']}%")
        
        # Check performance trends
        trends = performance['performance_trends']
        print(f"Latency trend: {trends['latency_trend']}")
        
        # Check recommendations
        for rec in performance['optimization_recommendations']:
            print(f"Optimization: {rec}")

# Run performance check
asyncio.run(check_performance())
```

## Integration with Monitoring Systems

### Prometheus Metrics

The health endpoints can be integrated with Prometheus for metrics collection:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'dfs-prophet-health'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/health/vectors'
    scrape_interval: 30s
```

### Grafana Dashboards

Create dashboards to visualize health metrics:

- **Health Score Trends**: Track health score over time
- **Memory Usage**: Monitor memory breakdown by vector type
- **Performance Metrics**: Track latency and search performance
- **Collection Status**: Monitor collection health percentages

### Alerting Rules

Set up alerting based on health thresholds:

```yaml
# alertmanager.yml
groups:
  - name: dfs-prophet-alerts
    rules:
      - alert: VectorHealthCritical
        expr: health_score < 50
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Vector system health is critical"
          
      - alert: VectorPerformanceDegraded
        expr: average_latency > 200
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Vector search performance is degraded"
```

## Best Practices

### 1. Regular Monitoring

- Set up automated health checks every 5-15 minutes
- Monitor health scores and trends over time
- Set up alerts for critical thresholds

### 2. Performance Optimization

- Use health check recommendations for system optimization
- Monitor memory usage and optimize vector storage
- Track latency trends and optimize search algorithms

### 3. Data Quality Management

- Regularly check data quality metrics
- Address consistency issues between vector types
- Monitor collection health and fix empty collections

### 4. Capacity Planning

- Monitor memory usage trends
- Plan for vector collection growth
- Optimize storage based on usage patterns

## Troubleshooting

### Common Issues

1. **Empty Collections**
   - Check if data loading processes are working
   - Verify collection initialization
   - Check for data pipeline errors

2. **High Latency**
   - Consider using quantized collections
   - Optimize search algorithms
   - Check system resources

3. **Memory Issues**
   - Optimize vector storage
   - Consider data archiving
   - Monitor memory trends

4. **Consistency Issues**
   - Verify data synchronization between collections
   - Check for data corruption
   - Validate vector dimensions

### Debug Mode

Enable debug logging for detailed health check information:

```python
import logging
logging.getLogger('dfs_prophet.api.routes.health').setLevel(logging.DEBUG)
```

## Future Enhancements

- **Real-time Health Streaming**: WebSocket-based real-time health updates
- **Predictive Health Analytics**: ML-based health prediction
- **Automated Remediation**: Self-healing capabilities for common issues
- **Health History**: Long-term health trend analysis
- **Custom Health Checks**: User-defined health check rules
