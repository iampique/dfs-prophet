# DFS Prophet API Examples

## Overview

This document provides comprehensive examples of how to use the DFS Prophet API for multi-vector player analysis. Each example includes complete code snippets and explanations.

## API Endpoints

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication
Currently, the API doesn't require authentication for local development.

## Core Search Endpoints

### 1. Basic Player Search

**Endpoint**: `GET /players/search`

**Description**: Search for players using natural language queries with configurable vector types.

```python
import requests
import json

def basic_player_search():
    """Basic player search with natural language query."""
    
    url = "http://localhost:8000/api/v1/players/search"
    
    params = {
        "query": "high passing yards quarterback",
        "limit": 5,
        "min_similarity": 0.7,
        "vector_types": ["stats", "context", "value"]
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        results = response.json()
        print(f"Found {len(results['results'])} players:")
        
        for result in results['results']:
            player = result['player']
            print(f"- {player['name']} ({player['position']}) - Score: {result['similarity_score']:.3f}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Usage
basic_player_search()
```

**Response Example**:
```json
{
  "results": [
    {
      "player": {
        "player_id": "12345",
        "name": "Patrick Mahomes",
        "position": "QB",
        "team": "KC",
        "season": 2024,
        "week": 1
      },
      "similarity_score": 0.892,
      "vector_contributions": {
        "stats": 0.45,
        "context": 0.28,
        "value": 0.27
      }
    }
  ],
  "total_results": 1,
  "query_time_ms": 45.2
}
```

### 2. Statistical Vector Search

**Endpoint**: `GET /players/search/stats`

**Description**: Search specifically using statistical performance vectors.

```python
def stats_vector_search():
    """Search using statistical performance vectors."""
    
    url = "http://localhost:8000/api/v1/players/search/stats"
    
    params = {
        "query": "rushing touchdowns running back",
        "limit": 10,
        "min_similarity": 0.6
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        results = response.json()
        print("Top rushing TD performers:")
        
        for result in results['results']:
            player = result['player']
            stats = player.get('stats', {})
            print(f"- {player['name']}: {stats.get('rushing_touchdowns', 0)} TDs")
    else:
        print(f"Error: {response.status_code}")

# Usage
stats_vector_search()
```

### 3. Contextual Vector Search

**Endpoint**: `GET /players/search/context`

**Description**: Search using game context and matchup vectors.

```python
def context_vector_search():
    """Search using contextual and matchup vectors."""
    
    url = "http://localhost:8000/api/v1/players/search/context"
    
    params = {
        "query": "favorable matchup home game",
        "limit": 8,
        "min_similarity": 0.65
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        results = response.json()
        print("Players with favorable matchups:")
        
        for result in results['results']:
            player = result['player']
            print(f"- {player['name']} vs {player.get('opponent_team', 'Unknown')}")
    else:
        print(f"Error: {response.status_code}")

# Usage
context_vector_search()
```

### 4. Value Vector Search

**Endpoint**: `GET /players/search/value`

**Description**: Search using DFS value and market vectors.

```python
def value_vector_search():
    """Search using DFS value and market vectors."""
    
    url = "http://localhost:8000/api/v1/players/search/value"
    
    params = {
        "query": "low ownership high value",
        "limit": 6,
        "min_similarity": 0.7
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        results = response.json()
        print("Value plays with low ownership:")
        
        for result in results['results']:
            player = result['player']
            dfs = player.get('dfs', {})
            print(f"- {player['name']}: ${dfs.get('salary', 0)} ({dfs.get('ownership_percentage', 0)}% owned)")
    else:
        print(f"Error: {response.status_code}")

# Usage
value_vector_search()
```

### 5. Fusion Search

**Endpoint**: `GET /players/search/fusion`

**Description**: Search using weighted combination of all vector types.

```python
def fusion_search():
    """Search using weighted fusion of all vector types."""
    
    url = "http://localhost:8000/api/v1/players/search/fusion"
    
    params = {
        "query": "elite quarterback with favorable matchup and good value",
        "limit": 5,
        "weights": {
            "stats": 0.4,
            "context": 0.3,
            "value": 0.3
        }
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        results = response.json()
        print("Elite QBs with favorable matchups and value:")
        
        for result in results['results']:
            player = result['player']
            print(f"- {player['name']}: Score {result['similarity_score']:.3f}")
    else:
        print(f"Error: {response.status_code}")

# Usage
fusion_search()
```

## Advanced Search Endpoints

### 6. Player Analysis

**Endpoint**: `GET /players/analyze/{player_id}`

**Description**: Get comprehensive analysis of a specific player.

```python
def analyze_player(player_id: str):
    """Get comprehensive analysis of a specific player."""
    
    url = f"http://localhost:8000/api/v1/players/analyze/{player_id}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        analysis = response.json()
        
        print(f"Analysis for {analysis['name']}:")
        print(f"Position: {analysis['position']}")
        print(f"Team: {analysis['team']}")
        
        # Vector strengths
        print("\nVector Strengths:")
        for vector_type, strength in analysis['vector_strengths'].items():
            print(f"- {vector_type}: {strength:.3f}")
        
        # Recommendations
        print("\nRecommendations:")
        for rec in analysis['recommendations']:
            print(f"- {rec}")
            
    else:
        print(f"Error: {response.status_code}")

# Usage
analyze_player("12345")
```

### 7. Vector Comparison

**Endpoint**: `POST /players/compare/vectors`

**Description**: Compare multiple players across different vector types.

```python
def compare_players(player_ids: list):
    """Compare multiple players across vector types."""
    
    url = "http://localhost:8000/api/v1/players/compare/vectors"
    
    data = {
        "player_ids": player_ids,
        "vector_types": ["stats", "context", "value"],
        "comparison_metrics": ["similarity", "performance", "value"]
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        comparison = response.json()
        
        print("Player Comparison Results:")
        for result in comparison['results']:
            player = result['player']
            print(f"\n{player['name']}:")
            
            for metric, score in result['metrics'].items():
                print(f"- {metric}: {score:.3f}")
    else:
        print(f"Error: {response.status_code}")

# Usage
compare_players(["12345", "67890", "11111"])
```

## Health and Monitoring Endpoints

### 8. System Health

**Endpoint**: `GET /health`

**Description**: Check overall system health.

```python
def check_system_health():
    """Check overall system health."""
    
    url = "http://localhost:8000/api/v1/health"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        health = response.json()
        
        print(f"System Status: {health['status']}")
        print(f"Qdrant Status: {health['qdrant']['status']}")
        print(f"Embedding Model: {health['embedding_model']['status']}")
        print(f"API Version: {health['version']}")
    else:
        print(f"Error: {response.status_code}")

# Usage
check_system_health()
```

### 9. Vector System Health

**Endpoint**: `GET /health/vectors`

**Description**: Check multi-vector system health.

```python
def check_vector_health():
    """Check multi-vector system health."""
    
    url = "http://localhost:8000/api/v1/health/vectors"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        health = response.json()
        
        print("Vector System Health:")
        print(f"Overall Status: {health['status']}")
        
        for vector_type, status in health['vector_collections'].items():
            print(f"- {vector_type}: {status['status']} ({status['points_count']} points)")
        
        print(f"Cross-vector Consistency: {health['cross_vector_consistency']['status']}")
    else:
        print(f"Error: {response.status_code}")

# Usage
check_vector_health()
```

### 10. Performance Metrics

**Endpoint**: `GET /health/performance`

**Description**: Get performance metrics for the multi-vector system.

```python
def get_performance_metrics():
    """Get performance metrics for the multi-vector system."""
    
    url = "http://localhost:8000/api/v1/health/performance"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        metrics = response.json()
        
        print("Performance Metrics:")
        print(f"Total Searches: {metrics['total_searches']}")
        print(f"Average Latency: {metrics['average_latency']:.2f}ms")
        
        print("\nStrategy Performance:")
        for strategy, perf in metrics['strategy_performance'].items():
            print(f"- {strategy}: {perf['latency']:.2f}ms, {perf['accuracy']:.3f}")
    else:
        print(f"Error: {response.status_code}")

# Usage
get_performance_metrics()
```

## Complete Example: Multi-Vector Analysis Workflow

```python
import asyncio
import aiohttp
import json
from typing import List, Dict, Any

class DFSProphetClient:
    """Client for DFS Prophet API with multi-vector capabilities."""
    
    def __init__(self, base_url: str = "http://localhost:8000/api/v1"):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def search_players(
        self,
        query: str,
        vector_types: List[str] = ["stats", "context", "value"],
        limit: int = 10,
        min_similarity: float = 0.6
    ) -> Dict[str, Any]:
        """Search for players using multiple vector types."""
        
        params = {
            "query": query,
            "vector_types": ",".join(vector_types),
            "limit": limit,
            "min_similarity": min_similarity
        }
        
        async with self.session.get(f"{self.base_url}/players/search", params=params) as response:
            return await response.json()
    
    async def analyze_player(self, player_id: str) -> Dict[str, Any]:
        """Get comprehensive analysis of a player."""
        
        async with self.session.get(f"{self.base_url}/players/analyze/{player_id}") as response:
            return await response.json()
    
    async def compare_players(self, player_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple players."""
        
        data = {
            "player_ids": player_ids,
            "vector_types": ["stats", "context", "value"]
        }
        
        async with self.session.post(f"{self.base_url}/players/compare/vectors", json=data) as response:
            return await response.json()

async def complete_analysis_workflow():
    """Complete multi-vector analysis workflow."""
    
    async with DFSProphetClient() as client:
        print("🔍 DFS Prophet Multi-Vector Analysis")
        print("=" * 50)
        
        # 1. Search for elite QBs
        print("\n1. Searching for elite quarterbacks...")
        qb_results = await client.search_players(
            "elite quarterback high fantasy points",
            vector_types=["stats"],
            limit=5
        )
        
        qb_ids = [result['player']['player_id'] for result in qb_results['results']]
        print(f"Found {len(qb_ids)} elite QBs")
        
        # 2. Analyze each QB
        print("\n2. Analyzing each quarterback...")
        for result in qb_results['results']:
            player = result['player']
            analysis = await client.analyze_player(player['player_id'])
            
            print(f"\n{player['name']} Analysis:")
            print(f"- Similarity Score: {result['similarity_score']:.3f}")
            print(f"- Vector Strengths: {analysis['vector_strengths']}")
            print(f"- Top Recommendation: {analysis['recommendations'][0] if analysis['recommendations'] else 'None'}")
        
        # 3. Compare QBs
        print("\n3. Comparing quarterbacks...")
        comparison = await client.compare_players(qb_ids[:3])
        
        print("QB Comparison Results:")
        for result in comparison['results']:
            player = result['player']
            print(f"\n{player['name']}:")
            for metric, score in result['metrics'].items():
                print(f"- {metric}: {score:.3f}")

# Run the complete workflow
if __name__ == "__main__":
    asyncio.run(complete_analysis_workflow())
```

## Error Handling

```python
def handle_api_errors(response):
    """Handle common API errors."""
    
    if response.status_code == 404:
        print("Error: Endpoint not found. Check the API URL.")
    elif response.status_code == 422:
        print("Error: Invalid request parameters.")
        errors = response.json()
        for error in errors.get('detail', []):
            print(f"- {error['msg']}")
    elif response.status_code == 500:
        print("Error: Internal server error. Check server logs.")
    elif response.status_code == 503:
        print("Error: Service unavailable. Check if Qdrant is running.")
    else:
        print(f"Error: {response.status_code} - {response.text}")

# Usage in API calls
response = requests.get(url, params=params)
if response.status_code != 200:
    handle_api_errors(response)
```

## Best Practices

1. **Use appropriate vector types** for your specific use case
2. **Set reasonable similarity thresholds** (0.6-0.8 for most searches)
3. **Limit results** to avoid overwhelming responses
4. **Handle errors gracefully** with proper error handling
5. **Use async clients** for better performance in production
6. **Monitor performance** using the health endpoints
7. **Cache results** when appropriate to reduce API calls

## Next Steps

- Explore the [Multi-Vector Guide](multi_vector_guide.md) for detailed usage patterns
- Check [Vector Types](vector_types.md) for technical details
- Run the [Working Example](examples/multi_vector_demo.py) for hands-on experience
