# Multi-Vector API Endpoints

This document describes the comprehensive multi-vector search and analysis API endpoints for DFS Prophet.

## Overview

The multi-vector API provides advanced player search capabilities using different vector types:
- **Statistical Vectors**: Performance metrics, fantasy points, efficiency
- **Contextual Vectors**: Game context, weather, opponent strength, venue
- **Value Vectors**: DFS value, salary efficiency, ownership trends, ROI

## Base URL

```
http://localhost:8000/players
```

## Endpoints

### 1. Statistical Vector Search

**GET** `/players/search/stats`

Search for players using statistical vector similarity only.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Search query for statistical similarity |
| `limit` | integer | ❌ | 10 | Maximum number of results (1-100) |
| `score_threshold` | float | ❌ | 0.5 | Minimum similarity score (0.0-1.0) |
| `position` | string | ❌ | - | Filter by player position |
| `team` | string | ❌ | - | Filter by team abbreviation |
| `season` | integer | ❌ | - | Filter by season year (2020-2024) |

#### Example Request

```bash
curl "http://localhost:8000/players/search/stats?query=elite%20quarterback%20with%20high%20fantasy%20points&limit=5"
```

#### Example Response

```json
{
  "query": "elite quarterback with high fantasy points",
  "vector_types": ["stats"],
  "fusion_strategy": "weighted_average",
  "weights": {"stats": 1.0},
  "results": [
    {
      "player_id": "mahomes_patrick",
      "name": "Patrick Mahomes",
      "position": "QB",
      "team": "KC",
      "season": 2024,
      "week": 1,
      "final_score": 0.313,
      "vector_contributions": [
        {
          "vector_type": "stats",
          "weight": 1.0,
          "score": 0.313,
          "contribution": 0.313,
          "explanation": "Statistical similarity: 26.8 fantasy points"
        }
      ],
      "primary_vector": "stats",
      "match_explanation": "Matched primarily on stats vector: Statistical similarity: 26.8 fantasy points",
      "fantasy_points": 26.8,
      "salary": 9500,
      "projected_points": 25.2,
      "stats": {
        "passing_yards": 320,
        "rushing_yards": 15,
        "receiving_yards": 0,
        "total_touchdowns": 3,
        "total_yards": 335
      }
    }
  ],
  "total_results": 5,
  "search_time_ms": 71.48,
  "embedding_time_ms": 2000.55,
  "fusion_time_ms": 0.01,
  "total_time_ms": 2072.11,
  "timestamp": "2024-01-15T10:30:00"
}
```

### 2. Contextual Vector Search

**GET** `/players/search/context`

Search for players using contextual vector similarity only.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Search query for contextual similarity |
| `limit` | integer | ❌ | 10 | Maximum number of results (1-100) |
| `score_threshold` | float | ❌ | 0.5 | Minimum similarity score (0.0-1.0) |
| `position` | string | ❌ | - | Filter by player position |
| `team` | string | ❌ | - | Filter by team abbreviation |
| `season` | integer | ❌ | - | Filter by season year (2020-2024) |

#### Example Request

```bash
curl "http://localhost:8000/players/search/context?query=players%20who%20perform%20well%20in%20cold%20weather&limit=5"
```

### 3. Value Vector Search

**GET** `/players/search/value`

Search for players using value vector similarity only.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Search query for DFS value patterns |
| `limit` | integer | ❌ | 10 | Maximum number of results (1-100) |
| `score_threshold` | float | ❌ | 0.5 | Minimum similarity score (0.0-1.0) |
| `position` | string | ❌ | - | Filter by player position |
| `team` | string | ❌ | - | Filter by team abbreviation |
| `season` | integer | ❌ | - | Filter by season year (2020-2024) |

#### Example Request

```bash
curl "http://localhost:8000/players/search/value?query=undervalued%20players%20with%20high%20salary%20efficiency&limit=5"
```

### 4. Multi-Vector Fusion Search

**GET** `/players/search/fusion`

Search for players using multi-vector fusion with weighted combination.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Search query for multi-vector analysis |
| `vector_types` | string | ❌ | "stats,context,value" | Comma-separated vector types |
| `weights` | string | ❌ | - | Comma-separated weights (e.g., "0.4,0.3,0.3") |
| `fusion_strategy` | string | ❌ | "weighted_average" | Fusion strategy |
| `limit` | integer | ❌ | 10 | Maximum number of results (1-100) |
| `score_threshold` | float | ❌ | 0.5 | Minimum similarity score (0.0-1.0) |
| `position` | string | ❌ | - | Filter by player position |
| `team` | string | ❌ | - | Filter by team abbreviation |
| `season` | integer | ❌ | - | Filter by season year (2020-2024) |

#### Fusion Strategies

- `weighted_average`: Weighted combination of vector scores
- `max_score`: Take the maximum score across all vectors
- `min_score`: Take the minimum score across all vectors
- `product`: Multiply scores from all vectors

#### Example Request

```bash
curl "http://localhost:8000/players/search/fusion?query=comprehensive%20player%20search&vector_types=stats,context,value&weights=0.4,0.3,0.3&fusion_strategy=weighted_average&limit=5"
```

### 5. Complete Player Analysis

**GET** `/players/analyze/{player_id}`

Perform complete multi-vector analysis of a specific player.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `player_id` | string | ✅ | - | Player ID to analyze |
| `vector_types` | string | ❌ | "stats,context,value" | Comma-separated vector types to analyze |

#### Example Request

```bash
curl "http://localhost:8000/players/analyze/mahomes_patrick?vector_types=stats,context,value"
```

#### Example Response

```json
{
  "player_id": "mahomes_patrick",
  "name": "Patrick Mahomes",
  "position": "QB",
  "team": "KC",
  "season": 2024,
  "week": 1,
  "stats_analysis": {
    "fantasy_points": 25.5,
    "efficiency_metrics": {
      "yards_per_attempt": 8.2,
      "touchdown_rate": 0.15
    },
    "consistency_score": 0.75,
    "trend_analysis": "Improving over last 3 weeks"
  },
  "context_analysis": {
    "weather_impact": 0.05,
    "opponent_strength": "Above average",
    "venue_factors": "Home game advantage",
    "time_analysis": "Primetime performance boost"
  },
  "value_analysis": {
    "salary_efficiency": 0.85,
    "ownership_trends": "Increasing",
    "roi_potential": "High",
    "market_position": "Undervalued"
  },
  "combined_analysis": {
    "overall_score": 0.78,
    "risk_reward_ratio": 1.2,
    "recommendation": "Strong play",
    "confidence": 0.85
  },
  "similar_players": {
    "stats": ["allen_josh", "burrow_joe"],
    "context": ["rodgers_aaron", "brady_tom"],
    "value": ["herbert_justin", "prescott_dak"]
  },
  "vector_strengths": {
    "stats": 0.85,
    "context": 0.72,
    "value": 0.68
  },
  "vector_weaknesses": {
    "stats": ["Inconsistent fantasy points"],
    "context": ["Poor weather performance"],
    "value": ["High ownership risk"]
  },
  "recommendations": [
    "Strong statistical profile suggests high floor",
    "Contextual factors favor performance",
    "Value metrics indicate good ROI potential"
  ],
  "risk_factors": [
    "Recent injury concerns",
    "Tough defensive matchup",
    "High ownership could limit upside"
  ],
  "timestamp": "2024-01-15T10:30:00"
}
```

### 6. Multi-Vector Player Comparison

**POST** `/players/compare/vectors`

Compare multiple players across different vector types.

#### Request Body

```json
{
  "player_ids": ["mahomes_patrick", "allen_josh", "burrow_joe"],
  "vector_types": ["stats", "context", "value"],
  "comparison_metric": "similarity"
}
```

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `player_ids` | array | ✅ | - | Player IDs to compare (2-5 players) |
| `vector_types` | array | ❌ | ["stats", "context", "value"] | Vector types to include |
| `comparison_metric` | string | ❌ | "similarity" | Metric to use for comparison |

#### Comparison Metrics

- `similarity`: Vector similarity scores
- `performance`: Performance-based comparison
- `value`: Value-based comparison
- `consistency`: Consistency-based comparison

#### Example Request

```bash
curl -X POST "http://localhost:8000/players/compare/vectors" \
  -H "Content-Type: application/json" \
  -d '{
    "player_ids": ["mahomes_patrick", "allen_josh", "burrow_joe"],
    "vector_types": ["stats", "context", "value"],
    "comparison_metric": "similarity"
  }'
```

#### Example Response

```json
{
  "comparison_metric": "similarity",
  "vector_types": ["stats", "context", "value"],
  "player_comparisons": {
    "stats": {
      "mahomes_patrick": 0.85,
      "allen_josh": 0.78,
      "burrow_joe": 0.72
    },
    "context": {
      "mahomes_patrick": 0.72,
      "allen_josh": 0.68,
      "burrow_joe": 0.75
    },
    "value": {
      "mahomes_patrick": 0.68,
      "allen_josh": 0.72,
      "burrow_joe": 0.65
    }
  },
  "vector_contributions": {
    "mahomes_patrick": {
      "stats": 0.85,
      "context": 0.72,
      "value": 0.68
    },
    "allen_josh": {
      "stats": 0.78,
      "context": 0.68,
      "value": 0.72
    },
    "burrow_joe": {
      "stats": 0.72,
      "context": 0.75,
      "value": 0.65
    }
  },
  "rankings": {
    "stats": ["mahomes_patrick", "allen_josh", "burrow_joe"],
    "context": ["burrow_joe", "mahomes_patrick", "allen_josh"],
    "value": ["allen_josh", "mahomes_patrick", "burrow_joe"]
  },
  "insights": [
    "Top stats performer: mahomes_patrick (0.850)",
    "Top context performer: burrow_joe (0.750)",
    "Top value performer: allen_josh (0.720)"
  ],
  "timestamp": "2024-01-15T10:30:00"
}
```

### 7. Enhanced Configurable Search

**GET** `/players/search`

Enhanced configurable player search with multi-vector support.

#### Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `query` | string | ✅ | - | Search query |
| `limit` | integer | ❌ | 10 | Maximum number of results (1-100) |
| `score_threshold` | float | ❌ | 0.5 | Minimum similarity score (0.0-1.0) |
| `collection_type` | string | ❌ | "binary" | Collection type (regular/binary) |
| `strategy` | string | ❌ | "TEXT_ONLY" | Embedding strategy |
| `vector_types` | string | ❌ | - | Comma-separated vector types for multi-vector search |
| `position` | string | ❌ | - | Filter by player position |
| `team` | string | ❌ | - | Filter by team abbreviation |
| `season` | integer | ❌ | - | Filter by season year (2020-2024) |

#### Example Request (Single Vector)

```bash
curl "http://localhost:8000/players/search?query=elite%20quarterback&collection_type=binary&limit=5"
```

#### Example Request (Multi Vector)

```bash
curl "http://localhost:8000/players/search?query=elite%20quarterback&vector_types=stats,context,value&limit=5"
```

## Response Models

### MultiVectorSearchResult

```json
{
  "player_id": "string",
  "name": "string",
  "position": "string",
  "team": "string",
  "season": "integer",
  "week": "integer",
  "final_score": "float",
  "vector_contributions": [
    {
      "vector_type": "string",
      "weight": "float",
      "score": "float",
      "contribution": "float",
      "explanation": "string"
    }
  ],
  "primary_vector": "string",
  "match_explanation": "string",
  "fantasy_points": "float",
  "salary": "integer",
  "projected_points": "float",
  "stats": {
    "passing_yards": "integer",
    "rushing_yards": "integer",
    "receiving_yards": "integer",
    "total_touchdowns": "integer",
    "total_yards": "integer"
  }
}
```

### VectorContribution

```json
{
  "vector_type": "string",
  "weight": "float",
  "score": "float",
  "contribution": "float",
  "explanation": "string"
}
```

## Usage Examples

### Finding Elite Quarterbacks

```bash
# Statistical search for elite QBs
curl "http://localhost:8000/players/search/stats?query=elite%20quarterback%20high%20fantasy%20points&limit=10"

# Contextual search for primetime performers
curl "http://localhost:8000/players/search/context?query=primetime%20performance%20cold%20weather&limit=10"

# Value search for undervalued players
curl "http://localhost:8000/players/search/value?query=undervalued%20salary%20efficiency&limit=10"

# Comprehensive fusion search
curl "http://localhost:8000/players/search/fusion?query=elite%20quarterback&vector_types=stats,context,value&weights=0.5,0.3,0.2&limit=10"
```

### Player Analysis

```bash
# Complete analysis of a specific player
curl "http://localhost:8000/players/analyze/mahomes_patrick"

# Compare multiple players
curl -X POST "http://localhost:8000/players/compare/vectors" \
  -H "Content-Type: application/json" \
  -d '{
    "player_ids": ["mahomes_patrick", "allen_josh"],
    "vector_types": ["stats", "context", "value"],
    "comparison_metric": "similarity"
  }'
```

## Performance Considerations

1. **Embedding Generation**: Most time is spent generating embeddings (1-2 seconds)
2. **Vector Search**: Very fast (milliseconds) once embeddings are generated
3. **Fusion**: Minimal overhead for combining multiple vectors
4. **Caching**: Embeddings are cached for repeated queries

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200 OK`: Successful request
- `400 Bad Request`: Invalid parameters
- `404 Not Found`: Player not found
- `500 Internal Server Error`: Server error

Error responses include detailed error messages:

```json
{
  "detail": "Multi-vector search failed: Invalid vector type 'invalid_type'"
}
```

## Rate Limiting

- Default: 100 requests per minute per IP
- Burst: 10 requests per second
- Custom limits can be configured

## Authentication

Currently, no authentication is required. For production use, consider implementing:

- API key authentication
- JWT tokens
- OAuth 2.0
