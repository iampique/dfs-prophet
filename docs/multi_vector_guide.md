# Multi-Vector DFS Prophet Guide

## Overview

DFS Prophet's multi-vector system provides sophisticated player analysis through four specialized vector types, each capturing different aspects of player performance and value. This guide explains how to effectively use these vectors for optimal DFS decision-making.

## Vector Types Overview

### 1. Statistical Vectors (`stats`)
**Purpose**: Capture player performance patterns and statistical trends
- **Best for**: Performance-based searches, historical analysis, consistency evaluation
- **Dimensions**: 768
- **Use cases**: Finding similar statistical performers, identifying trends, performance prediction

### 2. Contextual Vectors (`context`)
**Purpose**: Encode game situation and matchup factors
- **Best for**: Matchup analysis, situational plays, weather/opponent considerations
- **Dimensions**: 768
- **Use cases**: Favorable matchup identification, weather impact analysis, situational value plays

### 3. Value Vectors (`value`)
**Purpose**: Capture DFS market dynamics and value patterns
- **Best for**: Salary efficiency, ownership leverage, ROI optimization
- **Dimensions**: 768
- **Use cases**: Finding undervalued players, ownership arbitrage, salary efficiency analysis

### 4. Combined Vectors (`combined`)
**Purpose**: Fusion of all vector types for comprehensive analysis
- **Best for**: Holistic player evaluation, multi-factor analysis, comprehensive searches
- **Dimensions**: 768
- **Use cases**: Complete player profiling, multi-dimensional similarity, strategic planning

## Performance Trade-offs

### Search Speed vs. Accuracy

| Vector Type | Speed | Accuracy | Use Case |
|-------------|-------|----------|----------|
| `stats` | Fast | High | Performance-focused searches |
| `context` | Fast | Medium | Situational analysis |
| `value` | Fast | Medium | Market-based decisions |
| `combined` | Slower | Highest | Comprehensive analysis |

### Memory Usage

- **Individual vectors**: ~3KB per player per vector type
- **Combined vectors**: ~12KB per player (all types)
- **Binary quantization**: 40x memory reduction with minimal accuracy loss

### Recommended Usage Patterns

```python
# Fast performance searches
results = await vector_engine.search_vectors(
    query, CollectionType.REGULAR, vector_types=["stats"]
)

# Comprehensive analysis
results = await vector_engine.search_vectors(
    query, CollectionType.REGULAR, vector_types=["stats", "context", "value"]
)

# Binary quantized for speed
results = await vector_engine.search_vectors(
    query, CollectionType.BINARY_QUANTIZED, vector_types=["combined"]
)
```

## Implementation Patterns

### 1. Player Data Ingestion

```python
from dfs_prophet.data.models.player import Player, PlayerBase, PlayerStats, PlayerDFS
from dfs_prophet.core.embedding_generator import EmbeddingGenerator, EmbeddingStrategy

async def ingest_player_data(player_data: Dict[str, Any]) -> None:
    """Ingest player data with multi-vector embeddings."""
    
    # Create player object
    player = Player(
        base=PlayerBase(
            player_id=player_data["id"],
            name=player_data["name"],
            position=player_data["position"],
            team=player_data["team"],
            season=player_data["season"],
            week=player_data["week"]
        ),
        stats=PlayerStats(**player_data["stats"]),
        dfs=PlayerDFS(**player_data["dfs"])
    )
    
    # Generate embeddings for each vector type
    embedding_generator = EmbeddingGenerator()
    
    # Statistical embedding
    stats_embedding = await embedding_generator.generate_player_embedding(
        player, strategy=EmbeddingStrategy.STATISTICAL
    )
    
    # Contextual embedding
    context_embedding = await embedding_generator.generate_player_embedding(
        player, strategy=EmbeddingStrategy.CONTEXTUAL
    )
    
    # Value embedding
    value_embedding = await embedding_generator.generate_player_embedding(
        player, strategy=EmbeddingStrategy.VALUE
    )
    
    # Store in vector engine
    vector_engine = VectorEngine()
    await vector_engine.upsert_multi_vector_player(
        player, {
            "stats": stats_embedding,
            "context": context_embedding,
            "value": value_embedding
        }
    )
```

### 2. Multi-Vector Search

```python
async def search_similar_players(
    query: str,
    vector_types: List[str] = ["stats", "context", "value"],
    limit: int = 10
) -> List[SearchResult]:
    """Search for similar players using multiple vector types."""
    
    vector_engine = VectorEngine()
    
    # Generate query embedding
    embedding_generator = EmbeddingGenerator()
    query_embedding = await embedding_generator.generate_query_embedding(query)
    
    # Search with multiple vector types
    results = await vector_engine.search_vectors(
        query_embedding,
        CollectionType.REGULAR,
        vector_types=vector_types,
        limit=limit
    )
    
    return results
```

### 3. Weighted Multi-Vector Search

```python
async def weighted_search(
    query: str,
    weights: Dict[str, float] = {"stats": 0.4, "context": 0.3, "value": 0.3}
) -> List[SearchResult]:
    """Search with custom vector weights."""
    
    vector_engine = VectorEngine()
    
    # Use weighted fusion search
    results = await vector_engine.search_vectors(
        query,
        CollectionType.REGULAR,
        vector_types=list(weights.keys()),
        weights=weights,
        limit=10
    )
    
    return results
```

## Common Use Cases

### 1. Performance-Based Player Discovery

```python
# Find players with similar statistical profiles
similar_qbs = await search_similar_players(
    "high passing yards quarterback",
    vector_types=["stats"],
    limit=5
)
```

### 2. Matchup Analysis

```python
# Find players in favorable matchups
favorable_matchups = await search_similar_players(
    "weak opponent defense",
    vector_types=["context"],
    limit=10
)
```

### 3. Value Play Identification

```python
# Find undervalued players
value_plays = await search_similar_players(
    "low ownership high upside",
    vector_types=["value"],
    limit=8
)
```

### 4. Comprehensive Player Analysis

```python
# Full player profile analysis
comprehensive_results = await search_similar_players(
    "elite quarterback with favorable matchup",
    vector_types=["stats", "context", "value"],
    limit=5
)
```

## Best Practices

### 1. Vector Type Selection

- **Performance focus**: Use `stats` vector
- **Situational analysis**: Use `context` vector
- **Market analysis**: Use `value` vector
- **Comprehensive analysis**: Use all vectors or `combined`

### 2. Performance Optimization

- **Fast searches**: Use individual vectors or binary quantization
- **High accuracy**: Use combined vectors with regular quantization
- **Memory efficiency**: Use binary quantization for large datasets

### 3. Search Strategy

- **Specific queries**: Use targeted vector types
- **Broad queries**: Use combined vectors
- **Weighted searches**: Customize importance of different factors

### 4. Data Quality

- **Regular updates**: Keep embeddings current with latest data
- **Validation**: Verify embedding quality before deployment
- **Monitoring**: Track search performance and accuracy

## Troubleshooting

### Common Issues

1. **Empty search results**
   - Check if vectors exist for the specified types
   - Verify query embedding generation
   - Ensure collection is populated

2. **Poor search accuracy**
   - Validate embedding quality
   - Check vector type selection
   - Consider using combined vectors

3. **Slow search performance**
   - Use binary quantization
   - Limit vector types
   - Optimize query complexity

4. **Memory issues**
   - Enable binary quantization
   - Use individual vectors instead of combined
   - Implement vector cleanup

### Debugging Tips

```python
# Check collection status
stats = await vector_engine.get_collection_stats(CollectionType.REGULAR)
print(f"Points in collection: {stats['points_count']}")

# Verify vector existence
player_vectors = await vector_engine.get_player_vectors(player_id)
print(f"Available vectors: {player_vectors.keys()}")

# Test individual vector searches
for vector_type in ["stats", "context", "value"]:
    results = await vector_engine.search_vectors(
        query, CollectionType.REGULAR, vector_types=[vector_type]
    )
    print(f"{vector_type}: {len(results)} results")
```

## Advanced Features

### 1. Vector Fusion Strategies

- **Weighted Average**: Custom weights for each vector type
- **Max Score**: Use highest similarity score across vectors
- **Ensemble**: Combine multiple search strategies

### 2. Conditional Logic

- **Performance thresholds**: Filter by fantasy points
- **Value criteria**: Filter by salary efficiency
- **Matchup conditions**: Filter by opponent strength

### 3. Temporal Analysis

- **Recent performance**: Weight recent games more heavily
- **Trend analysis**: Identify improving/declining players
- **Seasonal patterns**: Account for seasonal variations

## Conclusion

The multi-vector system provides powerful tools for DFS analysis. Choose the right vector types for your specific use case, optimize for performance when needed, and leverage the combined power of all vectors for comprehensive analysis.

For more detailed information, see:
- [API Examples](api_examples.md)
- [Vector Types](vector_types.md)
- [Working Example](examples/multi_vector_demo.py)
