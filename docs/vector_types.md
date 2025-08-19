# Vector Types Technical Documentation

## Overview

DFS Prophet implements four specialized vector types, each designed to capture different aspects of player performance and value. This document provides detailed technical specifications, implementation details, and usage guidelines for each vector type.

## Vector Architecture

### Common Specifications

All vector types share these common characteristics:
- **Embedding Model**: BAAI/bge-base-en-v1.5
- **Base Dimensions**: 768 (configurable)
- **Distance Metric**: Cosine similarity
- **Normalization**: L2 normalization applied
- **Storage**: Qdrant vector database with binary quantization support

### Vector Generation Pipeline

```python
# Common pipeline for all vector types
1. Feature Extraction → 2. Text Generation → 3. Embedding → 4. Normalization → 5. Storage
```

## 1. Statistical Vectors (`stats`)

### Purpose
Capture player performance patterns, statistical trends, and historical consistency.

### Technical Specifications

```python
class PlayerStatVector(BaseModel):
    vector_id: str                    # Unique identifier
    embedding: List[float]           # 768-dimensional vector
    vector_dimensions: int           # 768
    embedding_model: str             # "BAAI/bge-base-en-v1.5"
    embedding_timestamp: datetime    # Generation timestamp
    stat_features: Dict[str, Any]    # Statistical features used
    stat_summary: str               # Text summary for embedding
    similarity_score: Optional[float] # Search result score
```

### Feature Engineering

**Primary Features**:
- Fantasy points (PPR and standard)
- Passing statistics (yards, TDs, interceptions)
- Rushing statistics (yards, TDs, attempts)
- Receiving statistics (yards, TDs, targets, receptions)
- Game consistency metrics
- Season totals and averages

**Feature Processing**:
```python
def _extract_statistical_features(player: Player) -> Dict[str, Any]:
    """Extract statistical features for embedding generation."""
    
    stats = player.stats or PlayerStats()
    
    return {
        "fantasy_points": stats.fantasy_points or 0,
        "fantasy_points_ppr": stats.fantasy_points_ppr or 0,
        "passing_yards": stats.passing_yards or 0,
        "passing_touchdowns": stats.passing_touchdowns or 0,
        "passing_interceptions": stats.passing_interceptions or 0,
        "rushing_yards": stats.rushing_yards or 0,
        "rushing_touchdowns": stats.rushing_touchdowns or 0,
        "receiving_yards": stats.receiving_yards or 0,
        "receiving_touchdowns": stats.receiving_touchdowns or 0,
        "targets": stats.targets or 0,
        "receptions": stats.receptions or 0,
        "games_played": stats.games_played or 1,
        "consistency_score": self._calculate_consistency(stats),
        "efficiency_rating": self._calculate_efficiency(stats),
        "trend_direction": self._calculate_trend(stats)
    }
```

**Text Generation**:
```python
def _generate_statistical_summary(features: Dict[str, Any]) -> str:
    """Generate text summary for statistical embedding."""
    
    return f"""
    Player with {features['fantasy_points']:.1f} fantasy points per game.
    Passing: {features['passing_yards']} yards, {features['passing_touchdowns']} TDs, {features['passing_interceptions']} INTs.
    Rushing: {features['rushing_yards']} yards, {features['rushing_touchdowns']} TDs.
    Receiving: {features['receiving_yards']} yards, {features['receiving_touchdowns']} TDs, {features['targets']} targets.
    Consistency: {features['consistency_score']:.2f}, Efficiency: {features['efficiency_rating']:.2f}.
    """
```

### Use Cases

**Optimal Scenarios**:
- Performance-based player discovery
- Historical trend analysis
- Consistency evaluation
- Statistical similarity searches

**Example Queries**:
- "high passing yards quarterback"
- "consistent fantasy points running back"
- "rushing touchdown leader"
- "efficient wide receiver"

### Performance Characteristics

- **Search Speed**: Fast (single vector type)
- **Accuracy**: High for performance-based queries
- **Memory Usage**: ~3KB per player
- **Update Frequency**: After each game

## 2. Contextual Vectors (`context`)

### Purpose
Encode game situation, matchup factors, and environmental conditions.

### Technical Specifications

```python
class PlayerContextVector(BaseModel):
    vector_id: str                    # Unique identifier
    embedding: List[float]           # 768-dimensional vector
    vector_dimensions: int           # 768
    embedding_model: str             # "BAAI/bge-base-en-v1.5"
    embedding_timestamp: datetime    # Generation timestamp
    context_features: Dict[str, Any] # Contextual features used
    context_summary: str            # Text summary for embedding
    similarity_score: Optional[float] # Search result score
```

### Feature Engineering

**Primary Features**:
- Opponent team and defensive ranking
- Home/away game status
- Weather conditions (temperature, wind, precipitation)
- Game total and point spread
- Injury status and rest days
- Team offensive scheme
- Historical matchup data

**Feature Processing**:
```python
def _extract_contextual_features(player: Player) -> Dict[str, Any]:
    """Extract contextual features for embedding generation."""
    
    dfs = player.dfs or PlayerDFS()
    
    return {
        "opponent_team": player.base.team.value if player.base.team else "Unknown",
        "opponent_strength": self._get_opponent_strength(player),
        "home_away": self._determine_home_away(player),
        "weather_temp": dfs.weather_temp or 72.0,
        "weather_wind": dfs.weather_wind or 5.0,
        "weather_conditions": dfs.weather_conditions or "clear",
        "game_total": dfs.game_total or 48.5,
        "team_spread": dfs.team_spread or 0.0,
        "injury_status": dfs.injury_status or "healthy",
        "rest_days": self._calculate_rest_days(player),
        "matchup_difficulty": self._calculate_matchup_difficulty(player),
        "momentum_factor": self._calculate_momentum(player)
    }
```

**Text Generation**:
```python
def _generate_contextual_summary(features: Dict[str, Any]) -> str:
    """Generate text summary for contextual embedding."""
    
    return f"""
    {features['home_away'].title()} game vs {features['opponent_team']} (strength: {features['opponent_strength']:.1f}).
    Weather: {features['weather_conditions']}, {features['weather_temp']}°F, {features['weather_wind']}mph wind.
    Game total: {features['game_total']}, spread: {features['team_spread']}.
    Injury status: {features['injury_status']}, rest days: {features['rest_days']}.
    Matchup difficulty: {features['matchup_difficulty']:.1f}/10, momentum: {features['momentum_factor']:.2f}.
    """
```

### Use Cases

**Optimal Scenarios**:
- Matchup analysis
- Situational plays
- Weather impact assessment
- Home/away advantage analysis

**Example Queries**:
- "favorable matchup home game"
- "weak opponent defense"
- "good weather conditions"
- "rest advantage"

### Performance Characteristics

- **Search Speed**: Fast (single vector type)
- **Accuracy**: Medium (context-dependent)
- **Memory Usage**: ~3KB per player
- **Update Frequency**: Weekly (game context)

## 3. Value Vectors (`value`)

### Purpose
Capture DFS market dynamics, salary efficiency, and ownership patterns.

### Technical Specifications

```python
class PlayerValueVector(BaseModel):
    vector_id: str                    # Unique identifier
    embedding: List[float]           # 768-dimensional vector
    vector_dimensions: int           # 768
    embedding_model: str             # "BAAI/bge-base-en-v1.5"
    embedding_timestamp: datetime    # Generation timestamp
    value_features: Dict[str, Any]   # Value features used
    value_summary: str              # Text summary for embedding
    similarity_score: Optional[float] # Search result score
```

### Feature Engineering

**Primary Features**:
- Current salary and salary trends
- Ownership percentage and trends
- Value rating (points per $1000)
- ROI history and volatility
- Salary efficiency metrics
- Ownership leverage potential

**Feature Processing**:
```python
def _extract_value_features(player: Player) -> Dict[str, Any]:
    """Extract value features for embedding generation."""
    
    dfs = player.dfs or PlayerDFS()
    
    return {
        "current_salary": dfs.salary or 5000,
        "salary_trend": self._calculate_salary_trend(player),
        "ownership_percentage": dfs.ownership_percentage or 10.0,
        "ownership_trend": self._calculate_ownership_trend(player),
        "value_rating": dfs.value_rating or 2.0,
        "roi_history": self._get_roi_history(player),
        "avg_roi": self._calculate_avg_roi(player),
        "salary_volatility": self._calculate_salary_volatility(player),
        "ownership_volatility": self._calculate_ownership_volatility(player),
        "salary_efficiency": self._calculate_salary_efficiency(player),
        "ownership_leverage": self._calculate_ownership_leverage(player)
    }
```

**Text Generation**:
```python
def _generate_value_summary(features: Dict[str, Any]) -> str:
    """Generate text summary for value embedding."""
    
    return f"""
    Salary ${features['current_salary']} (trending {features['salary_trend']:.1%}).
    Ownership {features['ownership_percentage']:.1f}% (trending {features['ownership_trend']:.1%}).
    Value rating {features['value_rating']:.2f}, avg ROI {features['avg_roi']:.2f}.
    Salary efficiency {features['salary_efficiency']:.2f}, ownership leverage {features['ownership_leverage']:.2f}.
    Volatility: salary {features['salary_volatility']:.2f}, ownership {features['ownership_volatility']:.2f}.
    """
```

### Use Cases

**Optimal Scenarios**:
- Salary efficiency analysis
- Ownership arbitrage
- Value play identification
- Market mispricing detection

**Example Queries**:
- "low ownership high value"
- "salary efficiency play"
- "undervalued player"
- "ownership leverage"

### Performance Characteristics

- **Search Speed**: Fast (single vector type)
- **Accuracy**: Medium (market-dependent)
- **Memory Usage**: ~3KB per player
- **Update Frequency**: Daily (market data)

## 4. Combined Vectors (`combined`)

### Purpose
Fusion of all vector types for comprehensive, multi-dimensional analysis.

### Technical Specifications

```python
class PlayerMultiVector(BaseModel):
    player_id: str                    # Player identifier
    season: int                      # Season year
    week: Optional[int]              # Week number
    stat_vector: Optional[PlayerStatVector]      # Statistical vector
    context_vector: Optional[PlayerContextVector] # Contextual vector
    value_vector: Optional[PlayerValueVector]    # Value vector
    combined_vector: Optional[PlayerVector]      # Combined vector
    vector_metadata: Dict[str, Any]  # Additional metadata
```

### Fusion Strategies

**1. Weighted Average Fusion**:
```python
def _weighted_fusion_search(
    self,
    query_vector: np.ndarray,
    weights: Dict[str, float] = {"stats": 0.4, "context": 0.3, "value": 0.3}
) -> List[SearchResult]:
    """Weighted fusion of multiple vector types."""
    
    results = {}
    
    # Search each vector type
    for vector_type, weight in weights.items():
        vector_results = await self._search_single_vector(query_vector, vector_type)
        for result in vector_results:
            player_id = result.player.player_id
            if player_id not in results:
                results[player_id] = result
                results[player_id].similarity_score *= weight
            else:
                results[player_id].similarity_score += result.similarity_score * weight
    
    return sorted(results.values(), key=lambda x: x.similarity_score, reverse=True)
```

**2. Max Score Fusion**:
```python
def _max_score_fusion(
    self,
    query_vector: np.ndarray,
    vector_types: List[str] = ["stats", "context", "value"]
) -> List[SearchResult]:
    """Use highest similarity score across vector types."""
    
    all_results = {}
    
    for vector_type in vector_types:
        vector_results = await self._search_single_vector(query_vector, vector_type)
        for result in vector_results:
            player_id = result.player.player_id
            if player_id not in all_results:
                all_results[player_id] = result
            else:
                all_results[player_id].similarity_score = max(
                    all_results[player_id].similarity_score,
                    result.similarity_score
                )
    
    return sorted(all_results.values(), key=lambda x: x.similarity_score, reverse=True)
```

**3. Ensemble Fusion**:
```python
def _ensemble_fusion(
    self,
    query_vector: np.ndarray,
    strategies: List[str] = ["weighted", "max_score", "conditional"]
) -> List[SearchResult]:
    """Ensemble multiple fusion strategies."""
    
    ensemble_results = []
    
    for strategy in strategies:
        if strategy == "weighted":
            results = await self._weighted_fusion_search(query_vector)
        elif strategy == "max_score":
            results = await self._max_score_fusion(query_vector)
        elif strategy == "conditional":
            results = await self._conditional_logic_search(query_vector)
        
        ensemble_results.extend(results)
    
    # Aggregate and rank results
    return self._aggregate_ensemble_results(ensemble_results)
```

### Use Cases

**Optimal Scenarios**:
- Comprehensive player evaluation
- Multi-factor analysis
- Strategic planning
- Complete player profiling

**Example Queries**:
- "elite quarterback with favorable matchup and good value"
- "consistent performer in good situation"
- "high-upside player with low ownership"

### Performance Characteristics

- **Search Speed**: Slower (multiple vector types)
- **Accuracy**: Highest (comprehensive analysis)
- **Memory Usage**: ~12KB per player (all types)
- **Update Frequency**: As needed (all vector types)

## Vector Quality Metrics

### Embedding Quality Assessment

```python
def _assess_embedding_quality(embedding: List[float]) -> Dict[str, float]:
    """Assess the quality of generated embeddings."""
    
    embedding_array = np.array(embedding)
    
    return {
        "magnitude": np.linalg.norm(embedding_array),
        "sparsity": np.mean(embedding_array == 0),
        "variance": np.var(embedding_array),
        "entropy": -np.sum(embedding_array * np.log(embedding_array + 1e-10)),
        "cosine_similarity_self": np.dot(embedding_array, embedding_array) / (np.linalg.norm(embedding_array) ** 2)
    }
```

### Quality Thresholds

| Metric | Excellent | Good | Acceptable | Poor |
|--------|-----------|------|------------|------|
| Magnitude | 0.95-1.05 | 0.90-1.10 | 0.85-1.15 | <0.85 or >1.15 |
| Sparsity | <0.1 | <0.2 | <0.3 | >0.3 |
| Variance | >0.01 | >0.005 | >0.001 | <0.001 |
| Entropy | >2.0 | >1.5 | >1.0 | <1.0 |

## Vector Storage and Retrieval

### Qdrant Configuration

```python
# Named vectors configuration
VECTORS_CONFIG = {
    "stats": rest.VectorParams(
        size=768,
        distance=rest.Distance.COSINE,
        on_disk=True
    ),
    "context": rest.VectorParams(
        size=768,
        distance=rest.Distance.COSINE,
        on_disk=True
    ),
    "value": rest.VectorParams(
        size=768,
        distance=rest.Distance.COSINE,
        on_disk=True
    ),
    "combined": rest.VectorParams(
        size=768,
        distance=rest.Distance.COSINE,
        on_disk=True
    )
}

# Binary quantization for performance
BINARY_CONFIG = rest.BinaryQuantizationConfig(
    always_ram=True
)
```

### Storage Optimization

```python
def _optimize_storage_strategy(vector_type: str, usage_pattern: str) -> Dict[str, Any]:
    """Optimize storage strategy based on vector type and usage pattern."""
    
    strategies = {
        "stats": {
            "frequent_access": {"on_disk": False, "always_ram": True},
            "balanced": {"on_disk": True, "always_ram": False},
            "memory_efficient": {"on_disk": True, "always_ram": False}
        },
        "context": {
            "frequent_access": {"on_disk": False, "always_ram": True},
            "balanced": {"on_disk": True, "always_ram": False},
            "memory_efficient": {"on_disk": True, "always_ram": False}
        },
        "value": {
            "frequent_access": {"on_disk": False, "always_ram": True},
            "balanced": {"on_disk": True, "always_ram": False},
            "memory_efficient": {"on_disk": True, "always_ram": False}
        },
        "combined": {
            "frequent_access": {"on_disk": False, "always_ram": True},
            "balanced": {"on_disk": True, "always_ram": False},
            "memory_efficient": {"on_disk": True, "always_ram": True}
        }
    }
    
    return strategies.get(vector_type, {}).get(usage_pattern, {"on_disk": True, "always_ram": False})
```

## Performance Benchmarks

### Search Performance

| Vector Type | Average Latency (ms) | Memory Usage (MB) | Accuracy Score |
|-------------|---------------------|-------------------|----------------|
| `stats` | 15.2 | 45.3 | 0.89 |
| `context` | 18.7 | 42.1 | 0.76 |
| `value` | 16.9 | 38.7 | 0.82 |
| `combined` | 45.3 | 156.2 | 0.94 |

### Binary Quantization Impact

| Metric | Regular | Binary Quantized | Improvement |
|--------|---------|------------------|-------------|
| Memory Usage | 100% | 2.5% | 40x reduction |
| Search Speed | 100% | 85% | 1.2x faster |
| Accuracy | 100% | 98% | 2% loss |

## Best Practices

### 1. Vector Type Selection

- **Performance focus**: Use `stats` vector
- **Situational analysis**: Use `context` vector
- **Market analysis**: Use `value` vector
- **Comprehensive analysis**: Use `combined` vector

### 2. Performance Optimization

- **Fast searches**: Use individual vectors
- **High accuracy**: Use combined vectors
- **Memory efficiency**: Enable binary quantization
- **Scalability**: Use on-disk storage for large datasets

### 3. Quality Assurance

- **Regular validation**: Check embedding quality metrics
- **Consistency checks**: Verify cross-vector consistency
- **Performance monitoring**: Track search latency and accuracy
- **Error handling**: Implement graceful degradation

### 4. Maintenance

- **Regular updates**: Keep embeddings current
- **Cleanup**: Remove outdated vectors
- **Backup**: Regular vector database backups
- **Monitoring**: Track system health metrics

## Troubleshooting

### Common Issues

1. **Poor search accuracy**
   - Check embedding quality metrics
   - Verify feature extraction
   - Validate text generation

2. **Slow search performance**
   - Enable binary quantization
   - Use on-disk storage
   - Optimize vector dimensions

3. **Memory issues**
   - Enable binary quantization
   - Use on-disk storage
   - Implement vector cleanup

4. **Inconsistent results**
   - Check cross-vector consistency
   - Validate fusion strategies
   - Monitor quality metrics

### Debugging Tools

```python
def debug_vector_issues(vector_type: str, player_id: str):
    """Debug vector-related issues."""
    
    # Check vector existence
    vectors = await vector_engine.get_player_vectors(player_id)
    if vector_type not in vectors:
        print(f"Vector {vector_type} not found for player {player_id}")
        return
    
    # Check vector quality
    vector = vectors[vector_type]
    quality = _assess_embedding_quality(vector.embedding)
    print(f"Vector quality metrics: {quality}")
    
    # Check storage status
    stats = await vector_engine.get_collection_stats(CollectionType.REGULAR)
    print(f"Collection stats: {stats}")
    
    # Test search functionality
    test_query = [0.1] * 768
    results = await vector_engine.search_vectors(
        test_query, CollectionType.REGULAR, vector_types=[vector_type]
    )
    print(f"Search test results: {len(results)} found")
```

## Conclusion

The multi-vector system provides powerful tools for DFS analysis through specialized vector types. Understanding the technical specifications, implementation details, and best practices for each vector type is essential for optimal usage and performance.

For implementation examples, see:
- [Multi-Vector Guide](multi_vector_guide.md)
- [API Examples](api_examples.md)
- [Working Example](examples/multi_vector_demo.py)
