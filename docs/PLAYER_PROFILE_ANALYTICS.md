# Player Profile Analytics

## Overview

The Player Profile Analytics module provides comprehensive multi-vector analysis of DFS players, enabling deep insights into player characteristics, performance patterns, and strategic opportunities.

## Core Features

### 🏷️ Player Archetype Classification
Automatically classifies players into strategic archetypes based on their multi-vector profiles:

- **Volume Rusher**: High-volume running backs with consistent carries
- **Red Zone Target**: Players frequently targeted in red zone situations
- **Deep Threat**: Wide receivers specializing in deep passes
- **Elite QB**: Elite quarterbacks with high fantasy production
- **Value Play**: Undervalued players with high potential
- **Anomaly**: Players with unusual vector combinations

### 📊 Multi-Dimensional Similarity Scoring
Identifies similar players across multiple vector dimensions:

- **Statistical Similarity**: Based on fantasy points, yards, touchdowns
- **Contextual Similarity**: Based on game context, matchups, schemes
- **Value Similarity**: Based on salary efficiency, ownership, upside
- **Weighted Fusion**: Combines all vectors with configurable weights

### 🔍 Vector Contribution Analysis
Analyzes how each vector type contributes to a player's overall profile:

- **Stats Vector**: Statistical performance contribution
- **Context Vector**: Game context and situational factors
- **Value Vector**: DFS value and market factors
- **Contribution Percentages**: Relative importance of each vector

### 🎯 Player Cluster Identification
Groups players into clusters based on multi-vector similarity:

- **K-means Clustering**: Adaptive clustering based on player count
- **PCA Dimensionality Reduction**: Handles high-dimensional vector data
- **Cluster Characteristics**: Average stats, salary, consistency per cluster
- **Cluster Archetypes**: Dominant archetype identification per cluster

### 🚨 Anomaly Detection
Identifies players with unusual vector patterns:

- **Z-score Analysis**: Statistical outlier detection
- **Vector Magnitude Analysis**: Unusual vector strength patterns
- **Cross-vector Discordance**: Inconsistent patterns across vectors
- **Recommendations**: Risk mitigation strategies for anomalies

### 🔮 Performance Prediction
Predicts future performance using multi-vector data:

- **Base Prediction**: Current fantasy points as baseline
- **Vector Adjustments**: Multi-vector factor analysis
- **Confidence Intervals**: Uncertainty quantification
- **Risk Factors**: Identification of performance risks
- **Upside/Downside**: Potential performance ranges

### ⚔️ Matchup Analysis
Analyzes player advantages against different opponents:

- **Opponent-specific Analysis**: Team-by-team matchup evaluation
- **Vector Advantage Scoring**: Which vectors favor the player
- **Matchup Strategy**: Recommended approach for each matchup
- **Confidence Scoring**: Reliability of matchup predictions

### 💰 Value Opportunity Identification
Identifies undervalued players and market opportunities:

- **Salary Efficiency**: Points per dollar analysis
- **Ownership Projection**: Low-ownership high-upside plays
- **Market Mispricing**: Discrepancies between projection and value
- **Expected Return**: Quantified opportunity potential

## Usage Examples

### Basic Player Analysis

```python
from dfs_prophet.analytics import PlayerProfileAnalyzer

# Initialize analyzer
analyzer = PlayerProfileAnalyzer()

# Analyze a player's complete profile
profile = await analyzer.analyze_player_profile(
    player_id="player_123",
    include_similarities=True,
    include_clusters=True,
    include_predictions=True
)

# Access analysis results
print(f"Archetype: {profile.archetype.archetype.value}")
print(f"Confidence: {profile.archetype.confidence_score:.1%}")
print(f"Predicted Points: {profile.predictions.predicted_fantasy_points:.1f}")
```

### Individual Analysis Components

```python
# Generate player embeddings
embeddings = await analyzer._generate_player_embeddings(player)

# Classify archetype
archetype = await analyzer._classify_archetype(player, embeddings)

# Analyze vector contributions
contributions = await analyzer._analyze_vector_contributions(player, embeddings)

# Find similar players
similarities = await analyzer._find_similar_players(player, embeddings)

# Detect anomalies
anomaly = await analyzer._detect_anomalies(player, embeddings)

# Predict performance
prediction = await analyzer._predict_performance(player, embeddings)

# Analyze matchups
matchups = await analyzer._analyze_matchups(player, embeddings)

# Identify value opportunities
opportunities = await analyzer._identify_value_opportunities(player, embeddings)
```

## Data Models

### PlayerProfile
Complete player analysis result containing all analysis components:

```python
@dataclass
class PlayerProfile:
    player_id: str
    player_name: str
    position: Position
    team: Team
    archetype: ArchetypeResult
    vector_contributions: List[VectorContribution]
    top_similarities: List[SimilarityResult]
    cluster_info: Optional[ClusterResult]
    anomaly_info: Optional[AnomalyResult]
    predictions: PredictionResult
    matchups: List[MatchupAnalysis]
    value_opportunities: List[ValueOpportunity]
    visualization_data: Dict[str, Any]
    analysis_timestamp: datetime
```

### ArchetypeResult
Player archetype classification result:

```python
@dataclass
class ArchetypeResult:
    archetype: ArchetypeType
    confidence_score: float
    primary_vectors: List[str]
    secondary_vectors: List[str]
    archetype_features: Dict[str, float]
    explanation: str
```

### VectorContribution
Analysis of vector contribution to player profile:

```python
@dataclass
class VectorContribution:
    vector_type: str
    contribution_score: float
    contribution_percentage: float
    explanation: str
    key_features: List[str]
```

### SimilarityResult
Multi-dimensional similarity analysis result:

```python
@dataclass
class SimilarityResult:
    player_id: str
    player_name: str
    overall_similarity: float
    vector_similarities: Dict[str, float]
    weighted_similarity: float
    rank: int
    explanation: str
```

### PredictionResult
Performance prediction with confidence intervals:

```python
@dataclass
class PredictionResult:
    predicted_fantasy_points: float
    confidence_interval: Tuple[float, float]
    prediction_factors: Dict[str, float]
    risk_factors: List[str]
    upside_potential: float
    downside_risk: float
```

## Configuration

### Archetype Definitions
Archetypes are defined with vector patterns and key features:

```python
archetype_definitions = {
    ArchetypeType.VOLUME_RUSHER: {
        "description": "High-volume running back with consistent carries",
        "primary_vectors": ["stats", "value"],
        "key_features": ["rushing_attempts", "rushing_yards", "touchdowns"],
        "vector_patterns": {
            "stats": {"rushing_attempts": 0.8, "rushing_yards": 0.7},
            "value": {"salary_efficiency": 0.6, "consistency": 0.7}
        }
    }
}
```

### Analysis Parameters
Configurable parameters for analysis:

```python
# Analysis parameters
max_similar_players = 10
cluster_min_size = 3
anomaly_threshold = 2.0  # Z-score threshold
prediction_horizon = 3  # weeks
```

## Visualization Data

The module prepares data for frontend visualizations:

### Radar Chart
Multi-dimensional player profile visualization:

```python
visualization_data = {
    "radar_chart": {
        "labels": ["Stats", "Context", "Value", "Consistency", "Upside"],
        "data": [0.8, 0.6, 0.7, 0.9, 0.5]
    }
}
```

### Vector Contributions
Pie chart showing vector importance:

```python
visualization_data = {
    "vector_contributions": {
        "labels": ["stats", "context", "value"],
        "data": [0.4, 0.3, 0.3]
    }
}
```

### Performance Trends
Historical performance visualization:

```python
visualization_data = {
    "performance_trend": {
        "labels": ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"],
        "data": [25.5, 28.2, 22.1, 30.8, 26.4]
    }
}
```

## Performance Considerations

### Caching
The analyzer implements intelligent caching:

- **Archetype Cache**: Caches archetype classifications
- **Similarity Cache**: Caches similarity search results
- **Cluster Cache**: Caches clustering results

### Async Operations
All analysis methods are async for optimal performance:

- **Concurrent Embedding Generation**: Parallel vector generation
- **Batch Processing**: Efficient handling of multiple players
- **Non-blocking Operations**: Responsive UI during analysis

### Memory Management
Efficient memory usage for large datasets:

- **Lazy Loading**: Load data only when needed
- **Vector Normalization**: Consistent vector dimensions
- **Garbage Collection**: Automatic cleanup of temporary data

## Error Handling

### Graceful Degradation
The analyzer handles errors gracefully:

- **Missing Data**: Uses defaults for missing player attributes
- **Vector Generation Failures**: Falls back to zero vectors
- **Clustering Failures**: Returns None instead of crashing
- **Prediction Errors**: Uses conservative estimates

### Logging
Comprehensive logging for debugging:

- **Performance Timing**: Track analysis duration
- **Error Tracking**: Detailed error messages
- **Cache Statistics**: Monitor cache effectiveness
- **Vector Quality**: Assess embedding quality

## Integration Points

### Vector Engine Integration
Uses the vector engine for player data:

```python
# Get player data from vector engine
results = await self.vector_engine.search_vectors(
    query_vector=[0.0] * 768,
    collection_type=CollectionType.REGULAR,
    limit=1000
)
```

### Embedding Generator Integration
Uses the embedding generator for vector creation:

```python
# Generate multi-vector embeddings
embeddings = await self.embedding_generator.generate_player_embedding(
    player, strategy=EmbeddingStrategy.STATISTICAL
)
```

### Configuration Integration
Uses centralized configuration:

```python
# Get vector weights and thresholds
self.vector_weights = self.settings.get_vector_weights()
self.performance_thresholds = self.settings.get_performance_thresholds()
```

## Future Enhancements

### Advanced Analytics
Planned enhancements include:

- **Time Series Analysis**: Historical trend analysis
- **Monte Carlo Simulations**: Probabilistic performance modeling
- **Machine Learning Models**: Advanced prediction algorithms
- **Real-time Updates**: Live analysis during games

### Visualization Enhancements
Advanced visualization capabilities:

- **Interactive Charts**: Dynamic player comparisons
- **Network Graphs**: Player relationship visualization
- **Heat Maps**: Multi-dimensional pattern visualization
- **3D Scatter Plots**: Vector space visualization

### API Integration
Enhanced API capabilities:

- **Batch Analysis**: Analyze multiple players simultaneously
- **Streaming Results**: Real-time analysis updates
- **Webhook Notifications**: Alert on significant changes
- **Rate Limiting**: API usage management

## Conclusion

The Player Profile Analytics module provides a comprehensive foundation for advanced DFS player analysis. By combining multi-vector data with sophisticated analytics, it enables deep insights into player characteristics, performance patterns, and strategic opportunities.

The modular design allows for easy extension and customization, while the robust error handling and performance optimizations ensure reliable operation in production environments.
