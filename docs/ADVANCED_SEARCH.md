# Advanced Search Features

DFS Prophet's advanced search engine provides sophisticated multi-vector search capabilities with dynamic weight adjustment, contextual search, archetype filtering, and A/B testing framework.

## Overview

The advanced search engine extends the basic vector search functionality with intelligent algorithms that adapt to different query types, game situations, and user preferences. It provides explanations for search results and tracks performance metrics for continuous optimization.

## Key Features

### 1. Dynamic Weight Adjustment

The search engine automatically adjusts vector weights based on query type and context:

```python
from dfs_prophet.search.advanced_search import AdvancedSearchEngine, QueryType, SearchContext

search_engine = AdvancedSearchEngine()

# Different query types get different weight adjustments
results = await search_engine.search(
    query="elite quarterbacks",
    query_type=QueryType.PERFORMANCE_SEARCH  # Boosts stats vector
)

# Context affects weights
context = SearchContext(
    weather_conditions="rainy and windy",
    injury_report={"RB": ["McCaffrey"]}
)
results = await search_engine.search(
    query="players for this situation",
    context=context  # Boosts context vector
)
```

**Weight Adjustments by Query Type:**
- `PERFORMANCE_SEARCH`: Boosts stats (1.5x), reduces context (0.7x)
- `VALUE_SEARCH`: Boosts value (1.4x), reduces stats (0.8x)
- `CONTEXT_SEARCH`: Boosts context (1.5x), reduces stats (0.6x)
- `MATCHUP_SEARCH`: Boosts context (1.3x), balanced stats/value
- `SLEEPER_PICK`: Boosts context (1.2x) and value (1.2x)
- `STACK_SEARCH`: Boosts context (1.4x), reduces value (0.6x)

### 2. Search Strategies

#### Weighted Fusion Search
Combines multiple vector types with configurable weights:

```python
from dfs_prophet.search.advanced_search import SearchStrategy

results = await search_engine.search(
    query="similar players",
    strategy=SearchStrategy.WEIGHTED_FUSION,
    limit=10
)
```

#### Conditional Logic Search
Multi-stage search with filtering and refinement:

```python
results = await search_engine.search(
    query="high-performing value players",
    strategy=SearchStrategy.CONDITIONAL_LOGIC,
    query_type=QueryType.VALUE_SEARCH
)
```

**Stages:**
1. **Initial Search**: Broad search across all vectors
2. **Conditional Filtering**: Apply filters based on query type
3. **Archetype Refinement**: Classify and enhance results

#### Temporal Search
Considers recent vs historical performance patterns:

```python
context = SearchContext(game_week=5)
results = await search_engine.search(
    query="players trending up",
    strategy=SearchStrategy.TEMPORAL_SEARCH,
    context=context
)
```

**Weighting:**
- Recent performance (last 3 weeks): 70% weight
- Historical performance: 30% weight

#### Ensemble Search
Combines multiple strategies for robust results:

```python
results = await search_engine.search(
    query="comprehensive player search",
    strategy=SearchStrategy.ENSEMBLE_SEARCH
)
```

**Combined Strategies:**
- Weighted Fusion
- Conditional Logic
- Temporal Search

#### Archetype Filtered Search
Filters results by player archetype:

```python
from dfs_prophet.analytics.profile_analyzer import ArchetypeType

results = await search_engine.search(
    query="volume rushers",
    strategy=SearchStrategy.ARCHETYPE_FILTERED
)
```

### 3. Contextual Search

The search engine adapts to game situations and external factors:

```python
context = SearchContext(
    game_week=8,
    opponent_team="Bills",
    home_away="away",
    weather_conditions="rainy and windy",
    injury_report={"RB": ["McCaffrey", "Taylor"]},
    salary_cap=50000.0,
    contest_type="tournament",
    user_preferences={"risk_tolerance": "high"}
)

results = await search_engine.search(
    query="players for this game",
    context=context
)
```

**Context Adjustments:**
- **Weather**: Rain/wind boosts context importance (1.3x)
- **Injuries**: Injury reports boost context (1.2x) and value (1.1x)
- **Home/Away**: Away games boost context (1.1x)
- **Contest Type**: Affects risk tolerance and strategy

### 4. A/B Testing Framework

Test different search strategies in production:

```python
from dfs_prophet.search.advanced_search import ABTestConfig

# Create A/B test
config = ABTestConfig(
    test_id="strategy_comparison_001",
    strategy_a=SearchStrategy.WEIGHTED_FUSION,
    strategy_b=SearchStrategy.ENSEMBLE_SEARCH,
    traffic_split=0.5,  # 50% traffic to strategy B
    duration_days=7,
    min_sample_size=100
)

test_id = await search_engine.create_ab_test(config)

# Use A/B test in searches
results = await search_engine.search(
    query="test query",
    ab_test_id=test_id  # Strategy chosen automatically
)

# Get results
ab_results = await search_engine.get_ab_test_results(test_id)
if ab_results:
    print(f"Winner: {ab_results.winner}")
    print(f"Confidence: {ab_results.confidence_level}")
```

### 5. Search Explanations

Every search result includes detailed explanations:

```python
results = await search_engine.search("elite quarterbacks")

for result in results:
    print(f"Player: {result.player.name}")
    print(f"Score: {result.similarity_score:.3f}")
    print(f"Strategy: {result.explanation.strategy_used}")
    print(f"Reasoning: {result.explanation.reasoning}")
    print(f"Vector Contributions: {result.explanation.vector_contributions}")
    print(f"Weight Adjustments: {result.explanation.weight_adjustments}")
    print(f"Confidence: {result.confidence_level}")
```

**Explanation Components:**
- **Strategy Used**: Which search algorithm was employed
- **Vector Contributions**: How each vector type contributed to the score
- **Weight Adjustments**: Dynamic weight modifications applied
- **Filters Applied**: Any conditional filters that were used
- **Reasoning**: Human-readable explanation of the result
- **Confidence Score**: Reliability of the match

## API Integration

### Advanced Search Endpoints

```python
# Basic advanced search
POST /players/search/advanced
{
    "query": "elite quarterbacks",
    "strategy": "weighted_fusion",
    "query_type": "performance_search",
    "context": {
        "game_week": 5,
        "weather_conditions": "rainy",
        "salary_cap": 50000.0
    },
    "limit": 10,
    "ab_test_id": "optional_test_id"
}

# A/B test management
POST /search/ab-tests
{
    "test_id": "strategy_test_001",
    "strategy_a": "weighted_fusion",
    "strategy_b": "ensemble_search",
    "traffic_split": 0.5,
    "duration_days": 7
}

GET /search/ab-tests/{test_id}/results

# Search analytics
GET /search/analytics
```

## Performance Monitoring

The search engine tracks comprehensive performance metrics:

```python
analytics = await search_engine.get_search_analytics()

print(f"Total searches: {analytics['total_searches']}")
print(f"Average latency: {analytics['average_latency']:.1f}ms")
print(f"Active A/B tests: {analytics['active_ab_tests']}")

# Strategy performance breakdown
for strategy, perf in analytics['strategy_performance'].items():
    print(f"{strategy}: {perf['latency']:.1f}ms avg, {perf['count']} searches")
```

## Configuration

### Environment Variables

```bash
# Vector weights (defaults)
VECTOR_WEIGHTS_STATS=0.4
VECTOR_WEIGHTS_CONTEXT=0.3
VECTOR_WEIGHTS_VALUE=0.3

# Search settings
ENABLE_VECTOR_FUSION=true
MAX_VECTORS_PER_SEARCH=3

# A/B testing
AB_TEST_TRAFFIC_SPLIT=0.5
AB_TEST_MIN_SAMPLE_SIZE=100
```

### Performance Thresholds

```python
# Configure performance thresholds
search_engine.thresholds = {
    "max_latency_ms": 200,
    "min_accuracy": 0.7,
    "max_memory_mb": 1000
}
```

## Best Practices

### 1. Query Type Selection

Choose the appropriate query type for your use case:

- **Performance Search**: For finding high-scoring players
- **Value Search**: For finding underpriced players
- **Context Search**: For situation-specific recommendations
- **Matchup Search**: For favorable defensive matchups
- **Sleeper Pick**: For low-ownership high-upside players

### 2. Context Utilization

Provide rich context for better results:

```python
context = SearchContext(
    game_week=current_week,
    weather_conditions=weather_data,
    injury_report=injury_data,
    salary_cap=contest_salary,
    contest_type=contest_type,
    user_preferences=user_prefs
)
```

### 3. A/B Testing Strategy

- Start with small traffic splits (10-20%)
- Use clear success metrics (accuracy, latency, user satisfaction)
- Run tests for sufficient duration (7-14 days minimum)
- Monitor for statistical significance

### 4. Performance Optimization

- Cache frequently used embeddings
- Use appropriate collection types (regular vs quantized)
- Monitor search latency and adjust thresholds
- Implement result caching for repeated queries

## Examples

### Example 1: Tournament Lineup Search

```python
context = SearchContext(
    game_week=8,
    contest_type="tournament",
    salary_cap=50000.0,
    user_preferences={"risk_tolerance": "high", "stack_preference": True}
)

results = await search_engine.search(
    query="tournament lineup players",
    strategy=SearchStrategy.ENSEMBLE_SEARCH,
    query_type=QueryType.SLEEPER_PICK,
    context=context,
    limit=20
)
```

### Example 2: Cash Game Value Search

```python
context = SearchContext(
    game_week=8,
    contest_type="cash",
    salary_cap=45000.0,
    user_preferences={"risk_tolerance": "low"}
)

results = await search_engine.search(
    query="safe value plays",
    strategy=SearchStrategy.CONDITIONAL_LOGIC,
    query_type=QueryType.VALUE_SEARCH,
    context=context,
    limit=10
)
```

### Example 3: Weather-Affected Search

```python
context = SearchContext(
    weather_conditions="rainy and windy",
    opponent_team="Bills",
    home_away="away"
)

results = await search_engine.search(
    query="weather-appropriate players",
    strategy=SearchStrategy.CONTEXTUAL_SEARCH,
    context=context,
    limit=15
)
```

## Troubleshooting

### Common Issues

1. **Empty Results**: Check if vector collections are populated
2. **High Latency**: Consider using quantized collections or caching
3. **Low Accuracy**: Verify embedding quality and weight adjustments
4. **A/B Test Not Ready**: Ensure minimum sample size is reached

### Debug Mode

Enable debug logging for detailed search information:

```python
import logging
logging.getLogger('dfs_prophet.search').setLevel(logging.DEBUG)
```

## Future Enhancements

- **Machine Learning Integration**: Learn optimal weights from user feedback
- **Real-time Context Updates**: Dynamic context adjustment during games
- **Advanced Filtering**: More sophisticated conditional logic
- **Multi-language Support**: Support for different languages and regions
- **Federated Search**: Search across multiple data sources
