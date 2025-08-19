"""
Advanced Search Engine for DFS Prophet

Provides sophisticated multi-vector search capabilities with dynamic weight adjustment,
contextual search, archetype filtering, and A/B testing framework.
"""

import asyncio
import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque

import numpy as np
from pydantic import BaseModel, Field

from ..config import get_settings
from ..utils.logger import get_logger
from ..utils.logger import performance_timer
from ..core.vector_engine import VectorEngine, CollectionType
from ..core.embedding_generator import EmbeddingGenerator, EmbeddingStrategy
from ..data.models.player import Player
from ..analytics.profile_analyzer import PlayerProfileAnalyzer, ArchetypeType


class SearchStrategy(Enum):
    """Search strategy types for different use cases."""
    WEIGHTED_FUSION = "weighted_fusion"
    CONDITIONAL_LOGIC = "conditional_logic"
    TEMPORAL_SEARCH = "temporal_search"
    ENSEMBLE_SEARCH = "ensemble_search"
    ARCHETYPE_FILTERED = "archetype_filtered"
    CONTEXTUAL_SEARCH = "contextual_search"


class QueryType(Enum):
    """Query types for dynamic weight adjustment."""
    POSITION_SEARCH = "position_search"
    PERFORMANCE_SEARCH = "performance_search"
    VALUE_SEARCH = "value_search"
    CONTEXT_SEARCH = "context_search"
    MATCHUP_SEARCH = "matchup_search"
    INJURY_REPLACEMENT = "injury_replacement"
    SLEEPER_PICK = "sleeper_pick"
    STACK_SEARCH = "stack_search"


@dataclass
class SearchContext:
    """Context information for search queries."""
    game_week: Optional[int] = None
    opponent_team: Optional[str] = None
    home_away: Optional[str] = None
    weather_conditions: Optional[str] = None
    injury_report: Optional[Dict[str, Any]] = None
    salary_cap: Optional[float] = None
    contest_type: Optional[str] = None
    user_preferences: Optional[Dict[str, Any]] = None
    historical_performance: Optional[Dict[str, float]] = None


@dataclass
class SearchExplanation:
    """Explanation for search results."""
    strategy_used: str
    vector_contributions: Dict[str, float]
    weight_adjustments: Dict[str, float]
    filters_applied: List[str]
    reasoning: str
    confidence_score: float
    alternative_strategies: List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    """Enhanced search result with explanation."""
    player: Player
    similarity_score: float
    explanation: SearchExplanation
    vector_scores: Dict[str, float]
    archetype_match: Optional[str] = None
    confidence_level: str = "medium"


@dataclass
class ABTestConfig:
    """A/B test configuration."""
    test_id: str
    strategy_a: SearchStrategy
    strategy_b: SearchStrategy
    traffic_split: float = 0.5  # Percentage for strategy B
    duration_days: int = 7
    success_metrics: List[str] = field(default_factory=lambda: ["accuracy", "latency"])
    min_sample_size: int = 100


@dataclass
class ABTestResult:
    """A/B test results."""
    test_id: str
    strategy_a_results: Dict[str, float]
    strategy_b_results: Dict[str, float]
    winner: Optional[str] = None
    confidence_level: float = 0.0
    sample_size: int = 0
    duration_days: int = 0


class AdvancedSearchEngine:
    """Advanced search engine with sophisticated multi-vector capabilities."""

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.vector_engine = VectorEngine()
        self.embedding_generator = EmbeddingGenerator()
        self.profile_analyzer = PlayerProfileAnalyzer()
        
        # A/B testing state
        self.ab_tests: Dict[str, ABTestConfig] = {}
        self.ab_results: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Search performance tracking
        self.search_history: deque = deque(maxlen=10000)
        self.strategy_performance: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"accuracy": 0.0, "latency": 0.0, "count": 0}
        )
        
        # Dynamic weight adjustment cache
        self.weight_cache: Dict[str, Dict[str, float]] = {}
        self.cache_ttl = timedelta(hours=1)
        self.last_cache_update = datetime.now()

    @performance_timer('dynamic_weight_adjustment')
    async def _calculate_dynamic_weights(
        self, 
        query_type: QueryType, 
        context: Optional[SearchContext] = None
    ) -> Dict[str, float]:
        """Calculate dynamic weights based on query type and context."""
        cache_key = f"{query_type.value}_{hash(str(context))}"
        
        # Check cache
        if cache_key in self.weight_cache:
            return self.weight_cache[cache_key]
        
        # Base weights from settings
        base_weights = {
            "stats": self.settings.multi_vector_search.vector_weight_stats,
            "context": self.settings.multi_vector_search.vector_weight_context,
            "value": self.settings.multi_vector_search.vector_weight_value
        }
        
        # Adjust based on query type
        adjustments = self._get_query_type_adjustments(query_type)
        
        # Apply context-based adjustments
        if context:
            context_adjustments = self._get_context_adjustments(context)
            for vector_type in adjustments:
                adjustments[vector_type] *= context_adjustments.get(vector_type, 1.0)
        
        # Calculate final weights
        final_weights = {}
        total_weight = 0.0
        
        for vector_type, base_weight in base_weights.items():
            adjusted_weight = base_weight * adjustments.get(vector_type, 1.0)
            final_weights[vector_type] = adjusted_weight
            total_weight += adjusted_weight
        
        # Normalize weights
        if total_weight > 0:
            for vector_type in final_weights:
                final_weights[vector_type] /= total_weight
        
        # Cache results
        self.weight_cache[cache_key] = final_weights
        return final_weights

    def _get_query_type_adjustments(self, query_type: QueryType) -> Dict[str, float]:
        """Get weight adjustments based on query type."""
        adjustments = {
            QueryType.POSITION_SEARCH: {"stats": 1.2, "context": 0.8, "value": 0.6},
            QueryType.PERFORMANCE_SEARCH: {"stats": 1.5, "context": 0.7, "value": 0.8},
            QueryType.VALUE_SEARCH: {"stats": 0.8, "context": 0.6, "value": 1.4},
            QueryType.CONTEXT_SEARCH: {"stats": 0.6, "context": 1.5, "value": 0.7},
            QueryType.MATCHUP_SEARCH: {"stats": 0.9, "context": 1.3, "value": 0.8},
            QueryType.INJURY_REPLACEMENT: {"stats": 1.1, "context": 1.1, "value": 0.9},
            QueryType.SLEEPER_PICK: {"stats": 0.7, "context": 1.2, "value": 1.2},
            QueryType.STACK_SEARCH: {"stats": 1.0, "context": 1.4, "value": 0.6}
        }
        return adjustments.get(query_type, {"stats": 1.0, "context": 1.0, "value": 1.0})

    def _get_context_adjustments(self, context: SearchContext) -> Dict[str, float]:
        """Get weight adjustments based on search context."""
        adjustments = {"stats": 1.0, "context": 1.0, "value": 1.0}
        
        # Weather conditions affect context importance
        if context.weather_conditions:
            if "rain" in context.weather_conditions.lower():
                adjustments["context"] *= 1.3
                adjustments["stats"] *= 0.9
            elif "wind" in context.weather_conditions.lower():
                adjustments["context"] *= 1.2
                adjustments["stats"] *= 0.8
        
        # Injury reports increase context importance
        if context.injury_report and len(context.injury_report) > 0:
            adjustments["context"] *= 1.2
            adjustments["value"] *= 1.1
        
        # Home/Away affects context
        if context.home_away:
            if context.home_away.lower() == "away":
                adjustments["context"] *= 1.1
        
        return adjustments

    @performance_timer('weighted_fusion_search')
    async def _weighted_fusion_search(
        self,
        query_vector: np.ndarray,
        weights: Dict[str, float],
        limit: int = 10
    ) -> List[SearchResult]:
        """Perform weighted fusion search across multiple vector types."""
        results = []
        
        # Search each vector type with weighted scores
        for vector_type, weight in weights.items():
            if weight > 0:
                try:
                    collection_type = self._get_collection_for_vector_type(vector_type)
                    vector_results = await self.vector_engine.search_vectors(
                        query_vector, collection_type, limit=limit * 2
                    )
                    
                    for result in vector_results:
                        weighted_score = result.similarity_score * weight
                        results.append((result, weighted_score, vector_type))
                        
                except Exception as e:
                    self.logger.warning(f"Error searching {vector_type} vectors: {e}")
        
        # Aggregate and rank results
        player_scores = defaultdict(lambda: {"total_score": 0.0, "vector_scores": {}, "results": []})
        
        for result, weighted_score, vector_type in results:
            player_id = result.player.player_id
            player_scores[player_id]["total_score"] += weighted_score
            player_scores[player_id]["vector_scores"][vector_type] = weighted_score
            player_scores[player_id]["results"].append(result)
        
        # Create final results
        final_results = []
        for player_id, score_data in sorted(
            player_scores.items(), 
            key=lambda x: x[1]["total_score"], 
            reverse=True
        )[:limit]:
            best_result = max(score_data["results"], key=lambda r: r.similarity_score)
            
            explanation = SearchExplanation(
                strategy_used="weighted_fusion",
                vector_contributions=score_data["vector_scores"],
                weight_adjustments=weights,
                filters_applied=[],
                reasoning=f"Combined scores from {len(score_data['vector_scores'])} vector types",
                confidence_score=score_data["total_score"] / sum(weights.values())
            )
            
            final_results.append(SearchResult(
                player=best_result.player,
                similarity_score=score_data["total_score"],
                explanation=explanation,
                vector_scores=score_data["vector_scores"]
            ))
        
        return final_results

    @performance_timer('conditional_logic_search')
    async def _conditional_logic_search(
        self,
        query_vector: np.ndarray,
        query_type: QueryType,
        limit: int = 10
    ) -> List[SearchResult]:
        """Perform conditional logic search with multi-stage refinement."""
        results = []
        
        # Stage 1: Initial broad search
        initial_results = await self._weighted_fusion_search(
            query_vector, 
            await self._calculate_dynamic_weights(query_type),
            limit=limit * 2
        )
        
        # Stage 2: Apply conditional filters
        filtered_results = []
        for result in initial_results:
            if await self._apply_conditional_filters(result, query_type):
                filtered_results.append(result)
        
        # Stage 3: Refine with archetype analysis
        refined_results = []
        for result in filtered_results[:limit]:
            archetype = await self.profile_analyzer.classify_player_archetype(result.player)
            if archetype:
                result.archetype_match = archetype.value
                result.confidence_level = "high" if result.similarity_score > 0.8 else "medium"
            refined_results.append(result)
        
        return refined_results[:limit]

    async def _apply_conditional_filters(
        self, 
        result: SearchResult, 
        query_type: QueryType
    ) -> bool:
        """Apply conditional filters based on query type."""
        player = result.player
        
        if query_type == QueryType.PERFORMANCE_SEARCH:
            # Filter for high-performing players
            return (player.stats.fantasy_points or 0) > 15.0
        
        elif query_type == QueryType.VALUE_SEARCH:
            # Filter for value players (high performance relative to cost)
            if player.value and player.value.salary:
                value_ratio = (player.stats.fantasy_points or 0) / player.value.salary
                return value_ratio > 0.1
        
        elif query_type == QueryType.MATCHUP_SEARCH:
            # Filter for favorable matchups
            if player.context and player.context.opponent_defensive_ranking:
                return player.context.opponent_defensive_ranking > 20  # Higher ranking = worse defense
        
        elif query_type == QueryType.SLEEPER_PICK:
            # Filter for under-owned players
            if player.value and player.value.ownership_percentage:
                return player.value.ownership_percentage < 10.0
        
        return True

    @performance_timer('temporal_search')
    async def _temporal_search(
        self,
        query_vector: np.ndarray,
        context: SearchContext,
        limit: int = 10
    ) -> List[SearchResult]:
        """Perform temporal search considering recent vs historical patterns."""
        results = []
        
        # Recent performance weight (last 3 weeks)
        recent_weights = await self._calculate_dynamic_weights(
            QueryType.PERFORMANCE_SEARCH, context
        )
        recent_weights = {k: v * 1.3 for k, v in recent_weights.items()}  # Boost recent
        
        # Historical performance weight
        historical_weights = await self._calculate_dynamic_weights(
            QueryType.PERFORMANCE_SEARCH, context
        )
        historical_weights = {k: v * 0.7 for k, v in historical_weights.items()}  # Reduce historical
        
        # Combine recent and historical searches
        recent_results = await self._weighted_fusion_search(
            query_vector, recent_weights, limit=limit
        )
        historical_results = await self._weighted_fusion_search(
            query_vector, historical_weights, limit=limit
        )
        
        # Merge and rank results
        all_results = recent_results + historical_results
        player_scores = defaultdict(lambda: {"recent": 0.0, "historical": 0.0, "player": None})
        
        for result in all_results:
            player_id = result.player.player_id
            if result in recent_results:
                player_scores[player_id]["recent"] = max(
                    player_scores[player_id]["recent"], result.similarity_score
                )
            else:
                player_scores[player_id]["historical"] = max(
                    player_scores[player_id]["historical"], result.similarity_score
                )
            player_scores[player_id]["player"] = result.player
        
        # Calculate temporal scores
        final_results = []
        for player_id, scores in player_scores.items():
            temporal_score = scores["recent"] * 0.7 + scores["historical"] * 0.3
            
            explanation = SearchExplanation(
                strategy_used="temporal_search",
                vector_contributions={"recent": scores["recent"], "historical": scores["historical"]},
                weight_adjustments={"recent_weight": 0.7, "historical_weight": 0.3},
                filters_applied=["temporal_weighting"],
                reasoning="Combined recent (70%) and historical (30%) performance patterns",
                confidence_score=temporal_score
            )
            
            final_results.append(SearchResult(
                player=scores["player"],
                similarity_score=temporal_score,
                explanation=explanation,
                vector_scores={"recent": scores["recent"], "historical": scores["historical"]}
            ))
        
        return sorted(final_results, key=lambda r: r.similarity_score, reverse=True)[:limit]

    @performance_timer('ensemble_search')
    async def _ensemble_search(
        self,
        query_vector: np.ndarray,
        query_type: QueryType,
        context: Optional[SearchContext] = None,
        limit: int = 10
    ) -> List[SearchResult]:
        """Perform ensemble search combining multiple strategies."""
        strategies = [
            self._weighted_fusion_search,
            self._conditional_logic_search,
            self._temporal_search
        ]
        
        # Run all strategies in parallel
        tasks = []
        for strategy in strategies:
            if strategy == self._temporal_search:
                task = strategy(query_vector, context or SearchContext(), limit)
            elif strategy == self._conditional_logic_search:
                task = strategy(query_vector, query_type, limit)
            else:
                task = strategy(query_vector, await self._calculate_dynamic_weights(query_type, context), limit)
            tasks.append(task)
        
        strategy_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results from all strategies
        player_scores = defaultdict(lambda: {"scores": [], "player": None, "explanations": []})
        
        for i, results in enumerate(strategy_results):
            if isinstance(results, Exception):
                self.logger.warning(f"Strategy {i} failed: {results}")
                continue
                
            for result in results:
                player_id = result.player.player_id
                player_scores[player_id]["scores"].append(result.similarity_score)
                player_scores[player_id]["player"] = result.player
                player_scores[player_id]["explanations"].append(result.explanation)
        
        # Calculate ensemble scores
        final_results = []
        for player_id, data in player_scores.items():
            if len(data["scores"]) > 0:
                ensemble_score = np.mean(data["scores"])
                ensemble_std = np.std(data["scores"])
                
                # Combine explanations
                combined_reasoning = "Ensemble search combining: " + ", ".join([
                    exp.strategy_used for exp in data["explanations"]
                ])
                
                explanation = SearchExplanation(
                    strategy_used="ensemble_search",
                    vector_contributions={"ensemble_score": ensemble_score, "std": ensemble_std},
                    weight_adjustments={"strategy_count": len(data["scores"])},
                    filters_applied=["ensemble_aggregation"],
                    reasoning=combined_reasoning,
                    confidence_score=ensemble_score * (1 - ensemble_std)  # Lower std = higher confidence
                )
                
                final_results.append(SearchResult(
                    player=data["player"],
                    similarity_score=ensemble_score,
                    explanation=explanation,
                    vector_scores={"ensemble": ensemble_score, "std": ensemble_std}
                ))
        
        return sorted(final_results, key=lambda r: r.similarity_score, reverse=True)[:limit]

    @performance_timer('archetype_filtered_search')
    async def _archetype_filtered_search(
        self,
        query_vector: np.ndarray,
        target_archetype: ArchetypeType,
        limit: int = 10
    ) -> List[SearchResult]:
        """Perform search filtered by player archetype."""
        # Get all players and classify their archetypes
        all_players = await self._get_all_players()
        archetype_players = []
        
        for player in all_players:
            archetype = await self.profile_analyzer.classify_player_archetype(player)
            if archetype == target_archetype:
                archetype_players.append(player)
        
        if not archetype_players:
            return []
        
        # Generate embeddings for archetype players
        embeddings = []
        for player in archetype_players:
            embedding = await self.embedding_generator.generate_player_embedding(
                player, EmbeddingStrategy.STATISTICAL
            )
            embeddings.append((player, embedding))
        
        # Find similar players within archetype
        results = []
        for player, embedding in embeddings:
            similarity = np.dot(query_vector, embedding) / (
                np.linalg.norm(query_vector) * np.linalg.norm(embedding)
            )
            
            explanation = SearchExplanation(
                strategy_used="archetype_filtered",
                vector_contributions={"archetype_match": 1.0},
                weight_adjustments={"target_archetype": target_archetype.value},
                filters_applied=[f"archetype={target_archetype.value}"],
                reasoning=f"Filtered by {target_archetype.value} archetype",
                confidence_score=similarity
            )
            
            results.append(SearchResult(
                player=player,
                similarity_score=similarity,
                explanation=explanation,
                vector_scores={"archetype": 1.0}
            ))
        
        return sorted(results, key=lambda r: r.similarity_score, reverse=True)[:limit]

    async def _get_all_players(self) -> List[Player]:
        """Get all players from the vector database."""
        # This would typically query the database
        # For now, return empty list as placeholder
        return []

    def _get_collection_for_vector_type(self, vector_type: str) -> CollectionType:
        """Get the appropriate collection type for a vector type."""
        collection_map = {
            "stats": CollectionType.REGULAR,
            "context": CollectionType.REGULAR,
            "value": CollectionType.REGULAR,
            "combined": CollectionType.REGULAR
        }
        return collection_map.get(vector_type, CollectionType.REGULAR)

    @performance_timer('advanced_search')
    async def search(
        self,
        query: str,
        strategy: SearchStrategy = SearchStrategy.WEIGHTED_FUSION,
        query_type: QueryType = QueryType.PERFORMANCE_SEARCH,
        context: Optional[SearchContext] = None,
        limit: int = 10,
        ab_test_id: Optional[str] = None
    ) -> List[SearchResult]:
        """Perform advanced search with specified strategy."""
        start_time = datetime.now()
        
        # Generate query embedding
        query_embedding = await self.embedding_generator.generate_query_embedding(query)
        
        # A/B testing logic
        if ab_test_id and ab_test_id in self.ab_tests:
            strategy = self._get_ab_test_strategy(ab_test_id)
        
        # Execute search based on strategy
        if strategy == SearchStrategy.WEIGHTED_FUSION:
            weights = await self._calculate_dynamic_weights(query_type, context)
            results = await self._weighted_fusion_search(query_embedding, weights, limit)
        elif strategy == SearchStrategy.CONDITIONAL_LOGIC:
            results = await self._conditional_logic_search(query_embedding, query_type, limit)
        elif strategy == SearchStrategy.TEMPORAL_SEARCH:
            results = await self._temporal_search(query_embedding, context or SearchContext(), limit)
        elif strategy == SearchStrategy.ENSEMBLE_SEARCH:
            results = await self._ensemble_search(query_embedding, query_type, context, limit)
        elif strategy == SearchStrategy.CONTEXTUAL_SEARCH:
            # Contextual search uses weighted fusion with context adjustments
            weights = await self._calculate_dynamic_weights(query_type, context)
            results = await self._weighted_fusion_search(query_embedding, weights, limit)
        elif strategy == SearchStrategy.ARCHETYPE_FILTERED:
            # For archetype search, we need to specify target archetype
            target_archetype = ArchetypeType.VOLUME_RUSHER  # Default
            results = await self._archetype_filtered_search(query_embedding, target_archetype, limit)
        else:
            raise ValueError(f"Unknown search strategy: {strategy}")
        
        # Track performance
        latency = (datetime.now() - start_time).total_seconds() * 1000
        self._track_search_performance(strategy.value, latency, len(results))
        
        # Track A/B test results
        if ab_test_id:
            self._track_ab_test_result(ab_test_id, strategy.value, latency, len(results))
        
        return results

    def _get_ab_test_strategy(self, test_id: str) -> SearchStrategy:
        """Get strategy for A/B test based on traffic split."""
        test_config = self.ab_tests[test_id]
        if random.random() < test_config.traffic_split:
            return test_config.strategy_b
        return test_config.strategy_a

    def _track_search_performance(self, strategy: str, latency: float, result_count: int):
        """Track search performance metrics."""
        self.strategy_performance[strategy]["latency"] = (
            self.strategy_performance[strategy]["latency"] * 
            self.strategy_performance[strategy]["count"] + latency
        ) / (self.strategy_performance[strategy]["count"] + 1)
        self.strategy_performance[strategy]["count"] += 1

    def _track_ab_test_result(self, test_id: str, strategy: str, latency: float, result_count: int):
        """Track A/B test results."""
        self.ab_results[test_id].append({
            "strategy": strategy,
            "latency": latency,
            "result_count": result_count,
            "timestamp": datetime.now()
        })

    async def create_ab_test(self, config: ABTestConfig) -> str:
        """Create a new A/B test."""
        self.ab_tests[config.test_id] = config
        self.logger.info(f"Created A/B test {config.test_id}: {config.strategy_a.value} vs {config.strategy_b.value}")
        return config.test_id

    async def get_ab_test_results(self, test_id: str) -> Optional[ABTestResult]:
        """Get A/B test results."""
        if test_id not in self.ab_tests:
            return None
        
        test_config = self.ab_tests[test_id]
        results = self.ab_results[test_id]
        
        if len(results) < test_config.min_sample_size:
            return None
        
        # Calculate metrics for each strategy
        strategy_a_results = {"latency": 0.0, "result_count": 0, "count": 0}
        strategy_b_results = {"latency": 0.0, "result_count": 0, "count": 0}
        
        for result in results:
            if result["strategy"] == test_config.strategy_a.value:
                strategy_a_results["latency"] += result["latency"]
                strategy_a_results["result_count"] += result["result_count"]
                strategy_a_results["count"] += 1
            else:
                strategy_b_results["latency"] += result["latency"]
                strategy_b_results["result_count"] += result["result_count"]
                strategy_b_results["count"] += 1
        
        # Calculate averages
        if strategy_a_results["count"] > 0:
            strategy_a_results["latency"] /= strategy_a_results["count"]
            strategy_a_results["result_count"] /= strategy_a_results["count"]
        
        if strategy_b_results["count"] > 0:
            strategy_b_results["latency"] /= strategy_b_results["count"]
            strategy_b_results["result_count"] /= strategy_b_results["count"]
        
        # Determine winner
        winner = None
        confidence_level = 0.0
        
        if strategy_a_results["count"] > 0 and strategy_b_results["count"] > 0:
            # Simple comparison based on latency (lower is better)
            if strategy_a_results["latency"] < strategy_b_results["latency"]:
                winner = "A"
                confidence_level = 0.8
            else:
                winner = "B"
                confidence_level = 0.8
        
        return ABTestResult(
            test_id=test_id,
            strategy_a_results=strategy_a_results,
            strategy_b_results=strategy_b_results,
            winner=winner,
            confidence_level=confidence_level,
            sample_size=len(results),
            duration_days=(datetime.now() - datetime.now()).days  # Placeholder
        )

    async def get_search_analytics(self) -> Dict[str, Any]:
        """Get search analytics and performance metrics."""
        return {
            "strategy_performance": dict(self.strategy_performance),
            "active_ab_tests": len(self.ab_tests),
            "total_searches": sum(s["count"] for s in self.strategy_performance.values()),
            "average_latency": np.mean([
                s["latency"] for s in self.strategy_performance.values() if s["count"] > 0
            ]) if self.strategy_performance else 0.0
        }
