"""
Player Profile Analytics for DFS Prophet

Provides comprehensive multi-vector player analysis including:
- Player archetype classification
- Multi-dimensional similarity scoring
- Vector contribution analysis
- Player cluster identification
- Anomaly detection
- Performance prediction
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import zscore
from scipy.spatial.distance import cosine
import warnings

from ..config import get_settings
from ..utils import get_logger, performance_timer
from ..data.models import Player, Position, Team
from ..core import get_vector_engine, get_embedding_generator, CollectionType


class ArchetypeType(Enum):
    """Player archetype classifications."""
    VOLUME_RUSHER = "volume_rusher"
    RED_ZONE_TARGET = "red_zone_target"
    DEEP_THREAT = "deep_threat"
    POSSESSION_RECEIVER = "possession_receiver"
    ELITE_QB = "elite_qb"
    GAME_MANAGER = "game_manager"
    HIGH_VOLUME_RB = "high_volume_rb"
    THIRD_DOWN_BACK = "third_down_back"
    TIGHT_END_RED_ZONE = "tight_end_red_zone"
    KICKER_ACCURATE = "kicker_accurate"
    DEFENSE_ELITE = "defense_elite"
    VALUE_PLAY = "value_play"
    ANOMALY = "anomaly"


class AnalysisType(Enum):
    """Types of player analysis."""
    ARCHETYPE = "archetype"
    SIMILARITY = "similarity"
    CONTRIBUTION = "contribution"
    CLUSTER = "cluster"
    ANOMALY = "anomaly"
    PREDICTION = "prediction"
    MATCHUP = "matchup"
    VALUE = "value"


@dataclass
class VectorContribution:
    """Vector contribution analysis result."""
    vector_type: str
    contribution_score: float
    contribution_percentage: float
    explanation: str
    key_features: List[str]


@dataclass
class ArchetypeResult:
    """Player archetype classification result."""
    archetype: ArchetypeType
    confidence_score: float
    primary_vectors: List[str]
    secondary_vectors: List[str]
    archetype_features: Dict[str, float]
    explanation: str


@dataclass
class SimilarityResult:
    """Multi-dimensional similarity analysis result."""
    player_id: str
    player_name: str
    overall_similarity: float
    vector_similarities: Dict[str, float]
    weighted_similarity: float
    rank: int
    explanation: str


@dataclass
class ClusterResult:
    """Player cluster analysis result."""
    cluster_id: int
    cluster_size: int
    cluster_center: Dict[str, float]
    cluster_players: List[str]
    cluster_archetype: Optional[ArchetypeType]
    cluster_characteristics: Dict[str, float]


@dataclass
class AnomalyResult:
    """Anomaly detection result."""
    anomaly_score: float
    anomaly_type: str
    anomalous_vectors: List[str]
    normal_range: Dict[str, Tuple[float, float]]
    explanation: str
    recommendations: List[str]


@dataclass
class PredictionResult:
    """Performance prediction result."""
    predicted_fantasy_points: float
    confidence_interval: Tuple[float, float]
    prediction_factors: Dict[str, float]
    risk_factors: List[str]
    upside_potential: float
    downside_risk: float


@dataclass
class MatchupAnalysis:
    """Matchup advantage analysis."""
    opponent_team: str
    advantage_score: float
    favorable_vectors: List[str]
    unfavorable_vectors: List[str]
    matchup_strategy: str
    confidence: float


@dataclass
class ValueOpportunity:
    """Value opportunity identification."""
    opportunity_type: str
    opportunity_score: float
    undervalued_vectors: List[str]
    market_mispricing: Dict[str, float]
    expected_return: float
    risk_level: str


@dataclass
class PlayerProfile:
    """Complete player profile analysis."""
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


class PlayerProfileAnalyzer:
    """Comprehensive player profile analyzer using multi-vector data."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.vector_engine = get_vector_engine()
        self.embedding_generator = get_embedding_generator()
        
        # Analysis cache
        self._archetype_cache: Dict[str, ArchetypeResult] = {}
        self._similarity_cache: Dict[str, List[SimilarityResult]] = {}
        self._cluster_cache: Optional[Dict[str, Any]] = None
        
        # Configuration
        self.vector_weights = self.settings.get_vector_weights()
        self.performance_thresholds = self.settings.get_performance_thresholds()
        
        # Archetype definitions
        self.archetype_definitions = self._initialize_archetype_definitions()
        
        # Analysis parameters
        self.max_similar_players = 10
        self.cluster_min_size = 3
        self.anomaly_threshold = 2.0  # Z-score threshold
        self.prediction_horizon = 3  # weeks
        
    def _initialize_archetype_definitions(self) -> Dict[ArchetypeType, Dict[str, Any]]:
        """Initialize archetype definitions with vector patterns."""
        return {
            ArchetypeType.VOLUME_RUSHER: {
                "description": "High-volume running back with consistent carries",
                "primary_vectors": ["stats", "value"],
                "key_features": ["rushing_attempts", "rushing_yards", "touchdowns"],
                "vector_patterns": {
                    "stats": {"rushing_attempts": 0.8, "rushing_yards": 0.7},
                    "value": {"salary_efficiency": 0.6, "consistency": 0.7}
                }
            },
            ArchetypeType.RED_ZONE_TARGET: {
                "description": "Player frequently targeted in red zone situations",
                "primary_vectors": ["stats", "context"],
                "key_features": ["red_zone_targets", "touchdowns", "target_share"],
                "vector_patterns": {
                    "stats": {"touchdowns": 0.8, "red_zone_targets": 0.9},
                    "context": {"red_zone_opportunity": 0.8}
                }
            },
            ArchetypeType.DEEP_THREAT: {
                "description": "Wide receiver specializing in deep passes",
                "primary_vectors": ["stats", "context"],
                "key_features": ["yards_per_reception", "deep_targets", "air_yards"],
                "vector_patterns": {
                    "stats": {"yards_per_reception": 0.8, "air_yards": 0.7},
                    "context": {"deep_route_percentage": 0.8}
                }
            },
            ArchetypeType.ELITE_QB: {
                "description": "Elite quarterback with high fantasy production",
                "primary_vectors": ["stats", "value"],
                "key_features": ["passing_yards", "touchdowns", "fantasy_points"],
                "vector_patterns": {
                    "stats": {"fantasy_points": 0.9, "passing_yards": 0.8},
                    "value": {"salary_efficiency": 0.7, "upside": 0.8}
                }
            },
            ArchetypeType.VALUE_PLAY: {
                "description": "Undervalued player with high potential",
                "primary_vectors": ["value", "context"],
                "key_features": ["salary_efficiency", "ownership_projection", "upside"],
                "vector_patterns": {
                    "value": {"salary_efficiency": 0.8, "upside": 0.7},
                    "context": {"favorable_matchup": 0.6}
                }
            },
            ArchetypeType.ANOMALY: {
                "description": "Player with unusual vector combinations",
                "primary_vectors": ["stats", "context", "value"],
                "key_features": ["anomaly_score", "vector_discordance"],
                "vector_patterns": {
                    "stats": {"anomaly": 0.8},
                    "context": {"anomaly": 0.8},
                    "value": {"anomaly": 0.8}
                }
            }
        }
    
    @performance_timer('analyze_player_profile')
    async def analyze_player_profile(
        self, 
        player_id: str,
        include_similarities: bool = True,
        include_clusters: bool = True,
        include_predictions: bool = True
    ) -> PlayerProfile:
        """Perform comprehensive player profile analysis."""
        self.logger.info(f"Starting comprehensive analysis for player {player_id}")
        
        # Get player data
        player = await self._get_player_data(player_id)
        if not player:
            raise ValueError(f"Player {player_id} not found")
        
        # Generate embeddings for analysis
        embeddings = await self._generate_player_embeddings(player)
        
        # Perform individual analyses
        archetype = await self._classify_archetype(player, embeddings)
        vector_contributions = await self._analyze_vector_contributions(player, embeddings)
        
        # Optional analyses
        top_similarities = []
        if include_similarities:
            top_similarities = await self._find_similar_players(player, embeddings)
        
        cluster_info = None
        if include_clusters:
            cluster_info = await self._identify_cluster(player, embeddings)
        
        anomaly_info = await self._detect_anomalies(player, embeddings)
        
        predictions = None
        if include_predictions:
            predictions = await self._predict_performance(player, embeddings)
        
        matchups = await self._analyze_matchups(player, embeddings)
        value_opportunities = await self._identify_value_opportunities(player, embeddings)
        
        # Prepare visualization data
        visualization_data = await self._prepare_visualization_data(
            player, embeddings, archetype, vector_contributions
        )
        
        # Create complete profile
        profile = PlayerProfile(
            player_id=player_id,
            player_name=player.name,
            position=player.position,
            team=player.team,
            archetype=archetype,
            vector_contributions=vector_contributions,
            top_similarities=top_similarities,
            cluster_info=cluster_info,
            anomaly_info=anomaly_info,
            predictions=predictions,
            matchups=matchups,
            value_opportunities=value_opportunities,
            visualization_data=visualization_data,
            analysis_timestamp=datetime.now()
        )
        
        self.logger.info(f"Completed analysis for player {player_id}")
        return profile
    
    async def _get_player_data(self, player_id: str) -> Optional[Player]:
        """Retrieve player data from vector engine."""
        try:
            # Search for player in vector engine
            results = await self.vector_engine.search_vectors(
                query_vector=[0.0] * 768,  # Dummy query
                collection_type=CollectionType.REGULAR,
                limit=1000  # Get all players
            )
            
            # Find the specific player
            for result in results:
                if result.player.player_id == player_id:
                    return result.player
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving player data: {e}")
            return None
    
    async def _generate_player_embeddings(self, player: Player) -> Dict[str, List[float]]:
        """Generate multi-vector embeddings for player."""
        embeddings = {}
        
        try:
            # Generate embeddings for each vector type
            from ..core.embedding_generator import EmbeddingStrategy
            
            # Statistical embedding
            stats_embedding = await self.embedding_generator.generate_player_embedding(
                player, strategy=EmbeddingStrategy.STATISTICAL
            )
            embeddings["stats"] = stats_embedding
            
            # Contextual embedding
            context_embedding = await self.embedding_generator.generate_player_embedding(
                player, strategy=EmbeddingStrategy.CONTEXTUAL
            )
            embeddings["context"] = context_embedding
            
            # Value embedding
            value_embedding = await self.embedding_generator.generate_player_embedding(
                player, strategy=EmbeddingStrategy.HYBRID
            )
            embeddings["value"] = value_embedding
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            # Return zero vectors as fallback
            return {
                "stats": [0.0] * 768,
                "context": [0.0] * 768,
                "value": [0.0] * 768
            }
    
    async def _classify_archetype(
        self, 
        player: Player, 
        embeddings: Dict[str, List[float]]
    ) -> ArchetypeResult:
        """Classify player archetype based on vector patterns."""
        cache_key = f"{player.player_id}_archetype"
        
        if cache_key in self._archetype_cache:
            return self._archetype_cache[cache_key]
        
        try:
            best_archetype = ArchetypeType.ANOMALY
            best_confidence = 0.0
            best_features = {}
            
            # Calculate similarity to each archetype pattern
            for archetype, definition in self.archetype_definitions.items():
                if archetype == ArchetypeType.ANOMALY:
                    continue
                
                confidence, features = await self._calculate_archetype_similarity(
                    player, embeddings, definition
                )
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_archetype = archetype
                    best_features = features
            
            # Check if player is an anomaly
            anomaly_score = await self._calculate_anomaly_score(player, embeddings)
            if anomaly_score > 0.7:  # High anomaly score
                best_archetype = ArchetypeType.ANOMALY
                best_confidence = anomaly_score
            
            # Determine primary and secondary vectors
            primary_vectors = self.archetype_definitions[best_archetype]["primary_vectors"]
            secondary_vectors = [v for v in ["stats", "context", "value"] if v not in primary_vectors]
            
            # Generate explanation
            explanation = self._generate_archetype_explanation(
                player, best_archetype, best_confidence, best_features
            )
            
            result = ArchetypeResult(
                archetype=best_archetype,
                confidence_score=best_confidence,
                primary_vectors=primary_vectors,
                secondary_vectors=secondary_vectors,
                archetype_features=best_features,
                explanation=explanation
            )
            
            # Cache result
            self._archetype_cache[cache_key] = result
            return result
            
        except Exception as e:
            self.logger.error(f"Error classifying archetype: {e}")
            return ArchetypeResult(
                archetype=ArchetypeType.ANOMALY,
                confidence_score=0.0,
                primary_vectors=[],
                secondary_vectors=[],
                archetype_features={},
                explanation="Error in archetype classification"
            )
    
    async def _calculate_archetype_similarity(
        self,
        player: Player,
        embeddings: Dict[str, List[float]],
        definition: Dict[str, Any]
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate similarity to archetype pattern."""
        try:
            total_confidence = 0.0
            feature_scores = {}
            
            # Calculate feature-based similarity
            for feature in definition.get("key_features", []):
                score = self._calculate_feature_score(player, feature)
                feature_scores[feature] = score
                total_confidence += score
            
            # Calculate vector pattern similarity
            vector_patterns = definition.get("vector_patterns", {})
            for vector_type, patterns in vector_patterns.items():
                if vector_type in embeddings:
                    vector_similarity = self._calculate_vector_pattern_similarity(
                        embeddings[vector_type], patterns
                    )
                    total_confidence += vector_similarity
            
            # Normalize confidence score
            max_possible = len(definition.get("key_features", [])) + len(vector_patterns)
            normalized_confidence = total_confidence / max_possible if max_possible > 0 else 0.0
            
            return normalized_confidence, feature_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating archetype similarity: {e}")
            return 0.0, {}
    
    def _calculate_feature_score(self, player: Player, feature: str) -> float:
        """Calculate score for a specific feature."""
        try:
            # Map features to player attributes with safe defaults
            feature_mapping = {
                "rushing_attempts": (player.stats.rushing_attempts or 0) / 25.0,  # Normalize
                "rushing_yards": (player.stats.rushing_yards or 0) / 100.0,
                "touchdowns": (player.stats.total_touchdowns or 0) / 10.0,
                "red_zone_targets": getattr(player.stats, 'red_zone_targets', 0) / 5.0,
                "target_share": getattr(player.stats, 'target_share', 0) / 0.3,
                "yards_per_reception": (player.stats.yards_per_reception or 0) / 15.0,
                "deep_targets": getattr(player.stats, 'deep_targets', 0) / 5.0,
                "air_yards": getattr(player.stats, 'air_yards', 0) / 100.0,
                "passing_yards": (player.stats.passing_yards or 0) / 300.0,
                "fantasy_points": (player.stats.fantasy_points or 0) / 25.0,
                "salary_efficiency": getattr(player.dfs, 'salary_efficiency', 0) / 0.1,
                "consistency": getattr(player.dfs, 'consistency', 0) / 0.8,
                "upside": getattr(player.dfs, 'upside', 0) / 0.2,
                "ownership_projection": getattr(player.dfs, 'ownership_projection', 0) / 0.2
            }
            
            score = feature_mapping.get(feature, 0.0)
            return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating feature score: {e}")
            return 0.0
    
    def _calculate_vector_pattern_similarity(
        self, 
        embedding: List[float], 
        patterns: Dict[str, float]
    ) -> float:
        """Calculate similarity between embedding and pattern."""
        try:
            # Simple cosine similarity with pattern weights
            # This is a simplified version - in practice, you'd use more sophisticated methods
            pattern_vector = [patterns.get(f"feature_{i}", 0.0) for i in range(len(embedding))]
            
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(embedding, pattern_vector))
            norm_a = sum(a * a for a in embedding) ** 0.5
            norm_b = sum(b * b for b in pattern_vector) ** 0.5
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            similarity = dot_product / (norm_a * norm_b)
            return max(0.0, similarity)  # Ensure non-negative
            
        except Exception as e:
            self.logger.error(f"Error calculating vector pattern similarity: {e}")
            return 0.0
    
    async def _calculate_anomaly_score(
        self, 
        player: Player, 
        embeddings: Dict[str, List[float]]
    ) -> float:
        """Calculate anomaly score for player."""
        try:
            # Get all players for comparison
            all_players = await self._get_all_players()
            
            if len(all_players) < 10:  # Need minimum players for comparison
                return 0.0
            
            # Calculate vector distances to other players
            distances = []
            for other_player in all_players:
                if other_player.player_id != player.player_id:
                    other_embeddings = await self._generate_player_embeddings(other_player)
                    
                    for vector_type in ["stats", "context", "value"]:
                        if vector_type in embeddings and vector_type in other_embeddings:
                            distance = cosine(embeddings[vector_type], other_embeddings[vector_type])
                            distances.append(distance)
            
            if not distances:
                return 0.0
            
            # Calculate anomaly score based on distance distribution
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            
            if std_distance == 0:
                return 0.0
            
            # Calculate how far this player is from the mean
            player_distances = []
            for vector_type in ["stats", "context", "value"]:
                if vector_type in embeddings:
                    vector_distances = [cosine(embeddings[vector_type], other_emb) 
                                      for other_emb in [emb for emb in distances if len(emb) == len(embeddings[vector_type])]]
                    if vector_distances:
                        player_distances.extend(vector_distances)
            
            if not player_distances:
                return 0.0
            
            avg_player_distance = np.mean(player_distances)
            z_score = abs(avg_player_distance - mean_distance) / std_distance
            
            # Convert to 0-1 scale
            anomaly_score = min(z_score / self.anomaly_threshold, 1.0)
            
            return anomaly_score
            
        except Exception as e:
            self.logger.error(f"Error calculating anomaly score: {e}")
            return 0.0
    
    def _generate_archetype_explanation(
        self,
        player: Player,
        archetype: ArchetypeType,
        confidence: float,
        features: Dict[str, float]
    ) -> str:
        """Generate human-readable explanation for archetype classification."""
        try:
            archetype_info = self.archetype_definitions[archetype]
            description = archetype_info["description"]
            
            # Get top features
            top_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:3]
            
            explanation = f"{player.name} is classified as a {archetype.value.replace('_', ' ')} "
            explanation += f"({description}) with {confidence:.1%} confidence. "
            
            if top_features:
                explanation += "Key characteristics include: "
                feature_descriptions = []
                for feature, score in top_features:
                    if score > 0.5:  # Only mention significant features
                        feature_descriptions.append(f"{feature} ({score:.1%})")
                
                explanation += ", ".join(feature_descriptions)
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating archetype explanation: {e}")
            return f"{player.name} archetype classification completed with {confidence:.1%} confidence."

    async def _analyze_vector_contributions(
        self, 
        player: Player, 
        embeddings: Dict[str, List[float]]
    ) -> List[VectorContribution]:
        """Analyze vector contributions to player profile."""
        try:
            contributions = []
            
            # Calculate contribution for each vector type
            for vector_type in ["stats", "context", "value"]:
                if vector_type not in embeddings:
                    continue
                
                # Calculate vector magnitude as contribution score
                vector_magnitude = np.linalg.norm(embeddings[vector_type])
                max_possible_magnitude = np.sqrt(len(embeddings[vector_type]))  # Normalized vector
                contribution_score = vector_magnitude / max_possible_magnitude
                
                # Calculate contribution percentage
                total_magnitude = sum(np.linalg.norm(emb) for emb in embeddings.values())
                contribution_percentage = vector_magnitude / total_magnitude if total_magnitude > 0 else 0.0
                
                # Identify key features for this vector type
                key_features = self._identify_key_features(player, vector_type)
                
                # Generate explanation
                explanation = self._generate_vector_explanation(player, vector_type, contribution_score, key_features)
                
                contribution = VectorContribution(
                    vector_type=vector_type,
                    contribution_score=contribution_score,
                    contribution_percentage=contribution_percentage,
                    explanation=explanation,
                    key_features=key_features
                )
                contributions.append(contribution)
            
            return contributions
            
        except Exception as e:
            self.logger.error(f"Error analyzing vector contributions: {e}")
            return []
    
    def _identify_key_features(self, player: Player, vector_type: str) -> List[str]:
        """Identify key features for a vector type."""
        try:
            if vector_type == "stats":
                features = []
                if (player.stats.rushing_attempts or 0) > 15:
                    features.append("high_rush_volume")
                if (player.stats.fantasy_points or 0) > 20:
                    features.append("high_fantasy_production")
                if (player.stats.total_touchdowns or 0) > 1:
                    features.append("touchdown_scorer")
                if (player.stats.passing_yards or 0) > 250:
                    features.append("passing_volume")
                return features[:3]  # Top 3 features
                
            elif vector_type == "context":
                features = []
                # Add context-specific features based on available data
                features.append("game_context")
                features.append("opponent_matchup")
                features.append("team_scheme")
                return features
                
            elif vector_type == "value":
                features = []
                if hasattr(player.dfs, 'salary_efficiency') and player.dfs.salary_efficiency > 0.1:
                    features.append("salary_efficient")
                if hasattr(player.dfs, 'upside') and player.dfs.upside > 0.15:
                    features.append("high_upside")
                if hasattr(player.dfs, 'consistency') and player.dfs.consistency > 0.7:
                    features.append("consistent")
                return features[:3]
                
            return []
            
        except Exception as e:
            self.logger.error(f"Error identifying key features: {e}")
            return []
    
    def _generate_vector_explanation(
        self, 
        player: Player, 
        vector_type: str, 
        contribution_score: float, 
        key_features: List[str]
    ) -> str:
        """Generate explanation for vector contribution."""
        try:
            if vector_type == "stats":
                explanation = f"Statistical performance contributes {contribution_score:.1%} to {player.name}'s profile. "
                if key_features:
                    explanation += f"Key statistical features: {', '.join(key_features)}."
                    
            elif vector_type == "context":
                explanation = f"Game context contributes {contribution_score:.1%} to {player.name}'s profile. "
                explanation += "Includes matchup factors, weather conditions, and team schemes."
                
            elif vector_type == "value":
                explanation = f"DFS value factors contribute {contribution_score:.1%} to {player.name}'s profile. "
                if key_features:
                    explanation += f"Key value features: {', '.join(key_features)}."
                    
            else:
                explanation = f"{vector_type.title()} vector contributes {contribution_score:.1%} to the profile."
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating vector explanation: {e}")
            return f"{vector_type.title()} vector analysis completed."
    
    async def _find_similar_players(
        self, 
        player: Player, 
        embeddings: Dict[str, List[float]]
    ) -> List[SimilarityResult]:
        """Find similar players using multi-vector similarity."""
        cache_key = f"{player.player_id}_similarities"
        
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        try:
            # Get all players for comparison
            all_players = await self._get_all_players()
            
            similarities = []
            for other_player in all_players:
                if other_player.player_id == player.player_id:
                    continue
                
                # Generate embeddings for comparison
                other_embeddings = await self._generate_player_embeddings(other_player)
                
                # Calculate vector similarities
                vector_similarities = {}
                for vector_type in ["stats", "context", "value"]:
                    if vector_type in embeddings and vector_type in other_embeddings:
                        similarity = 1 - cosine(embeddings[vector_type], other_embeddings[vector_type])
                        vector_similarities[vector_type] = max(0.0, similarity)
                
                # Calculate weighted overall similarity
                weighted_similarity = sum(
                    vector_similarities.get(vt, 0.0) * self.vector_weights.get(vt, 0.33)
                    for vt in ["stats", "context", "value"]
                )
                
                # Calculate unweighted overall similarity
                overall_similarity = np.mean(list(vector_similarities.values())) if vector_similarities else 0.0
                
                # Generate explanation
                explanation = self._generate_similarity_explanation(
                    player, other_player, vector_similarities, weighted_similarity
                )
                
                similarity_result = SimilarityResult(
                    player_id=other_player.player_id,
                    player_name=other_player.name,
                    overall_similarity=overall_similarity,
                    vector_similarities=vector_similarities,
                    weighted_similarity=weighted_similarity,
                    rank=0,  # Will be set after sorting
                    explanation=explanation
                )
                similarities.append(similarity_result)
            
            # Sort by weighted similarity and assign ranks
            similarities.sort(key=lambda x: x.weighted_similarity, reverse=True)
            for i, similarity in enumerate(similarities[:self.max_similar_players]):
                similarity.rank = i + 1
            
            # Cache results
            self._similarity_cache[cache_key] = similarities[:self.max_similar_players]
            return similarities[:self.max_similar_players]
            
        except Exception as e:
            self.logger.error(f"Error finding similar players: {e}")
            return []
    
    def _generate_similarity_explanation(
        self,
        player: Player,
        other_player: Player,
        vector_similarities: Dict[str, float],
        weighted_similarity: float
    ) -> str:
        """Generate explanation for player similarity."""
        try:
            explanation = f"{other_player.name} shows {weighted_similarity:.1%} similarity to {player.name}. "
            
            # Identify strongest similarity vector
            if vector_similarities:
                strongest_vector = max(vector_similarities.items(), key=lambda x: x[1])
                explanation += f"Strongest similarity in {strongest_vector[0]} vector ({strongest_vector[1]:.1%}). "
            
            # Add position context
            if player.position == other_player.position:
                explanation += f"Both are {player.position.value}s."
            else:
                explanation += f"Cross-position similarity: {player.position.value} vs {other_player.position.value}."
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating similarity explanation: {e}")
            return f"Similarity analysis completed for {other_player.name}."
    
    async def _identify_cluster(
        self, 
        player: Player, 
        embeddings: Dict[str, List[float]]
    ) -> Optional[ClusterResult]:
        """Identify player cluster using multi-vector data."""
        try:
            # Get all players for clustering
            all_players = await self._get_all_players()
            
            if len(all_players) < self.cluster_min_size:
                return None
            
            # Prepare data for clustering
            cluster_data = []
            player_ids = []
            
            for p in all_players:
                p_embeddings = await self._generate_player_embeddings(p)
                
                # Combine all vectors into a single feature vector
                combined_vector = []
                for vector_type in ["stats", "context", "value"]:
                    if vector_type in p_embeddings:
                        combined_vector.extend(p_embeddings[vector_type])
                
                if combined_vector:
                    cluster_data.append(combined_vector)
                    player_ids.append(p.player_id)
            
            if len(cluster_data) < self.cluster_min_size:
                return None
            
            # Perform clustering
            cluster_data = np.array(cluster_data)
            
            # Use PCA for dimensionality reduction if needed
            if cluster_data.shape[1] > 50:
                pca = PCA(n_components=50)
                cluster_data = pca.fit_transform(cluster_data)
            
            # Perform K-means clustering
            n_clusters = min(5, len(cluster_data) // 3)  # Adaptive number of clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(cluster_data)
            
            # Find player's cluster
            player_idx = player_ids.index(player.player_id) if player.player_id in player_ids else -1
            if player_idx == -1:
                return None
            
            player_cluster = cluster_labels[player_idx]
            
            # Get cluster members
            cluster_members = [player_ids[i] for i, label in enumerate(cluster_labels) if label == player_cluster]
            
            # Calculate cluster characteristics
            cluster_center = kmeans.cluster_centers_[player_cluster].tolist()
            
            # Determine cluster archetype
            cluster_archetype = self._determine_cluster_archetype(cluster_members, all_players)
            
            # Calculate cluster characteristics
            cluster_characteristics = self._calculate_cluster_characteristics(cluster_members, all_players)
            
            return ClusterResult(
                cluster_id=int(player_cluster),
                cluster_size=len(cluster_members),
                cluster_center=dict(enumerate(cluster_center)),
                cluster_players=cluster_members,
                cluster_archetype=cluster_archetype,
                cluster_characteristics=cluster_characteristics
            )
            
        except Exception as e:
            self.logger.error(f"Error identifying cluster: {e}")
            return None
    
    def _determine_cluster_archetype(
        self, 
        cluster_members: List[str], 
        all_players: List[Player]
    ) -> Optional[ArchetypeType]:
        """Determine the dominant archetype for a cluster."""
        try:
            archetype_counts = {}
            
            for player_id in cluster_members:
                player = next((p for p in all_players if p.player_id == player_id), None)
                if player:
                    # Simple archetype determination based on position and stats
                    if player.position == Position.RB and player.stats.rushing_attempts > 15:
                        archetype = ArchetypeType.VOLUME_RUSHER
                    elif player.position == Position.QB and player.stats.fantasy_points > 25:
                        archetype = ArchetypeType.ELITE_QB
                    elif player.position == Position.WR and player.stats.total_touchdowns > 1:
                        archetype = ArchetypeType.RED_ZONE_TARGET
                    else:
                        archetype = ArchetypeType.VALUE_PLAY
                    
                    archetype_counts[archetype] = archetype_counts.get(archetype, 0) + 1
            
            if archetype_counts:
                dominant_archetype = max(archetype_counts.items(), key=lambda x: x[1])[0]
                return dominant_archetype
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error determining cluster archetype: {e}")
            return None
    
    def _calculate_cluster_characteristics(
        self, 
        cluster_members: List[str], 
        all_players: List[Player]
    ) -> Dict[str, float]:
        """Calculate characteristics of a cluster."""
        try:
            cluster_players = [p for p in all_players if p.player_id in cluster_members]
            
            if not cluster_players:
                return {}
            
            characteristics = {
                "avg_fantasy_points": np.mean([p.stats.fantasy_points for p in cluster_players]),
                "avg_salary": np.mean([p.dfs.salary for p in cluster_players]),
                "avg_consistency": np.mean([getattr(p.dfs, 'consistency', 0.5) for p in cluster_players]),
                "avg_upside": np.mean([getattr(p.dfs, 'upside', 0.1) for p in cluster_players]),
                "position_diversity": len(set(p.position for p in cluster_players)) / len(cluster_players)
            }
            
            return characteristics
            
        except Exception as e:
            self.logger.error(f"Error calculating cluster characteristics: {e}")
            return {}
    
    async def _detect_anomalies(
        self, 
        player: Player, 
        embeddings: Dict[str, List[float]]
    ) -> Optional[AnomalyResult]:
        """Detect anomalies in player profile."""
        try:
            # Calculate anomaly score
            anomaly_score = await self._calculate_anomaly_score(player, embeddings)
            
            if anomaly_score < 0.3:  # Low anomaly score
                return None
            
            # Identify anomalous vectors
            anomalous_vectors = []
            normal_ranges = {}
            
            for vector_type in ["stats", "context", "value"]:
                if vector_type in embeddings:
                    # Calculate vector statistics
                    vector_magnitude = np.linalg.norm(embeddings[vector_type])
                    
                    # Get all players for comparison
                    all_players = await self._get_all_players()
                    all_magnitudes = []
                    
                    for other_player in all_players:
                        if other_player.player_id != player.player_id:
                            other_embeddings = await self._generate_player_embeddings(other_player)
                            if vector_type in other_embeddings:
                                other_magnitude = np.linalg.norm(other_embeddings[vector_type])
                                all_magnitudes.append(other_magnitude)
                    
                    if all_magnitudes:
                        mean_magnitude = np.mean(all_magnitudes)
                        std_magnitude = np.std(all_magnitudes)
                        
                        if std_magnitude > 0:
                            z_score = abs(vector_magnitude - mean_magnitude) / std_magnitude
                            if z_score > self.anomaly_threshold:
                                anomalous_vectors.append(vector_type)
                        
                        normal_ranges[vector_type] = (mean_magnitude - 2*std_magnitude, mean_magnitude + 2*std_magnitude)
            
            # Determine anomaly type
            anomaly_type = "vector_magnitude" if anomalous_vectors else "overall_profile"
            
            # Generate explanation
            explanation = self._generate_anomaly_explanation(player, anomaly_score, anomalous_vectors)
            
            # Generate recommendations
            recommendations = self._generate_anomaly_recommendations(player, anomalous_vectors)
            
            return AnomalyResult(
                anomaly_score=anomaly_score,
                anomaly_type=anomaly_type,
                anomalous_vectors=anomalous_vectors,
                normal_range=normal_ranges,
                explanation=explanation,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return None
    
    def _generate_anomaly_explanation(
        self, 
        player: Player, 
        anomaly_score: float, 
        anomalous_vectors: List[str]
    ) -> str:
        """Generate explanation for anomaly detection."""
        try:
            explanation = f"{player.name} shows {anomaly_score:.1%} anomaly score. "
            
            if anomalous_vectors:
                explanation += f"Anomalous vectors: {', '.join(anomalous_vectors)}. "
                explanation += "This player has unusual patterns in these vector types."
            else:
                explanation += "Overall profile shows unusual characteristics compared to similar players."
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error generating anomaly explanation: {e}")
            return f"Anomaly detection completed for {player.name}."
    
    def _generate_anomaly_recommendations(
        self, 
        player: Player, 
        anomalous_vectors: List[str]
    ) -> List[str]:
        """Generate recommendations for anomalous players."""
        try:
            recommendations = []
            
            if "stats" in anomalous_vectors:
                recommendations.append("Monitor statistical performance closely - unusual patterns detected")
            
            if "context" in anomalous_vectors:
                recommendations.append("Consider matchup and situational factors more carefully")
            
            if "value" in anomalous_vectors:
                recommendations.append("Review salary and ownership projections - may be mispriced")
            
            if len(anomalous_vectors) > 1:
                recommendations.append("High anomaly across multiple vectors - exercise caution")
            
            if not recommendations:
                recommendations.append("Continue monitoring for pattern changes")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating anomaly recommendations: {e}")
            return ["Monitor player performance closely"]
    
    async def _predict_performance(
        self, 
        player: Player, 
        embeddings: Dict[str, List[float]]
    ) -> PredictionResult:
        """Predict player performance using multi-vector data."""
        try:
            # Get historical performance data
            historical_data = await self._get_historical_performance(player)
            
            # Calculate prediction factors
            prediction_factors = self._calculate_prediction_factors(player, embeddings, historical_data)
            
            # Predict fantasy points
            base_prediction = player.stats.fantasy_points  # Use current as base
            adjustment = sum(prediction_factors.values())
            predicted_fantasy_points = max(0.0, base_prediction + adjustment)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(predicted_fantasy_points, prediction_factors)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(player, embeddings)
            
            # Calculate upside and downside
            upside_potential = predicted_fantasy_points * 1.3  # 30% upside
            downside_risk = predicted_fantasy_points * 0.7   # 30% downside
            
            return PredictionResult(
                predicted_fantasy_points=predicted_fantasy_points,
                confidence_interval=confidence_interval,
                prediction_factors=prediction_factors,
                risk_factors=risk_factors,
                upside_potential=upside_potential,
                downside_risk=downside_risk
            )
            
        except Exception as e:
            self.logger.error(f"Error predicting performance: {e}")
            return PredictionResult(
                predicted_fantasy_points=player.stats.fantasy_points,
                confidence_interval=(player.stats.fantasy_points * 0.8, player.stats.fantasy_points * 1.2),
                prediction_factors={},
                risk_factors=["Prediction model error"],
                upside_potential=player.stats.fantasy_points * 1.2,
                downside_risk=player.stats.fantasy_points * 0.8
            )
    
    async def _get_historical_performance(self, player: Player) -> Dict[str, Any]:
        """Get historical performance data for player."""
        try:
            # This would typically query a database for historical data
            # For now, return mock data
            return {
                "avg_fantasy_points": player.stats.fantasy_points,
                "consistency": getattr(player.dfs, 'consistency', 0.7),
                "trend": "stable",
                "volatility": 0.2
            }
        except Exception as e:
            self.logger.error(f"Error getting historical performance: {e}")
            return {}
    
    def _calculate_prediction_factors(
        self, 
        player: Player, 
        embeddings: Dict[str, List[float]], 
        historical_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate factors that influence performance prediction."""
        try:
            factors = {}
            
            # Vector-based factors
            for vector_type in ["stats", "context", "value"]:
                if vector_type in embeddings:
                    vector_magnitude = np.linalg.norm(embeddings[vector_type])
                    factors[f"{vector_type}_strength"] = vector_magnitude * 0.1
            
            # Historical factors
            factors["consistency_bonus"] = historical_data.get("consistency", 0.7) * 0.5
            factors["trend_adjustment"] = 0.2 if historical_data.get("trend") == "improving" else -0.1
            
            # Position-specific factors
            if player.position == Position.QB:
                factors["qb_volume"] = player.stats.passing_yards / 300.0 * 0.3
            elif player.position == Position.RB:
                factors["rb_volume"] = player.stats.rushing_attempts / 20.0 * 0.3
            elif player.position == Position.WR:
                factors["wr_targets"] = getattr(player.stats, 'targets', 0) / 10.0 * 0.3
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction factors: {e}")
            return {}
    
    def _calculate_confidence_interval(
        self, 
        prediction: float, 
        factors: Dict[str, float]
    ) -> Tuple[float, float]:
        """Calculate confidence interval for prediction."""
        try:
            # Simple confidence interval based on factor complexity
            factor_count = len(factors)
            confidence_width = 0.1 + (factor_count * 0.02)  # Wider interval for more factors
            
            lower_bound = prediction * (1 - confidence_width)
            upper_bound = prediction * (1 + confidence_width)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence interval: {e}")
            return (prediction * 0.8, prediction * 1.2)
    
    def _identify_risk_factors(
        self, 
        player: Player, 
        embeddings: Dict[str, List[float]]
    ) -> List[str]:
        """Identify risk factors for player performance."""
        try:
            risk_factors = []
            
            # Check for low consistency
            if hasattr(player.dfs, 'consistency') and player.dfs.consistency < 0.6:
                risk_factors.append("Low consistency")
            
            # Check for high volatility
            if hasattr(player.dfs, 'volatility') and player.dfs.volatility > 0.3:
                risk_factors.append("High volatility")
            
            # Check for injury risk
            if hasattr(player, 'injury_status') and player.injury_status == "questionable":
                risk_factors.append("Injury concern")
            
            # Check for vector anomalies
            for vector_type in ["stats", "context", "value"]:
                if vector_type in embeddings:
                    vector_magnitude = np.linalg.norm(embeddings[vector_type])
                    if vector_magnitude < 0.3:  # Low vector strength
                        risk_factors.append(f"Weak {vector_type} vector")
            
            if not risk_factors:
                risk_factors.append("Standard risk factors")
            
            return risk_factors
            
        except Exception as e:
            self.logger.error(f"Error identifying risk factors: {e}")
            return ["Risk assessment error"]
    
    async def _analyze_matchups(
        self, 
        player: Player, 
        embeddings: Dict[str, List[float]]
    ) -> List[MatchupAnalysis]:
        """Analyze matchup advantages for player."""
        try:
            matchups = []
            
            # Get all teams for matchup analysis
            all_teams = list(Team)
            
            for opponent_team in all_teams:
                if opponent_team == player.team:
                    continue
                
                # Calculate matchup advantage
                advantage_score = await self._calculate_matchup_advantage(player, opponent_team, embeddings)
                
                # Identify favorable and unfavorable vectors
                favorable_vectors = []
                unfavorable_vectors = []
                
                for vector_type in ["stats", "context", "value"]:
                    if vector_type in embeddings:
                        vector_advantage = self._calculate_vector_matchup_advantage(
                            player, opponent_team, vector_type
                        )
                        if vector_advantage > 0.1:
                            favorable_vectors.append(vector_type)
                        elif vector_advantage < -0.1:
                            unfavorable_vectors.append(vector_type)
                
                # Generate matchup strategy
                matchup_strategy = self._generate_matchup_strategy(
                    player, opponent_team, favorable_vectors, unfavorable_vectors
                )
                
                # Calculate confidence
                confidence = 0.7 + (len(favorable_vectors) - len(unfavorable_vectors)) * 0.1
                confidence = max(0.3, min(0.9, confidence))
                
                matchup = MatchupAnalysis(
                    opponent_team=opponent_team.value,
                    advantage_score=advantage_score,
                    favorable_vectors=favorable_vectors,
                    unfavorable_vectors=unfavorable_vectors,
                    matchup_strategy=matchup_strategy,
                    confidence=confidence
                )
                matchups.append(matchup)
            
            # Sort by advantage score and return top 5
            matchups.sort(key=lambda x: x.advantage_score, reverse=True)
            return matchups[:5]
            
        except Exception as e:
            self.logger.error(f"Error analyzing matchups: {e}")
            return []
    
    async def _calculate_matchup_advantage(
        self, 
        player: Player, 
        opponent_team: Team, 
        embeddings: Dict[str, List[float]]
    ) -> float:
        """Calculate matchup advantage score."""
        try:
            # Simple matchup advantage calculation
            # In practice, this would use defensive rankings, historical performance, etc.
            
            base_advantage = 0.0
            
            # Position-specific advantages
            if player.position == Position.QB:
                base_advantage += 0.1  # QBs generally have advantage
            elif player.position == Position.RB:
                base_advantage += 0.05  # RBs moderate advantage
            elif player.position == Position.WR:
                base_advantage += 0.08  # WRs slight advantage
            
            # Vector-based adjustments
            for vector_type in ["stats", "context", "value"]:
                if vector_type in embeddings:
                    vector_strength = np.linalg.norm(embeddings[vector_type])
                    base_advantage += vector_strength * 0.05
            
            # Add some randomness for demonstration
            import random
            base_advantage += random.uniform(-0.1, 0.1)
            
            return max(-1.0, min(1.0, base_advantage))  # Clamp between -1 and 1
            
        except Exception as e:
            self.logger.error(f"Error calculating matchup advantage: {e}")
            return 0.0
    
    def _calculate_vector_matchup_advantage(
        self, 
        player: Player, 
        opponent_team: Team, 
        vector_type: str
    ) -> float:
        """Calculate matchup advantage for specific vector type."""
        try:
            # Simplified vector-specific matchup calculation
            if vector_type == "stats":
                return player.stats.fantasy_points / 30.0 - 0.5  # Normalize around 0
            elif vector_type == "context":
                return 0.1  # Slight positive for context
            elif vector_type == "value":
                return getattr(player.dfs, 'salary_efficiency', 0.1) - 0.1
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating vector matchup advantage: {e}")
            return 0.0
    
    def _generate_matchup_strategy(
        self, 
        player: Player, 
        opponent_team: Team, 
        favorable_vectors: List[str], 
        unfavorable_vectors: List[str]
    ) -> str:
        """Generate matchup strategy based on vector analysis."""
        try:
            if favorable_vectors and not unfavorable_vectors:
                return f"Strong matchup advantage. Leverage {', '.join(favorable_vectors)} vectors."
            elif favorable_vectors and unfavorable_vectors:
                return f"Mixed matchup. Focus on {', '.join(favorable_vectors)} strengths."
            elif not favorable_vectors and unfavorable_vectors:
                return f"Challenging matchup. Monitor {', '.join(unfavorable_vectors)} factors closely."
            else:
                return "Neutral matchup. Standard approach recommended."
                
        except Exception as e:
            self.logger.error(f"Error generating matchup strategy: {e}")
            return "Matchup analysis completed."
    
    async def _identify_value_opportunities(
        self, 
        player: Player, 
        embeddings: Dict[str, List[float]]
    ) -> List[ValueOpportunity]:
        """Identify value opportunities for player."""
        try:
            opportunities = []
            
            # Check for salary efficiency opportunities
            if hasattr(player.dfs, 'salary_efficiency') and player.dfs.salary_efficiency > 0.15:
                opportunities.append(ValueOpportunity(
                    opportunity_type="salary_efficiency",
                    opportunity_score=player.dfs.salary_efficiency,
                    undervalued_vectors=["value"],
                    market_mispricing={"salary": player.dfs.salary, "projected_points": player.dfs.projected_points},
                    expected_return=player.dfs.salary_efficiency * 100,
                    risk_level="low"
                ))
            
            # Check for ownership projection opportunities
            if hasattr(player.dfs, 'ownership_projection') and player.dfs.ownership_projection < 0.1:
                opportunities.append(ValueOpportunity(
                    opportunity_type="low_ownership",
                    opportunity_score=1.0 - player.dfs.ownership_projection,
                    undervalued_vectors=["value", "context"],
                    market_mispricing={"ownership": player.dfs.ownership_projection},
                    expected_return=20.0,  # 20% return potential
                    risk_level="medium"
                ))
            
            # Check for upside opportunities
            if hasattr(player.dfs, 'upside') and player.dfs.upside > 0.2:
                opportunities.append(ValueOpportunity(
                    opportunity_type="high_upside",
                    opportunity_score=player.dfs.upside,
                    undervalued_vectors=["value", "stats"],
                    market_mispricing={"upside": player.dfs.upside},
                    expected_return=player.dfs.upside * 150,
                    risk_level="high"
                ))
            
            # Sort by opportunity score
            opportunities.sort(key=lambda x: x.opportunity_score, reverse=True)
            return opportunities[:3]  # Top 3 opportunities
            
        except Exception as e:
            self.logger.error(f"Error identifying value opportunities: {e}")
            return []
    
    async def _prepare_visualization_data(
        self,
        player: Player,
        embeddings: Dict[str, List[float]],
        archetype: ArchetypeResult,
        vector_contributions: List[VectorContribution]
    ) -> Dict[str, Any]:
        """Prepare data for frontend visualizations."""
        try:
            visualization_data = {
                "radar_chart": {
                    "labels": ["Stats", "Context", "Value", "Consistency", "Upside"],
                    "data": [
                        vector_contributions[0].contribution_score if len(vector_contributions) > 0 else 0.0,
                        vector_contributions[1].contribution_score if len(vector_contributions) > 1 else 0.0,
                        vector_contributions[2].contribution_score if len(vector_contributions) > 2 else 0.0,
                        getattr(player.dfs, 'consistency', 0.5),
                        getattr(player.dfs, 'upside', 0.1)
                    ]
                },
                "vector_contributions": {
                    "labels": [vc.vector_type for vc in vector_contributions],
                    "data": [vc.contribution_percentage for vc in vector_contributions]
                },
                "archetype_distribution": {
                    "archetype": archetype.archetype.value,
                    "confidence": archetype.confidence_score,
                    "primary_vectors": archetype.primary_vectors,
                    "secondary_vectors": archetype.secondary_vectors
                },
                "performance_trend": {
                    "labels": ["Week 1", "Week 2", "Week 3", "Week 4", "Week 5"],
                    "data": [player.stats.fantasy_points] * 5  # Mock trend data
                },
                "similarity_network": {
                    "nodes": [{"id": player.player_id, "name": player.name, "group": 1}],
                    "links": []
                }
            }
            
            return visualization_data
            
        except Exception as e:
            self.logger.error(f"Error preparing visualization data: {e}")
            return {}
    
    async def _get_all_players(self) -> List[Player]:
        """Get all players from vector engine."""
        try:
            results = await self.vector_engine.search_vectors(
                query_vector=[0.0] * 768,
                collection_type=CollectionType.REGULAR,
                limit=1000
            )
            
            return [result.player for result in results]
            
        except Exception as e:
            self.logger.error(f"Error getting all players: {e}")
            return []
