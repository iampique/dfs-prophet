"""
Best-in-class embedding generator for DFS Prophet using BGE-base-en-v1.5.

Features:
- Async embedding generation
- Batch processing capabilities  
- Feature normalization and scaling
- Vector dimension optimization for binary quantization
- Caching mechanisms for performance
- Support for different embedding strategies
"""

import asyncio
import hashlib
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
from datetime import datetime
from enum import Enum

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ..config import get_settings
from ..utils import get_logger, performance_timer
from ..data.models import Player, PlayerStats


class EmbeddingStrategy(str, Enum):
    """Embedding generation strategies."""
    STATISTICAL = "statistical"
    CONTEXTUAL = "contextual"
    HYBRID = "hybrid"
    TEXT_ONLY = "text_only"


class FeatureNormalizer:
    """Feature normalization and scaling for statistical data."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.standard_scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self._fitted = False
    
    def normalize_statistical_features(self, stats: PlayerStats) -> Dict[str, float]:
        """Normalize statistical features for embedding generation."""
        features = {}
        
        # Passing features (QB)
        if stats.passing_yards is not None:
            features["passing_yards_norm"] = self._normalize_value(stats.passing_yards, 0, 600)
        if stats.passing_touchdowns is not None:
            features["passing_tds_norm"] = self._normalize_value(stats.passing_touchdowns, 0, 8)
        if stats.completion_percentage is not None:
            features["completion_pct_norm"] = self._normalize_value(stats.completion_percentage, 0, 100)
        if stats.qb_rating is not None:
            features["qb_rating_norm"] = self._normalize_value(stats.qb_rating, 0, 158.3)
        
        # Rushing features (RB, QB, WR)
        if stats.rushing_yards is not None:
            features["rushing_yards_norm"] = self._normalize_value(stats.rushing_yards, 0, 200)
        if stats.rushing_touchdowns is not None:
            features["rushing_tds_norm"] = self._normalize_value(stats.rushing_touchdowns, 0, 4)
        if stats.yards_per_carry is not None:
            features["ypc_norm"] = self._normalize_value(stats.yards_per_carry, 0, 10)
        
        # Receiving features (WR, TE, RB)
        if stats.receiving_yards is not None:
            features["receiving_yards_norm"] = self._normalize_value(stats.receiving_yards, 0, 300)
        if stats.receiving_touchdowns is not None:
            features["receiving_tds_norm"] = self._normalize_value(stats.receiving_touchdowns, 0, 4)
        if stats.receptions is not None:
            features["receptions_norm"] = self._normalize_value(stats.receptions, 0, 15)
        if stats.targets is not None:
            features["targets_norm"] = self._normalize_value(stats.targets, 0, 20)
        if stats.catch_percentage is not None:
            features["catch_pct_norm"] = self._normalize_value(stats.catch_percentage, 0, 100)
        
        # Fantasy points
        if stats.fantasy_points is not None:
            features["fantasy_points_norm"] = self._normalize_value(stats.fantasy_points, 0, 50)
        if stats.fantasy_points_ppr is not None:
            features["fantasy_points_ppr_norm"] = self._normalize_value(stats.fantasy_points_ppr, 0, 60)
        
        # Advanced metrics
        if stats.snap_percentage is not None:
            features["snap_pct_norm"] = self._normalize_value(stats.snap_percentage, 0, 100)
        if stats.red_zone_targets is not None:
            features["redzone_targets_norm"] = self._normalize_value(stats.red_zone_targets, 0, 5)
        
        return features
    
    def _normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize a value to 0-1 range."""
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)
    
    def create_text_features(self, player: Player) -> str:
        """Create text features for contextual embedding."""
        base = player.base
        stats = player.stats
        
        # Core player info
        text_parts = [
            f"{base.name}",
            f"{base.position.value}",
            f"{base.team.value}",
            f"season {base.season}"
        ]
        
        if base.week:
            text_parts.append(f"week {base.week}")
        
        # Statistical highlights
        if stats:
            if stats.passing_yards and stats.passing_yards > 200:
                text_parts.append(f"{stats.passing_yards:.0f} passing yards")
            if stats.passing_touchdowns and stats.passing_touchdowns > 1:
                text_parts.append(f"{stats.passing_touchdowns:.1f} passing touchdowns")
            
            if stats.rushing_yards and stats.rushing_yards > 50:
                text_parts.append(f"{stats.rushing_yards:.0f} rushing yards")
            if stats.rushing_touchdowns and stats.rushing_touchdowns > 0:
                text_parts.append(f"{stats.rushing_touchdowns:.1f} rushing touchdowns")
            
            if stats.receiving_yards and stats.receiving_yards > 50:
                text_parts.append(f"{stats.receiving_yards:.0f} receiving yards")
            if stats.receiving_touchdowns and stats.receiving_touchdowns > 0:
                text_parts.append(f"{stats.receiving_touchdowns:.1f} receiving touchdowns")
            
            if stats.fantasy_points:
                text_parts.append(f"{stats.fantasy_points:.1f} fantasy points")
            
            if stats.snap_percentage and stats.snap_percentage > 70:
                text_parts.append(f"{stats.snap_percentage:.0f}% snap share")
        
        return " ".join(text_parts)


class EmbeddingCache:
    """Caching mechanism for embeddings with LRU eviction."""
    
    def __init__(self, max_size: int = 10000):
        self.logger = get_logger(__name__)
        self.max_size = max_size
        self._cache: Dict[str, Tuple[List[float], datetime]] = {}
        self._access_count: Dict[str, int] = {}
    
    def _generate_cache_key(self, player_id: str, strategy: EmbeddingStrategy, **kwargs) -> str:
        """Generate cache key for embedding."""
        key_data = {
            "player_id": player_id,
            "strategy": strategy,
            **kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, player_id: str, strategy: EmbeddingStrategy, **kwargs) -> Optional[List[float]]:
        """Get embedding from cache."""
        cache_key = self._generate_cache_key(player_id, strategy, **kwargs)
        
        if cache_key in self._cache:
            embedding, timestamp = self._cache[cache_key]
            self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
            
            # Check if cache entry is still valid (24 hours)
            if (datetime.now() - timestamp).total_seconds() < 86400:
                self.logger.debug(f"Cache hit for player {player_id} with strategy {strategy}")
                return embedding
            else:
                # Remove expired entry
                del self._cache[cache_key]
                del self._access_count[cache_key]
        
        return None
    
    def set(self, player_id: str, strategy: EmbeddingStrategy, embedding: List[float], **kwargs) -> None:
        """Store embedding in cache."""
        cache_key = self._generate_cache_key(player_id, strategy, **kwargs)
        
        # Implement LRU eviction if cache is full
        if len(self._cache) >= self.max_size:
            self._evict_least_used()
        
        self._cache[cache_key] = (embedding, datetime.now())
        self._access_count[cache_key] = 1
        self.logger.debug(f"Cached embedding for player {player_id} with strategy {strategy}")
    
    def _evict_least_used(self) -> None:
        """Evict least recently used cache entries."""
        if not self._access_count:
            return
        
        # Remove 10% of least used entries
        num_to_evict = max(1, len(self._cache) // 10)
        sorted_keys = sorted(self._access_count.items(), key=lambda x: x[1])
        
        for key, _ in sorted_keys[:num_to_evict]:
            del self._cache[key]
            del self._access_count[key]
        
        self.logger.debug(f"Evicted {num_to_evict} cache entries")


class EmbeddingGenerator:
    """Best-in-class embedding generator for DFS Prophet."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.normalizer = FeatureNormalizer()
        self.cache = EmbeddingCache(max_size=10000)
        
        # Load model
        self._model: Optional[SentenceTransformer] = None
        
        # Performance tracking
        self.total_embeddings_generated = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.batch_processing_time = 0.0
        self.avg_embedding_time = 0.0
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the BGE model."""
        if self._model is None:
            self.logger.info(f"Loading embedding model: {self.settings.embedding.model_name}")
            self._model = SentenceTransformer(self.settings.embedding.model_name)
        return self._model
    
    @performance_timer('generate_player_embedding')
    async def generate_player_embedding(
        self,
        player: Player,
        strategy: EmbeddingStrategy = EmbeddingStrategy.HYBRID,
        use_cache: bool = True
    ) -> List[float]:
        """Generate embedding for a player using specified strategy."""
        start_time = time.time()
        
        # Check cache first
        if use_cache:
            cached_embedding = self.cache.get(player.player_id, strategy)
            if cached_embedding:
                self.cache_hits += 1
                return cached_embedding
        
        self.cache_misses += 1
        
        # Generate embedding based on strategy
        if strategy == EmbeddingStrategy.STATISTICAL:
            embedding = await self._generate_statistical_embedding(player)
        elif strategy == EmbeddingStrategy.CONTEXTUAL:
            embedding = await self._generate_contextual_embedding(player)
        elif strategy == EmbeddingStrategy.HYBRID:
            embedding = await self._generate_hybrid_embedding(player)
        elif strategy == EmbeddingStrategy.TEXT_ONLY:
            embedding = await self._generate_text_embedding(player)
        else:
            raise ValueError(f"Unknown embedding strategy: {strategy}")
        
        # Cache the result
        if use_cache:
            self.cache.set(player.player_id, strategy, embedding)
        
        # Update performance metrics
        self.total_embeddings_generated += 1
        embedding_time = time.time() - start_time
        self.avg_embedding_time = (
            (self.avg_embedding_time * (self.total_embeddings_generated - 1) + embedding_time) 
            / self.total_embeddings_generated
        )
        
        return embedding
    
    async def _generate_statistical_embedding(self, player: Player) -> List[float]:
        """Generate embedding from statistical features."""
        if not player.stats:
            raise ValueError("Player stats required for statistical embedding")
        
        # Normalize statistical features
        features = self.normalizer.normalize_statistical_features(player.stats)
        
        # Convert to feature vector
        feature_vector = list(features.values())
        
        # Pad or truncate to target dimensions for binary quantization
        target_dim = self.settings.vector_db.vector_dimensions
        if len(feature_vector) < target_dim:
            # Pad with zeros
            feature_vector.extend([0.0] * (target_dim - len(feature_vector)))
        elif len(feature_vector) > target_dim:
            # Truncate
            feature_vector = feature_vector[:target_dim]
        
        # Normalize the final vector
        feature_vector = np.array(feature_vector)
        feature_vector = feature_vector / (np.linalg.norm(feature_vector) + 1e-8)
        
        return feature_vector.tolist()
    
    async def _generate_contextual_embedding(self, player: Player) -> List[float]:
        """Generate embedding from contextual text features."""
        text_features = self.normalizer.create_text_features(player)
        
        # Generate embedding using BGE model
        embedding = self.model.encode(text_features, normalize_embeddings=True)
        return embedding.tolist()
    
    async def _generate_hybrid_embedding(self, player: Player) -> List[float]:
        """Generate hybrid embedding combining statistical and contextual features."""
        # Generate both types of embeddings
        statistical_embedding = await self._generate_statistical_embedding(player)
        contextual_embedding = await self._generate_contextual_embedding(player)
        
        # Combine embeddings (weighted average)
        stats_weight = 0.3
        context_weight = 0.7
        
        combined = np.array(statistical_embedding) * stats_weight + np.array(contextual_embedding) * context_weight
        
        # Normalize
        combined = combined / (np.linalg.norm(combined) + 1e-8)
        
        return combined.tolist()
    
    async def _generate_text_embedding(self, player: Player) -> List[float]:
        """Generate embedding from comprehensive text representation."""
        # Create a more structured representation that emphasizes key identifiers
        text_parts = [
            f"{player.name}",
            f"{player.position.value}",
            f"{player.team.value}",
            f"season {player.base.season}"
        ]
        
        if player.base.week:
            text_parts.append(f"week {player.base.week}")
        
        # Add performance highlights with more emphasis
        if player.stats:
            if player.stats.fantasy_points and player.stats.fantasy_points > 10:
                text_parts.extend([f"{player.stats.fantasy_points:.0f} fantasy points", "high performer"])
            if player.stats.passing_yards and player.stats.passing_yards > 200:
                text_parts.extend([f"{player.stats.passing_yards:.0f} passing yards", "quarterback", "passer"])
            if player.stats.rushing_yards and player.stats.rushing_yards > 50:
                text_parts.extend([f"{player.stats.rushing_yards:.0f} rushing yards", "running back", "rusher"])
            if player.stats.receiving_yards and player.stats.receiving_yards > 50:
                text_parts.extend([f"{player.stats.receiving_yards:.0f} receiving yards", "receiver", "catcher"])
        
        # Add DFS context
        if player.dfs:
            if player.dfs.salary > 5000:
                text_parts.extend([f"high salary ${player.dfs.salary}", "expensive", "premium"])
            if player.dfs.projected_points > 15:
                text_parts.extend([f"projected {player.dfs.projected_points:.0f} points", "high projection"])
        
        # Repeat key identifiers for emphasis
        text_parts.extend([player.name, player.position.value, player.team.value])
        
        text_representation = " ".join(text_parts)
        
        # Generate embedding
        embedding = self.model.encode(text_representation, normalize_embeddings=True)
        return embedding.tolist()
    
    @performance_timer('batch_embedding_generation')
    async def generate_batch_embeddings(
        self,
        players: List[Player],
        strategy: EmbeddingStrategy = EmbeddingStrategy.HYBRID,
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """Generate embeddings for multiple players in batches."""
        if not players:
            return []
        
        start_time = time.time()
        batch_size = batch_size or self.settings.embedding.batch_size
        embeddings = []
        
        # Process in batches
        for i in range(0, len(players), batch_size):
            batch = players[i:i + batch_size]
            
            # Generate embeddings for batch
            batch_embeddings = await asyncio.gather(
                *[self.generate_player_embedding(player, strategy) for player in batch]
            )
            embeddings.extend(batch_embeddings)
        
        # Update batch processing metrics
        self.batch_processing_time = time.time() - start_time
        
        return embeddings
    
    @performance_timer('generate_query_embedding')
    async def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query."""
        # Enhance the query with context for better matching
        enhanced_query = query.lower().strip()
        
        # Add position context for better matching
        if "quarterback" in enhanced_query or "qb" in enhanced_query:
            enhanced_query = f"quarterback {enhanced_query} passer"
        elif "running back" in enhanced_query or "rb" in enhanced_query:
            enhanced_query = f"running back {enhanced_query} rusher"
        elif "wide receiver" in enhanced_query or "wr" in enhanced_query:
            enhanced_query = f"wide receiver {enhanced_query} receiver catcher"
        elif "tight end" in enhanced_query or "te" in enhanced_query:
            enhanced_query = f"tight end {enhanced_query} receiver"
        
        # Add team context if mentioned
        if "chiefs" in enhanced_query or "kc" in enhanced_query:
            enhanced_query = f"Kansas City Chiefs {enhanced_query}"
        elif "patriots" in enhanced_query or "ne" in enhanced_query:
            enhanced_query = f"New England Patriots {enhanced_query}"
        elif "mahomes" in enhanced_query:
            enhanced_query = f"Patrick Mahomes {enhanced_query} quarterback Kansas City Chiefs"
        # Add more teams as needed
        
        # Repeat the query for emphasis
        enhanced_query = f"{enhanced_query} {enhanced_query}"
        
        embedding = self.model.encode(enhanced_query, normalize_embeddings=True)
        return embedding.tolist()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        cache_hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0
            else 0
        )
        
        return {
            "total_embeddings_generated": self.total_embeddings_generated,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "cache_size": len(self.cache._cache),
            "avg_embedding_time_ms": self.avg_embedding_time * 1000,
            "batch_processing_time_ms": self.batch_processing_time * 1000,
            "model_name": self.settings.embedding.model_name,
            "vector_dimensions": self.settings.vector_db.vector_dimensions,
            "binary_quantization_enabled": self.settings.binary_quantization.enabled
        }
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache._cache.clear()
        self.cache._access_count.clear()
        self.logger.info("Embedding cache cleared")


# Global embedding generator instance
_embedding_generator: Optional[EmbeddingGenerator] = None


def get_embedding_generator() -> EmbeddingGenerator:
    """Get the global embedding generator instance."""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator



