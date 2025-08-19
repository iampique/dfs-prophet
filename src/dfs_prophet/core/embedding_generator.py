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


class MultiVectorEmbeddingStrategy(str, Enum):
    """Multi-vector embedding generation strategies."""
    STATISTICAL = "statistical"
    CONTEXTUAL = "contextual"
    VALUE = "value"
    COMBINED = "combined"
    ALL = "all"


class StatisticalEmbedder:
    """Specialized embedder for statistical performance features."""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", shared_model: Optional[SentenceTransformer] = None):
        self.model_name = model_name
        self.logger = get_logger(__name__)
        self._model = shared_model  # Use shared model if provided
        self.vector_dimensions = 768
        
    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            self.logger.info(f"Loading statistical embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def preprocess_statistical_features(self, player_data: Dict[str, Any]) -> str:
        """Preprocess statistical features into text representation."""
        text_parts = []
        
        # Basic player info
        text_parts.append(f"{player_data.get('player_name', 'Unknown')} {player_data.get('position', 'Unknown')}")
        
        # Statistical features
        if player_data.get('fantasy_points'):
            text_parts.append(f"{player_data['fantasy_points']:.1f} fantasy points")
        
        if player_data.get('passing_yards'):
            text_parts.append(f"{player_data['passing_yards']:.0f} passing yards")
        
        if player_data.get('rushing_yards'):
            text_parts.append(f"{player_data['rushing_yards']:.0f} rushing yards")
        
        if player_data.get('receiving_yards'):
            text_parts.append(f"{player_data['receiving_yards']:.0f} receiving yards")
        
        if player_data.get('total_touchdowns'):
            text_parts.append(f"{player_data['total_touchdowns']:.1f} total touchdowns")
        
        # Efficiency metrics
        if player_data.get('yards_per_attempt'):
            text_parts.append(f"{player_data['yards_per_attempt']:.1f} yards per attempt")
        
        if player_data.get('yards_per_carry'):
            text_parts.append(f"{player_data['yards_per_carry']:.1f} yards per carry")
        
        if player_data.get('touchdown_rate'):
            text_parts.append(f"{player_data['touchdown_rate']:.3f} touchdown rate")
        
        # Position-specific efficiency
        if player_data.get('qb_efficiency'):
            text_parts.append(f"QB efficiency {player_data['qb_efficiency']:.2f}")
        
        if player_data.get('rb_efficiency'):
            text_parts.append(f"RB efficiency {player_data['rb_efficiency']:.2f}")
        
        if player_data.get('wr_te_efficiency'):
            text_parts.append(f"WR/TE efficiency {player_data['wr_te_efficiency']:.2f}")
        
        # Advanced metrics
        if player_data.get('red_zone_efficiency'):
            text_parts.append(f"red zone efficiency {player_data['red_zone_efficiency']:.2f}")
        
        if player_data.get('third_down_conversion_rate'):
            text_parts.append(f"third down conversion {player_data['third_down_conversion_rate']:.2f}")
        
        # Temporal features
        if player_data.get('fantasy_points_last_4_avg'):
            text_parts.append(f"4-week avg {player_data['fantasy_points_last_4_avg']:.1f} fantasy points")
        
        if player_data.get('consistency_score'):
            text_parts.append(f"consistency score {player_data['consistency_score']:.2f}")
        
        return " ".join(text_parts)
    
    async def generate_embedding(self, player_data: Dict[str, Any]) -> Tuple[List[float], Dict[str, Any]]:
        """Generate statistical embedding for a player."""
        try:
            # Preprocess features
            text_representation = self.preprocess_statistical_features(player_data)
            
            # Generate embedding
            embedding = self.model.encode(text_representation, normalize_embeddings=True)
            
            # Quality metrics
            quality_metrics = {
                'text_length': len(text_representation),
                'feature_count': len([k for k, v in player_data.items() if v is not None and k != 'player_name']),
                'embedding_norm': np.linalg.norm(embedding),
                'embedding_std': np.std(embedding)
            }
            
            return embedding.tolist(), quality_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to generate statistical embedding: {e}")
            raise


class ContextualEmbedder:
    """Specialized embedder for game context features."""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", shared_model: Optional[SentenceTransformer] = None):
        self.model_name = model_name
        self.logger = get_logger(__name__)
        self._model = shared_model  # Use shared model if provided
        self.vector_dimensions = 768
        
    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            self.logger.info(f"Loading contextual embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def preprocess_contextual_features(self, player_data: Dict[str, Any]) -> str:
        """Preprocess contextual features into text representation with enhanced specificity."""
        text_parts = []
        
        # Basic player info with position-specific context
        player_name = player_data.get('player_name', 'Unknown')
        position = player_data.get('position', 'Unknown')
        text_parts.append(f"{player_name} {position}")
        
        # Enhanced home/away context with specific advantages
        if player_data.get('home_away'):
            home_away = player_data['home_away']
            text_parts.append(f"{home_away} game")
            
            # Add specific home field advantages
            if home_away == 'home':
                if position == 'QB':
                    text_parts.append("home quarterback crowd noise advantage")
                elif position == 'RB':
                    text_parts.append("home running back familiar field conditions")
                elif position == 'WR':
                    text_parts.append("home wide receiver crowd support")
            else:
                if position == 'QB':
                    text_parts.append("away quarterback crowd noise challenge")
                elif position == 'RB':
                    text_parts.append("away running back unfamiliar field")
                elif position == 'WR':
                    text_parts.append("away wide receiver hostile environment")
        
        # Enhanced venue context with specific stadium characteristics
        if player_data.get('venue'):
            venue = player_data['venue']
            text_parts.append(f"venue {venue}")
            
            # Add venue-specific characteristics
            if 'Arrowhead' in venue:
                text_parts.append("loudest stadium in NFL")
                if position == 'QB':
                    text_parts.append("quarterback communication challenges")
            elif 'Lambeau' in venue:
                text_parts.append("frozen tundra conditions")
                if position == 'RB':
                    text_parts.append("running back cold weather advantage")
            elif 'MetLife' in venue:
                text_parts.append("wind tunnel stadium")
                if position == 'WR':
                    text_parts.append("wide receiver wind challenges")
            elif 'Dome' in venue or 'Stadium' in venue:
                text_parts.append("controlled environment")
                if position == 'WR':
                    text_parts.append("wide receiver ideal conditions")
        
        # Enhanced weather context with position-specific impacts
        if player_data.get('weather_conditions'):
            weather = player_data['weather_conditions']
            text_parts.append(f"weather {weather}")
            
            # Position-specific weather impacts
            if weather in ['rain', 'snow']:
                if position == 'QB':
                    text_parts.append("quarterback wet ball handling")
                elif position == 'RB':
                    text_parts.append("running back slippery conditions")
                elif position == 'WR':
                    text_parts.append("wide receiver wet field challenges")
            elif weather == 'windy':
                if position == 'QB':
                    text_parts.append("quarterback wind affecting throws")
                elif position == 'WR':
                    text_parts.append("wide receiver wind affecting routes")
            elif weather == 'clear':
                if position == 'WR':
                    text_parts.append("wide receiver ideal visibility")
        
        # Enhanced temperature context
        if player_data.get('weather_temp'):
            temp = player_data['weather_temp']
            text_parts.append(f"temperature {temp:.0f} degrees")
            
            if temp < 40:
                text_parts.append("cold weather game")
                if position == 'RB':
                    text_parts.append("running back cold weather advantage")
            elif temp > 80:
                text_parts.append("hot weather game")
                if position == 'QB':
                    text_parts.append("quarterback heat fatigue factor")
        
        # Enhanced wind context
        if player_data.get('weather_wind'):
            wind = player_data['weather_wind']
            text_parts.append(f"wind {wind:.0f} mph")
            
            if wind > 15:
                text_parts.append("high wind conditions")
                if position == 'QB':
                    text_parts.append("quarterback wind affecting accuracy")
                elif position == 'WR':
                    text_parts.append("wide receiver wind affecting catches")
        
        # Enhanced game flow context
        if player_data.get('game_total'):
            total = player_data['game_total']
            text_parts.append(f"game total {total:.1f}")
            
            if total > 50:
                text_parts.append("high scoring game expected")
                if position == 'QB':
                    text_parts.append("quarterback passing opportunities")
                elif position == 'WR':
                    text_parts.append("wide receiver touchdown opportunities")
            elif total < 40:
                text_parts.append("low scoring defensive game")
                if position == 'RB':
                    text_parts.append("running back conservative game plan")
        
        # Enhanced spread context
        if player_data.get('team_spread'):
            spread = player_data['team_spread']
            text_parts.append(f"spread {spread:+.1f}")
            
            if spread > 3:
                text_parts.append("heavy favorite game script")
                if position == 'RB':
                    text_parts.append("running back clock management role")
            elif spread < -3:
                text_parts.append("underdog comeback game script")
                if position == 'QB':
                    text_parts.append("quarterback passing comeback mode")
                elif position == 'WR':
                    text_parts.append("wide receiver garbage time opportunities")
        
        # Enhanced opponent context
        if player_data.get('opponent_team'):
            opponent = player_data['opponent_team']
            text_parts.append(f"vs {opponent}")
        
        # Enhanced defensive difficulty with position-specific context
        if player_data.get('defensive_difficulty'):
            difficulty = player_data['defensive_difficulty']
            if difficulty < 0.3:
                text_parts.append("easy defensive matchup")
                if position == 'QB':
                    text_parts.append("quarterback favorable passing conditions")
                elif position == 'RB':
                    text_parts.append("running back favorable rushing conditions")
                elif position == 'WR':
                    text_parts.append("wide receiver favorable coverage")
            elif difficulty > 0.7:
                text_parts.append("difficult defensive matchup")
                if position == 'QB':
                    text_parts.append("quarterback challenging pass defense")
                elif position == 'RB':
                    text_parts.append("running back challenging run defense")
                elif position == 'WR':
                    text_parts.append("wide receiver challenging coverage")
            else:
                text_parts.append("medium defensive matchup")
        
        # Enhanced defensive rank context
        if player_data.get('defensive_rank'):
            rank = player_data['defensive_rank']
            text_parts.append(f"opponent defensive rank {rank}")
            
            if rank <= 10:
                text_parts.append("elite defensive opponent")
            elif rank >= 25:
                text_parts.append("weak defensive opponent")
        
        # Enhanced weather impact with specific effects
        if player_data.get('weather_impact'):
            impact = player_data['weather_impact']
            if impact < -0.2:
                text_parts.append("significantly adverse weather impact")
            elif impact < 0:
                text_parts.append("moderate adverse weather impact")
            elif impact > 0.2:
                text_parts.append("significantly favorable weather impact")
            elif impact > 0:
                text_parts.append("moderate favorable weather impact")
        
        # Enhanced home advantage with specific benefits
        if player_data.get('home_advantage'):
            advantage = player_data['home_advantage']
            if advantage > 0.1:
                text_parts.append("strong home field advantage")
            elif advantage > 0.05:
                text_parts.append("moderate home field advantage")
            elif advantage < -0.1:
                text_parts.append("significant away game disadvantage")
            elif advantage < -0.05:
                text_parts.append("moderate away game disadvantage")
        
        # Enhanced primetime context
        if player_data.get('primetime_bonus'):
            bonus = player_data['primetime_bonus']
            if bonus > 0.1:
                text_parts.append("major primetime game boost")
            elif bonus > 0.05:
                text_parts.append("moderate primetime game boost")
        
        return " ".join(text_parts)
    
    async def generate_embedding(self, player_data: Dict[str, Any]) -> Tuple[List[float], Dict[str, Any]]:
        """Generate contextual embedding for a player."""
        try:
            # Preprocess features
            text_representation = self.preprocess_contextual_features(player_data)
            
            # Generate embedding
            embedding = self.model.encode(text_representation, normalize_embeddings=True)
            
            # Quality metrics
            quality_metrics = {
                'text_length': len(text_representation),
                'context_features': len([k for k, v in player_data.items() if v is not None and k in ['weather_conditions', 'venue', 'opponent_team', 'game_total']]),
                'embedding_norm': np.linalg.norm(embedding),
                'embedding_std': np.std(embedding)
            }
            
            return embedding.tolist(), quality_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to generate contextual embedding: {e}")
            raise


class ValueEmbedder:
    """Specialized embedder for DFS value features."""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", shared_model: Optional[SentenceTransformer] = None):
        self.model_name = model_name
        self.logger = get_logger(__name__)
        self._model = shared_model  # Use shared model if provided
        self.vector_dimensions = 768
        
    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            self.logger.info(f"Loading value embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def preprocess_value_features(self, player_data: Dict[str, Any]) -> str:
        """Preprocess value features into text representation."""
        text_parts = []
        
        # Basic player info
        text_parts.append(f"{player_data.get('player_name', 'Unknown')} {player_data.get('position', 'Unknown')}")
        
        # Salary and ownership
        if player_data.get('salary'):
            text_parts.append(f"salary ${player_data['salary']:,}")
        
        if player_data.get('ownership_percentage'):
            text_parts.append(f"ownership {player_data['ownership_percentage']:.1f}%")
        
        if player_data.get('projected_points'):
            text_parts.append(f"projected {player_data['projected_points']:.1f} points")
        
        # Value metrics
        if player_data.get('value_rating'):
            text_parts.append(f"value rating {player_data['value_rating']:.2f}")
        
        if player_data.get('points_per_dollar'):
            text_parts.append(f"points per dollar {player_data['points_per_dollar']:.2f}")
        
        if player_data.get('roi'):
            text_parts.append(f"ROI {player_data['roi']:.2f}")
        
        # Trends
        if player_data.get('salary_trend'):
            trend = "increasing" if player_data['salary_trend'] > 1 else "decreasing"
            text_parts.append(f"salary {trend}")
        
        if player_data.get('ownership_trend'):
            trend = "increasing" if player_data['ownership_trend'] > 1 else "decreasing"
            text_parts.append(f"ownership {trend}")
        
        # Volatility
        if player_data.get('salary_volatility'):
            volatility = "high" if player_data['salary_volatility'] > 0.15 else "low"
            text_parts.append(f"{volatility} salary volatility")
        
        if player_data.get('ownership_volatility'):
            volatility = "high" if player_data['ownership_volatility'] > 0.25 else "low"
            text_parts.append(f"{volatility} ownership volatility")
        
        # Consistency
        if player_data.get('consistency_rating'):
            consistency = "consistent" if player_data['consistency_rating'] > 7 else "volatile"
            text_parts.append(f"{consistency} performance")
        
        if player_data.get('volatility_score'):
            volatility = "high" if player_data['volatility_score'] > 0.3 else "low"
            text_parts.append(f"{volatility} volatility score")
        
        # Contract factors
        if player_data.get('contract_year_impact'):
            if player_data['contract_year_impact'] > 0:
                text_parts.append("contract year")
        
        if player_data.get('cap_percentage_impact'):
            cap_pct = player_data['cap_percentage_impact'] * 10
            text_parts.append(f"cap percentage {cap_pct:.1f}%")
        
        # Market efficiency
        if player_data.get('ownership_vs_projection'):
            ratio = player_data['ownership_vs_projection']
            if ratio > 1.2:
                text_parts.append("over-owned")
            elif ratio < 0.8:
                text_parts.append("under-owned")
            else:
                text_parts.append("appropriately owned")
        
        return " ".join(text_parts)
    
    async def generate_embedding(self, player_data: Dict[str, Any]) -> Tuple[List[float], Dict[str, Any]]:
        """Generate value embedding for a player."""
        try:
            # Preprocess features
            text_representation = self.preprocess_value_features(player_data)
            
            # Generate embedding
            embedding = self.model.encode(text_representation, normalize_embeddings=True)
            
            # Quality metrics
            quality_metrics = {
                'text_length': len(text_representation),
                'value_features': len([k for k, v in player_data.items() if v is not None and k in ['salary', 'ownership_percentage', 'value_rating', 'roi']]),
                'embedding_norm': np.linalg.norm(embedding),
                'embedding_std': np.std(embedding)
            }
            
            return embedding.tolist(), quality_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to generate value embedding: {e}")
            raise


class MultiVectorEmbeddingGenerator:
    """Multi-vector embedding generator with specialized embedders."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Create shared model instance to avoid multiple loads
        self.logger.info(f"Loading shared embedding model: {self.settings.embedding.model_name}")
        self.shared_model = SentenceTransformer(self.settings.embedding.model_name)
        
        # Initialize specialized embedders with shared model
        self.statistical_embedder = StatisticalEmbedder(self.settings.embedding.model_name, self.shared_model)
        self.contextual_embedder = ContextualEmbedder(self.settings.embedding.model_name, self.shared_model)
        self.value_embedder = ValueEmbedder(self.settings.embedding.model_name, self.shared_model)
        
        # Cache for embeddings
        self.embedding_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
    def _get_cache_key(self, player_id: str, vector_type: str) -> str:
        """Generate cache key for embedding."""
        return f"{player_id}_{vector_type}_{self.settings.embedding.model_name}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached embedding is still valid."""
        if cache_key not in self.embedding_cache:
            return False
        
        cache_entry = self.embedding_cache[cache_key]
        return (time.time() - cache_entry['timestamp']) < self.cache_ttl
    
    async def generate_statistical_embedding(self, player_data: Dict[str, Any]) -> Tuple[List[float], Dict[str, Any]]:
        """Generate statistical embedding for a player."""
        player_id = player_data.get('player_id', 'unknown')
        cache_key = self._get_cache_key(player_id, 'statistical')
        
        if self._is_cache_valid(cache_key):
            self.logger.debug(f"Using cached statistical embedding for {player_id}")
            return self.embedding_cache[cache_key]['embedding'], self.embedding_cache[cache_key]['quality']
        
        embedding, quality = await self.statistical_embedder.generate_embedding(player_data)
        
        # Cache the result
        self.embedding_cache[cache_key] = {
            'embedding': embedding,
            'quality': quality,
            'timestamp': time.time()
        }
        
        return embedding, quality
    
    async def generate_contextual_embedding(self, player_data: Dict[str, Any]) -> Tuple[List[float], Dict[str, Any]]:
        """Generate contextual embedding for a player."""
        player_id = player_data.get('player_id', 'unknown')
        cache_key = self._get_cache_key(player_id, 'contextual')
        
        if self._is_cache_valid(cache_key):
            self.logger.debug(f"Using cached contextual embedding for {player_id}")
            return self.embedding_cache[cache_key]['embedding'], self.embedding_cache[cache_key]['quality']
        
        embedding, quality = await self.contextual_embedder.generate_embedding(player_data)
        
        # Cache the result
        self.embedding_cache[cache_key] = {
            'embedding': embedding,
            'quality': quality,
            'timestamp': time.time()
        }
        
        return embedding, quality
    
    async def generate_value_embedding(self, player_data: Dict[str, Any]) -> Tuple[List[float], Dict[str, Any]]:
        """Generate value embedding for a player."""
        player_id = player_data.get('player_id', 'unknown')
        cache_key = self._get_cache_key(player_id, 'value')
        
        if self._is_cache_valid(cache_key):
            self.logger.debug(f"Using cached value embedding for {player_id}")
            return self.embedding_cache[cache_key]['embedding'], self.embedding_cache[cache_key]['quality']
        
        embedding, quality = await self.value_embedder.generate_embedding(player_data)
        
        # Cache the result
        self.embedding_cache[cache_key] = {
            'embedding': embedding,
            'quality': quality,
            'timestamp': time.time()
        }
        
        return embedding, quality
    
    async def generate_multi_vector_embeddings(self, player_data: Dict[str, Any], vector_types: List[str] = None) -> Dict[str, Tuple[List[float], Dict[str, Any]]]:
        """Generate multiple vector embeddings for a player."""
        if vector_types is None:
            vector_types = ['statistical', 'contextual', 'value']
        
        results = {}
        
        # Generate embeddings concurrently
        tasks = []
        if 'statistical' in vector_types:
            tasks.append(self.generate_statistical_embedding(player_data))
        if 'contextual' in vector_types:
            tasks.append(self.generate_contextual_embedding(player_data))
        if 'value' in vector_types:
            tasks.append(self.generate_value_embedding(player_data))
        
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        vector_type_index = 0
        for vector_type in vector_types:
            if vector_type_index < len(embeddings):
                result = embeddings[vector_type_index]
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to generate {vector_type} embedding: {result}")
                    results[vector_type] = (None, {'error': str(result)})
                else:
                    results[vector_type] = result
                vector_type_index += 1
        
        return results
    
    async def generate_batch_multi_vector_embeddings(self, players_data: List[Dict[str, Any]], vector_types: List[str] = None) -> List[Dict[str, Tuple[List[float], Dict[str, Any]]]]:
        """Generate multi-vector embeddings for a batch of players."""
        if vector_types is None:
            vector_types = ['statistical', 'contextual', 'value']
        
        results = []
        
        # Process players in batches
        batch_size = 10  # Process 10 players at a time
        for i in range(0, len(players_data), batch_size):
            batch = players_data[i:i + batch_size]
            
            # Generate embeddings for batch
            batch_tasks = [self.generate_multi_vector_embeddings(player_data, vector_types) for player_data in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process batch results
            for result in batch_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Failed to generate batch embeddings: {result}")
                    results.append({'error': str(result)})
                else:
                    results.append(result)
        
        return results
    
    def calculate_cross_vector_similarity(self, embeddings1: Dict[str, List[float]], embeddings2: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate similarity between different vector types."""
        similarities = {}
        
        for vector_type in embeddings1.keys():
            if vector_type in embeddings2:
                vec1 = np.array(embeddings1[vector_type])
                vec2 = np.array(embeddings2[vector_type])
                
                # Handle case where embeddings might be nested lists
                if vec1.ndim > 1:
                    vec1 = vec1.flatten()
                if vec2.ndim > 1:
                    vec2 = vec2.flatten()
                
                # Cosine similarity
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                similarities[vector_type] = float(similarity)
        
        return similarities
    
    def validate_embedding_quality(self, embedding: List[float], quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate embedding quality and provide scoring."""
        validation_results = {
            'is_valid': True,
            'quality_score': 0.0,
            'issues': []
        }
        
        # Check embedding dimensions
        if len(embedding) != self.statistical_embedder.vector_dimensions:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Invalid dimensions: {len(embedding)} != {self.statistical_embedder.vector_dimensions}")
        
        # Check embedding norm
        embedding_norm = np.linalg.norm(embedding)
        if embedding_norm < 0.1 or embedding_norm > 2.0:
            validation_results['issues'].append(f"Abnormal embedding norm: {embedding_norm:.3f}")
        
        # Check embedding standard deviation
        embedding_std = np.std(embedding)
        if embedding_std < 0.01 or embedding_std > 0.5:
            validation_results['issues'].append(f"Abnormal embedding std: {embedding_std:.3f}")
        
        # Quality scoring based on metrics
        quality_score = 0.0
        
        # Text length score (longer is better, up to a point)
        if 'text_length' in quality_metrics:
            text_length = quality_metrics['text_length']
            if text_length > 50:
                quality_score += 0.3
            elif text_length > 20:
                quality_score += 0.2
            else:
                quality_score += 0.1
        
        # Feature count score
        feature_count_key = None
        for key in ['feature_count', 'context_features', 'value_features']:
            if key in quality_metrics:
                feature_count_key = key
                break
        
        if feature_count_key:
            feature_count = quality_metrics[feature_count_key]
            if feature_count > 10:
                quality_score += 0.4
            elif feature_count > 5:
                quality_score += 0.3
            else:
                quality_score += 0.1
        
        # Embedding quality score
        if 'embedding_norm' in quality_metrics:
            norm = quality_metrics['embedding_norm']
            if 0.8 <= norm <= 1.2:
                quality_score += 0.3
            else:
                quality_score += 0.1
        
        validation_results['quality_score'] = min(1.0, quality_score)
        
        return validation_results
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        self.logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        cache_size = len(self.embedding_cache)
        cache_keys = list(self.embedding_cache.keys())
        
        # Count by vector type
        vector_type_counts = {}
        for key in cache_keys:
            vector_type = key.split('_')[1] if '_' in key else 'unknown'
            vector_type_counts[vector_type] = vector_type_counts.get(vector_type, 0) + 1
        
        return {
            'cache_size': cache_size,
            'vector_type_counts': vector_type_counts,
            'cache_keys': cache_keys[:10]  # First 10 keys for debugging
        }


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
        
        # Initialize multi-vector generator
        self.multi_vector_generator = MultiVectorEmbeddingGenerator()
        
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
            # Use the shared model from multi-vector generator to avoid multiple loads
            self._model = self.multi_vector_generator.shared_model
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
    
    @performance_timer('generate_multi_vector_embeddings')
    async def generate_multi_vector_embeddings(
        self,
        player: Player,
        vector_types: List[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Tuple[List[float], Dict[str, Any]]]:
        """Generate multiple specialized embeddings for a player."""
        if vector_types is None:
            vector_types = ['statistical', 'contextual', 'value']
        
        # Convert player to dictionary format for multi-vector generator
        player_data = self._player_to_dict(player)
        
        # Generate multi-vector embeddings
        multi_embeddings = await self.multi_vector_generator.generate_multi_vector_embeddings(
            player_data, vector_types
        )
        
        # Update performance metrics
        self.total_embeddings_generated += len(multi_embeddings)
        
        return multi_embeddings
    
    @performance_timer('generate_batch_multi_vector_embeddings')
    async def generate_batch_multi_vector_embeddings(
        self,
        players: List[Player],
        vector_types: List[str] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Tuple[List[float], Dict[str, Any]]]]:
        """Generate multi-vector embeddings for a batch of players."""
        if vector_types is None:
            vector_types = ['statistical', 'contextual', 'value']
        
        # Convert players to dictionary format
        players_data = [self._player_to_dict(player) for player in players]
        
        # Generate batch embeddings
        batch_embeddings = await self.multi_vector_generator.generate_batch_multi_vector_embeddings(
            players_data, vector_types
        )
        
        # Update performance metrics
        self.total_embeddings_generated += len(batch_embeddings) * len(vector_types)
        
        return batch_embeddings
    
    def _player_to_dict(self, player: Player) -> Dict[str, Any]:
        """Convert Player object to dictionary for multi-vector processing."""
        player_data = {
            'player_id': player.player_id,
            'player_name': player.name,
            'position': player.position.value,
            'team': player.team.value,
            'season': player.base.season,
            'week': player.base.week
        }
        
        # Add statistical features
        if player.stats:
            stats = player.stats
            player_data.update({
                'fantasy_points': stats.fantasy_points,
                'passing_yards': stats.passing_yards,
                'rushing_yards': stats.rushing_yards,
                'receiving_yards': stats.receiving_yards,
                'passing_touchdowns': stats.passing_touchdowns,
                'rushing_touchdowns': stats.rushing_touchdowns,
                'receiving_touchdowns': stats.receiving_touchdowns,
                'receptions': stats.receptions,
                'targets': stats.targets,
                'total_touchdowns': (stats.passing_touchdowns or 0) + (stats.rushing_touchdowns or 0) + (stats.receiving_touchdowns or 0)
            })
        
        # Add DFS features
        if player.dfs:
            dfs = player.dfs
            player_data.update({
                'salary': dfs.salary,
                'projected_points': dfs.projected_points,
                'ownership_percentage': dfs.ownership_percentage,
                'value_rating': dfs.value_rating,
                'game_total': dfs.game_total,
                'team_spread': dfs.team_spread,
                'weather_conditions': dfs.weather_conditions,
                'salary_trend': dfs.salary_trend,
                'ownership_trend': dfs.ownership_trend,
                'salary_volatility': dfs.salary_volatility,
                'ownership_volatility': dfs.ownership_volatility,
                'avg_roi': dfs.avg_roi,
                'consistency_rating': dfs.consistency_rating
            })
        
        return player_data
    
    def calculate_cross_vector_similarity(
        self,
        embeddings1: Dict[str, List[float]],
        embeddings2: Dict[str, List[float]]
    ) -> Dict[str, float]:
        """Calculate similarity between different vector types."""
        return self.multi_vector_generator.calculate_cross_vector_similarity(embeddings1, embeddings2)
    
    def validate_embedding_quality(
        self,
        embedding: List[float],
        quality_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate embedding quality and provide scoring."""
        return self.multi_vector_generator.validate_embedding_quality(embedding, quality_metrics)
    
    def get_multi_vector_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for multi-vector embeddings."""
        return self.multi_vector_generator.get_cache_stats()
    
    def clear_multi_vector_cache(self):
        """Clear the multi-vector embedding cache."""
        self.multi_vector_generator.clear_cache()
    
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
    
    @performance_timer('generate_batch_embedding_generation')
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
    async def generate_query_embedding(self, query: str, vector_type: str = "combined") -> List[float]:
        """Generate embedding for search query with vector type specialization."""
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
        
        # Vector type-specific enhancements
        if vector_type == "stats":
            enhanced_query = f"statistical performance {enhanced_query} fantasy points yards touchdowns efficiency"
        elif vector_type == "context":
            enhanced_query = f"game situation {enhanced_query} weather opponent venue matchup context"
        elif vector_type == "value":
            enhanced_query = f"DFS value {enhanced_query} salary ownership ROI market efficiency"
        elif vector_type == "combined":
            enhanced_query = f"comprehensive analysis {enhanced_query} stats context value"
        
        # Repeat the query for emphasis
        enhanced_query = f"{enhanced_query} {enhanced_query}"
        
        embedding = self.model.encode(enhanced_query, normalize_embeddings=True)
        return embedding.tolist()
    
    @performance_timer('generate_stats_query_embedding')
    async def generate_stats_query_embedding(self, query: str) -> List[float]:
        """Generate statistical vector embedding for search query."""
        return await self.generate_query_embedding(query, "stats")
    
    @performance_timer('generate_context_query_embedding')
    async def generate_context_query_embedding(self, query: str) -> List[float]:
        """Generate contextual vector embedding for search query."""
        return await self.generate_query_embedding(query, "context")
    
    @performance_timer('generate_value_query_embedding')
    async def generate_value_query_embedding(self, query: str) -> List[float]:
        """Generate value vector embedding for search query."""
        return await self.generate_query_embedding(query, "value")
    
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



