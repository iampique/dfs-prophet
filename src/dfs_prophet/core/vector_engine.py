"""
Vector engine for DFS Prophet with Qdrant integration and binary quantization.

Features:
- Async QdrantClient connection management
- Collection creation with binary quantization configuration
- Dual collection setup (regular vs binary quantized)
- Vector operations (upsert, search, batch operations)
- Performance monitoring and comparison
- Error handling with retry logic
- Health check methods
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse
import httpx

from ..config import get_settings
from ..utils import get_logger, performance_timer, log_vector_operation
from ..data.models import Player, PlayerVector, QuantizationComparison


class CollectionType(str, Enum):
    """Collection types for vector storage."""
    REGULAR = "regular"
    BINARY_QUANTIZED = "binary_quantized"


class MultiVectorCollectionType(str, Enum):
    """Multi-vector collection types for named vectors."""
    MULTI_VECTOR_REGULAR = "multi_vector_regular"
    MULTI_VECTOR_QUANTIZED = "multi_vector_quantized"


class VectorType(str, Enum):
    """Named vector types for multi-vector collections."""
    STATISTICAL = "stats"
    CONTEXTUAL = "context"
    VALUE = "value"
    COMBINED = "combined"


@dataclass
class NamedVectorConfig:
    """Configuration for named vectors in multi-vector collections."""
    name: str
    size: int
    distance: rest.Distance
    quantization_config: Optional[rest.BinaryQuantization] = None
    on_disk: bool = False


@dataclass
class MultiVectorSearchRequest:
    """Multi-vector search request with named vectors."""
    query_vectors: Dict[str, List[float]]
    vector_weights: Optional[Dict[str, float]] = None
    limit: int = 10
    score_threshold: float = 0.3
    filter_conditions: Optional[Dict[str, Any]] = None
    search_strategy: str = "weighted_combination"  # weighted_combination, max_score, avg_score


@dataclass
class MultiVectorSearchResult:
    """Multi-vector search result with vector-specific scores."""
    player: Player
    combined_score: float
    vector_scores: Dict[str, float]
    vector_id: str
    collection_type: MultiVectorCollectionType
    search_strategy: str
    vector_weights: Optional[Dict[str, float]] = None


@dataclass
class SearchResult:
    """Search result with metadata."""
    player: Player
    score: float
    vector_id: str
    collection_type: CollectionType


@dataclass
class PerformanceMetrics:
    """Performance metrics for vector operations."""
    operation: str
    collection_type: CollectionType
    duration_ms: float
    vector_count: int
    memory_usage_mb: Optional[float] = None
    accuracy: Optional[float] = None


@dataclass
class MultiVectorPerformanceMetrics:
    """Performance metrics for multi-vector operations."""
    operation: str
    collection_type: MultiVectorCollectionType
    duration_ms: float
    vector_count: int
    vector_types: List[str]
    memory_usage_mb: Optional[float] = None
    accuracy: Optional[float] = None
    vector_type_performance: Dict[str, float] = None  # Performance per vector type


class VectorEngine:
    """Core vector engine with Qdrant integration and binary quantization."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Client management
        self._async_client: Optional[AsyncQdrantClient] = None
        self._sync_client: Optional[QdrantClient] = None
        
        # Performance tracking
        self.performance_metrics: List[PerformanceMetrics] = []
        self.multi_vector_performance_metrics: List[MultiVectorPerformanceMetrics] = []
        self.connection_health = True
        self.last_health_check = 0
        
        # Collection names
        self.regular_collection = f"{self.settings.vector_db.players_collection}_regular"
        self.quantized_collection = f"{self.settings.vector_db.players_collection}_quantized"
        
        # Multi-vector collection names
        self.multi_vector_regular_collection = f"{self.settings.vector_db.players_collection}_multi_regular"
        self.multi_vector_quantized_collection = f"{self.settings.vector_db.players_collection}_multi_quantized"
        
        # Named vector configurations
        self.named_vector_configs = {
            VectorType.STATISTICAL: NamedVectorConfig(
                name=VectorType.STATISTICAL.value,
                size=768,
                distance=rest.Distance.COSINE,
                on_disk=self.settings.vector_db.on_disk_payload
            ),
            VectorType.CONTEXTUAL: NamedVectorConfig(
                name=VectorType.CONTEXTUAL.value,
                size=768,
                distance=rest.Distance.COSINE,
                on_disk=self.settings.vector_db.on_disk_payload
            ),
            VectorType.VALUE: NamedVectorConfig(
                name=VectorType.VALUE.value,
                size=768,
                distance=rest.Distance.COSINE,
                on_disk=self.settings.vector_db.on_disk_payload
            ),
            VectorType.COMBINED: NamedVectorConfig(
                name=VectorType.COMBINED.value,
                size=768,
                distance=rest.Distance.COSINE,
                on_disk=self.settings.vector_db.on_disk_payload
            )
        }
    
    @property
    def async_client(self) -> AsyncQdrantClient:
        """Get async Qdrant client with lazy initialization."""
        if self._async_client is None:
            self._async_client = AsyncQdrantClient(
                url=self.settings.get_qdrant_url(),
                api_key=self.settings.qdrant.api_key,
                timeout=self.settings.qdrant.timeout,
                prefer_grpc=self.settings.qdrant.prefer_grpc
            )
        return self._async_client
    
    @property
    def sync_client(self) -> QdrantClient:
        """Get sync Qdrant client with lazy initialization."""
        if self._sync_client is None:
            self._sync_client = QdrantClient(
                url=self.settings.get_qdrant_url(),
                api_key=self.settings.qdrant.api_key,
                timeout=self.settings.qdrant.timeout,
                prefer_grpc=self.settings.qdrant.prefer_grpc
            )
        return self._sync_client
    
    async def initialize_collections(self) -> None:
        """Initialize both regular and binary quantized collections."""
        self.logger.info("Initializing vector collections...")
        
        # Create regular collection
        await self._create_regular_collection()
        
        # Create binary quantized collection
        await self._create_quantized_collection()
        
        # Create multi-vector collections
        await self._create_multi_vector_collections()
        
        self.logger.info("Vector collections initialized successfully")
    
    async def _create_regular_collection(self) -> None:
        """Create regular collection without quantization."""
        try:
            # Check if collection exists
            collections = await self.async_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.regular_collection not in collection_names:
                self.logger.info(f"Creating regular collection: {self.regular_collection}")
                
                await self.async_client.create_collection(
                    collection_name=self.regular_collection,
                    vectors_config=rest.VectorParams(
                        size=self.settings.vector_db.vector_dimensions,
                        distance=rest.Distance.COSINE,
                        on_disk=self.settings.vector_db.on_disk_payload
                    )
                )
                
                # Note: Payload indexes can be added later for performance optimization
                
                self.logger.info(f"Regular collection {self.regular_collection} created successfully")
            else:
                self.logger.info(f"Regular collection {self.regular_collection} already exists")
                
        except Exception as e:
            self.logger.error(f"Failed to create regular collection: {e}")
            raise
    
    async def _create_quantized_collection(self) -> None:
        """Create binary quantized collection for memory efficiency."""
        try:
            # Check if collection exists
            collections = await self.async_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.quantized_collection not in collection_names:
                self.logger.info(f"Creating binary quantized collection: {self.quantized_collection}")
                
                # Binary quantization configuration
                quantization_config = rest.BinaryQuantization(
                    binary=rest.BinaryQuantizationConfig(
                        always_ram=self.settings.binary_quantization.always_ram
                    )
                )
                
                await self.async_client.create_collection(
                    collection_name=self.quantized_collection,
                    vectors_config=rest.VectorParams(
                        size=self.settings.vector_db.vector_dimensions,
                        distance=rest.Distance.COSINE,
                        on_disk=self.settings.vector_db.on_disk_payload,
                        quantization_config=quantization_config
                    )
                )
                
                # Note: Payload indexes can be added later for performance optimization
                
                self.logger.info(f"Binary quantized collection {self.quantized_collection} created successfully")
            else:
                self.logger.info(f"Binary quantized collection {self.quantized_collection} already exists")
                
        except Exception as e:
            self.logger.error(f"Failed to create binary quantized collection: {e}")
            raise
    
    async def _create_multi_vector_collections(self) -> None:
        """Create multi-vector collections with named vectors."""
        try:
            # Create multi-vector regular collection
            await self._create_multi_vector_regular_collection()
            
            # Create multi-vector quantized collection
            await self._create_multi_vector_quantized_collection()
            
            self.logger.info("Multi-vector collections initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to create multi-vector collections: {e}")
            raise
    
    async def _create_multi_vector_regular_collection(self) -> None:
        """Create multi-vector regular collection with named vectors."""
        try:
            # Check if collection exists
            collections = await self.async_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.multi_vector_regular_collection not in collection_names:
                self.logger.info(f"Creating multi-vector regular collection: {self.multi_vector_regular_collection}")
                
                # Create named vectors configuration
                named_vectors = {}
                for vector_type, config in self.named_vector_configs.items():
                    named_vectors[vector_type.value] = rest.VectorParams(
                        size=config.size,
                        distance=config.distance,
                        on_disk=config.on_disk
                    )
                
                await self.async_client.create_collection(
                    collection_name=self.multi_vector_regular_collection,
                    vectors_config=named_vectors
                )
                
                self.logger.info(f"Multi-vector regular collection {self.multi_vector_regular_collection} created successfully")
            else:
                self.logger.info(f"Multi-vector regular collection {self.multi_vector_regular_collection} already exists")
                
        except Exception as e:
            self.logger.error(f"Failed to create multi-vector regular collection: {e}")
            raise
    
    async def _create_multi_vector_quantized_collection(self) -> None:
        """Create multi-vector quantized collection with named vectors and binary quantization."""
        try:
            # Check if collection exists
            collections = await self.async_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.multi_vector_quantized_collection not in collection_names:
                self.logger.info(f"Creating multi-vector quantized collection: {self.multi_vector_quantized_collection}")
                
                # Create named vectors configuration with quantization
                named_vectors = {}
                for vector_type, config in self.named_vector_configs.items():
                    # Binary quantization configuration
                    quantization_config = rest.BinaryQuantization(
                        binary=rest.BinaryQuantizationConfig(
                            always_ram=self.settings.binary_quantization.always_ram
                        )
                    )
                    
                    named_vectors[vector_type.value] = rest.VectorParams(
                        size=config.size,
                        distance=config.distance,
                        on_disk=config.on_disk,
                        quantization_config=quantization_config
                    )
                
                await self.async_client.create_collection(
                    collection_name=self.multi_vector_quantized_collection,
                    vectors_config=named_vectors
                )
                
                self.logger.info(f"Multi-vector quantized collection {self.multi_vector_quantized_collection} created successfully")
            else:
                self.logger.info(f"Multi-vector quantized collection {self.multi_vector_quantized_collection} already exists")
                
        except Exception as e:
            self.logger.error(f"Failed to create multi-vector quantized collection: {e}")
            raise
    
    @performance_timer('upsert_player_vector')
    async def upsert_player_vector(
        self,
        player: Player,
        vector: List[float],
        collection_type: CollectionType = CollectionType.REGULAR
    ) -> bool:
        """Upsert a player vector to the specified collection."""
        try:
            if collection_type == CollectionType.REGULAR:
                collection_name = self.regular_collection
            elif collection_type == CollectionType.BINARY_QUANTIZED:
                collection_name = self.quantized_collection
            else:
                raise ValueError(f"Unsupported collection type: {collection_type}")
            
            # Prepare payload
            payload = {
                "player_id": player.player_id,
                "name": player.name,
                "position": player.position.value,
                "team": player.team.value,
                "season": player.base.season,
                "week": player.base.week,
                "timestamp": time.time()
            }
            
            # Add stats to payload if available
            if player.stats:
                payload.update({
                    "fantasy_points": player.stats.fantasy_points,
                    "passing_yards": player.stats.passing_yards,
                    "rushing_yards": player.stats.rushing_yards,
                    "receiving_yards": player.stats.receiving_yards,
                })
            
            # Add DFS data to payload if available
            if player.dfs:
                payload.update({
                    "salary": player.dfs.salary,
                    "projected_points": player.dfs.projected_points,
                    "ownership_percentage": player.dfs.ownership_percentage,
                })
            
            # Upsert vector
            await self.async_client.upsert(
                collection_name=collection_name,
                points=[
                    rest.PointStruct(
                        id=hash(player.player_id) % (2**63),  # Convert to integer ID
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            
            log_vector_operation(
                "UPSERT",
                collection_name,
                1,
                time.time()
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upsert player vector: {e}")
            return False
    
    @performance_timer('batch_upsert_vectors')
    async def batch_upsert_vectors(
        self,
        players: List[Player],
        vectors: List[List[float]],
        collection_type: CollectionType = CollectionType.REGULAR,
        batch_size: int = 100
    ) -> int:
        """Batch upsert vectors with configurable batch size."""
        try:
            if collection_type == CollectionType.REGULAR:
                collection_name = self.regular_collection
            elif collection_type == CollectionType.BINARY_QUANTIZED:
                collection_name = self.quantized_collection
            else:
                raise ValueError(f"Unsupported collection type: {collection_type}")
            
            total_upserted = 0
            
            # Process in batches
            for i in range(0, len(players), batch_size):
                batch_players = players[i:i + batch_size]
                batch_vectors = vectors[i:i + batch_size]
                
                points = []
                for player, vector in zip(batch_players, batch_vectors):
                    # Prepare payload
                    payload = {
                        "player_id": player.player_id,
                        "name": player.name,
                        "position": player.position.value,
                        "team": player.team.value,
                        "season": player.base.season,
                        "week": player.base.week,
                        "timestamp": time.time()
                    }
                    
                    # Add stats to payload if available
                    if player.stats:
                        payload.update({
                            "fantasy_points": player.stats.fantasy_points,
                            "passing_yards": player.stats.passing_yards,
                            "rushing_yards": player.stats.rushing_yards,
                            "receiving_yards": player.stats.receiving_yards,
                        })
                    
                    # Add DFS data to payload if available
                    if player.dfs:
                        payload.update({
                            "salary": player.dfs.salary,
                            "projected_points": player.dfs.projected_points,
                            "ownership_percentage": player.dfs.ownership_percentage,
                        })
                    
                    points.append(
                        rest.PointStruct(
                            id=hash(player.player_id) % (2**63),  # Convert to integer ID
                            vector=vector,
                            payload=payload
                        )
                    )
                
                # Upsert batch
                await self.async_client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                
                total_upserted += len(points)
                
                log_vector_operation(
                    "BATCH_UPSERT",
                    collection_name,
                    len(points),
                    time.time()
                )
            
            return total_upserted
            
        except Exception as e:
            self.logger.error(f"Failed to batch upsert vectors: {e}")
            return 0
    
    @performance_timer('upsert_multi_vector_player')
    async def upsert_multi_vector_player(
        self,
        player: Player,
        multi_vectors: Dict[str, List[float]],
        collection_type: MultiVectorCollectionType = MultiVectorCollectionType.MULTI_VECTOR_REGULAR
    ) -> bool:
        """Upsert a player with multiple named vectors."""
        try:
            collection_name = (
                self.multi_vector_regular_collection if collection_type == MultiVectorCollectionType.MULTI_VECTOR_REGULAR
                else self.multi_vector_quantized_collection
            )
            
            # Prepare payload
            payload = {
                "player_id": player.player_id,
                "name": player.name,
                "position": player.position.value,
                "team": player.team.value,
                "season": player.base.season,
                "week": player.base.week,
                "timestamp": time.time(),
                "vector_types": list(multi_vectors.keys())
            }
            
            # Add stats to payload if available
            if player.stats:
                payload.update({
                    "fantasy_points": player.stats.fantasy_points,
                    "passing_yards": player.stats.passing_yards,
                    "rushing_yards": player.stats.rushing_yards,
                    "receiving_yards": player.stats.receiving_yards,
                })
            
            # Add DFS data to payload if available
            if player.dfs:
                payload.update({
                    "salary": player.dfs.salary,
                    "projected_points": player.dfs.projected_points,
                    "ownership_percentage": player.dfs.ownership_percentage,
                })
            
            # Prepare named vectors
            vectors = {}
            for vector_type, vector_data in multi_vectors.items():
                if vector_type in self.named_vector_configs:
                    vectors[vector_type] = vector_data
            
            # Upsert multi-vector point
            await self.async_client.upsert(
                collection_name=collection_name,
                points=[
                    rest.PointStruct(
                        id=hash(player.player_id) % (2**63),  # Convert to integer ID
                        vector=vectors,  # Named vectors go as dictionary
                        payload=payload
                    )
                ]
            )
            
            log_vector_operation(
                "MULTI_VECTOR_UPSERT",
                collection_name,
                1,
                time.time()
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upsert multi-vector player: {e}")
            return False
    
    @performance_timer('batch_upsert_multi_vector_players')
    async def batch_upsert_multi_vector_players(
        self,
        players: List[Player],
        multi_vectors_list: List[Dict[str, List[float]]],
        collection_type: MultiVectorCollectionType = MultiVectorCollectionType.MULTI_VECTOR_REGULAR,
        batch_size: int = 50
    ) -> int:
        """Batch upsert players with multiple named vectors."""
        try:
            collection_name = (
                self.multi_vector_regular_collection if collection_type == MultiVectorCollectionType.MULTI_VECTOR_REGULAR
                else self.multi_vector_quantized_collection
            )
            
            total_upserted = 0
            
            # Process in batches
            for i in range(0, len(players), batch_size):
                batch_players = players[i:i + batch_size]
                batch_multi_vectors = multi_vectors_list[i:i + batch_size]
                
                points = []
                all_vectors = {}
                for player, multi_vectors in zip(batch_players, batch_multi_vectors):
                    # Prepare payload
                    payload = {
                        "player_id": player.player_id,
                        "name": player.name,
                        "position": player.position.value,
                        "team": player.team.value,
                        "season": player.base.season,
                        "week": player.base.week,
                        "timestamp": time.time(),
                        "vector_types": list(multi_vectors.keys())
                    }
                    
                    # Add stats to payload if available
                    if player.stats:
                        payload.update({
                            "fantasy_points": player.stats.fantasy_points,
                            "passing_yards": player.stats.passing_yards,
                            "rushing_yards": player.stats.rushing_yards,
                            "receiving_yards": player.stats.receiving_yards,
                        })
                    
                    # Add DFS data to payload if available
                    if player.dfs:
                        payload.update({
                            "salary": player.dfs.salary,
                            "projected_points": player.dfs.projected_points,
                            "ownership_percentage": player.dfs.ownership_percentage,
                        })
                    
                    # Prepare named vectors
                    player_id = hash(player.player_id) % (2**63)
                    if player_id not in all_vectors:
                        all_vectors[player_id] = {}
                    for vector_type, vector_data in multi_vectors.items():
                        if vector_type in self.named_vector_configs:
                            all_vectors[player_id][vector_type] = vector_data
                    
                    points.append(
                        rest.PointStruct(
                            id=player_id,
                            vector=all_vectors.get(player_id, {}),  # Named vectors go as dictionary
                            payload=payload
                        )
                    )
                
                # Upsert batch with named vectors
                await self.async_client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                
                total_upserted += len(points)
                
                log_vector_operation(
                    "BATCH_MULTI_VECTOR_UPSERT",
                    collection_name,
                    len(points),
                    time.time()
                )
            
            return total_upserted
            
        except Exception as e:
            self.logger.error(f"Failed to batch upsert multi-vector players: {e}")
            return 0
    
    @performance_timer('search_vectors')
    async def search_vectors(
        self,
        query_vector: List[float],
        collection_type: CollectionType = CollectionType.REGULAR,
        limit: int = 10,
        score_threshold: float = 0.5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Search vectors in the specified collection."""
        try:
            if collection_type == CollectionType.REGULAR:
                collection_name = self.regular_collection
            elif collection_type == CollectionType.BINARY_QUANTIZED:
                collection_name = self.quantized_collection
            else:
                raise ValueError(f"Unsupported collection type: {collection_type}")
            
            # Build search request
            search_request = rest.SearchRequest(
                vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                with_vector=False
            )
            
            # Add filters if provided
            if filters:
                search_request.filter = self._build_filter(filters)
            
            # Execute search
            search_results = await self.async_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True
            )
            
            # Convert to SearchResult objects
            results = []
            for result in search_results:
                player = self._payload_to_player(result.payload)
                if player:
                    results.append(SearchResult(
                        player=player,
                        score=result.score,
                        vector_id=str(result.id),
                        collection_type=collection_type
                    ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search vectors: {e}")
            return []
    
    @performance_timer('search_multi_vector')
    async def search_multi_vector(
        self,
        search_request: MultiVectorSearchRequest,
        collection_type: MultiVectorCollectionType = MultiVectorCollectionType.MULTI_VECTOR_REGULAR
    ) -> List[MultiVectorSearchResult]:
        """Search using multiple named vectors with advanced strategies."""
        try:
            collection_name = (
                self.multi_vector_regular_collection if collection_type == MultiVectorCollectionType.MULTI_VECTOR_REGULAR
                else self.multi_vector_quantized_collection
            )
            
            # Build search request based on strategy
            if search_request.search_strategy == "weighted_combination":
                return await self._weighted_combination_search(search_request, collection_name)
            elif search_request.search_strategy == "max_score":
                return await self._max_score_search(search_request, collection_name)
            elif search_request.search_strategy == "avg_score":
                return await self._avg_score_search(search_request, collection_name)
            else:
                # Default to weighted combination
                return await self._weighted_combination_search(search_request, collection_name)
                
        except Exception as e:
            self.logger.error(f"Failed to search multi-vector: {e}")
            return []
    
    async def _weighted_combination_search(
        self,
        search_request: MultiVectorSearchRequest,
        collection_name: str
    ) -> List[MultiVectorSearchResult]:
        """Perform weighted combination search across multiple vectors."""
        try:
            # Default weights if not provided
            if not search_request.vector_weights:
                search_request.vector_weights = {
                    vector_type: 1.0 / len(search_request.query_vectors)
                    for vector_type in search_request.query_vectors.keys()
                }
            
            # Build search requests for each vector type
            search_requests = []
            for vector_type, query_vector in search_request.query_vectors.items():
                search_requests.append({
                    "collection_name": collection_name,
                    "query_vector": query_vector,
                    "limit": search_request.limit * 2,  # Get more results for combination
                    "score_threshold": search_request.score_threshold,
                    "with_payload": True,
                    "vector_name": "stats"  # Use stats vector for now
                })
            
            # Execute searches concurrently
            search_tasks = [
                self.async_client.search(**search_req) for search_req in search_requests
            ]
            
            search_results_list = await asyncio.gather(*search_tasks)
            
            # Combine and weight results
            combined_results = {}
            for vector_type, results in zip(search_request.query_vectors.keys(), search_results_list):
                weight = search_request.vector_weights.get(vector_type, 1.0)
                for result in results:
                    player_id = result.payload.get("player_id")
                    if player_id not in combined_results:
                        combined_results[player_id] = {
                            "player": self._payload_to_player(result.payload),
                            "vector_scores": {},
                            "payload": result.payload
                        }
                    combined_results[player_id]["vector_scores"][vector_type] = result.score * weight
            
            # Calculate combined scores
            final_results = []
            for player_id, data in combined_results.items():
                if data["player"]:
                    combined_score = sum(data["vector_scores"].values())
                    final_results.append(MultiVectorSearchResult(
                        player=data["player"],
                        combined_score=combined_score,
                        vector_scores=data["vector_scores"],
                        vector_id=player_id,
                        collection_type=MultiVectorCollectionType.MULTI_VECTOR_REGULAR,
                        search_strategy="weighted_combination",
                        vector_weights=search_request.vector_weights
                    ))
            
            # Sort by combined score and limit
            final_results.sort(key=lambda x: x.combined_score, reverse=True)
            return final_results[:search_request.limit]
            
        except Exception as e:
            self.logger.error(f"Failed to perform weighted combination search: {e}")
            return []
    
    async def _max_score_search(
        self,
        search_request: MultiVectorSearchRequest,
        collection_name: str
    ) -> List[MultiVectorSearchResult]:
        """Perform max score search across multiple vectors."""
        try:
            # Execute searches for each vector type
            all_results = {}
            
            for vector_type, query_vector in search_request.query_vectors.items():
                search_results = await self.async_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=search_request.limit,
                    score_threshold=search_request.score_threshold,
                    with_payload=True,
                    vector_name="stats"  # Use stats vector for now
                )
                
                for result in search_results:
                    player_id = result.payload.get("player_id")
                    if player_id not in all_results:
                        all_results[player_id] = {
                            "player": self._payload_to_player(result.payload),
                            "vector_scores": {},
                            "payload": result.payload
                        }
                    all_results[player_id]["vector_scores"][vector_type] = result.score
            
            # Calculate max scores
            final_results = []
            for player_id, data in all_results.items():
                if data["player"]:
                    max_score = max(data["vector_scores"].values())
                    final_results.append(MultiVectorSearchResult(
                        player=data["player"],
                        combined_score=max_score,
                        vector_scores=data["vector_scores"],
                        vector_id=player_id,
                        collection_type=MultiVectorCollectionType.MULTI_VECTOR_REGULAR,
                        search_strategy="max_score"
                    ))
            
            # Sort by max score and limit
            final_results.sort(key=lambda x: x.combined_score, reverse=True)
            return final_results[:search_request.limit]
            
        except Exception as e:
            self.logger.error(f"Failed to perform max score search: {e}")
            return []
    
    async def _avg_score_search(
        self,
        search_request: MultiVectorSearchRequest,
        collection_name: str
    ) -> List[MultiVectorSearchResult]:
        """Perform average score search across multiple vectors."""
        try:
            # Execute searches for each vector type
            all_results = {}
            
            for vector_type, query_vector in search_request.query_vectors.items():
                search_results = await self.async_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=search_request.limit,
                    score_threshold=search_request.score_threshold,
                    with_payload=True,
                    vector_name="stats"  # Use stats vector for now
                )
                
                for result in search_results:
                    player_id = result.payload.get("player_id")
                    if player_id not in all_results:
                        all_results[player_id] = {
                            "player": self._payload_to_player(result.payload),
                            "vector_scores": {},
                            "payload": result.payload
                        }
                    all_results[player_id]["vector_scores"][vector_type] = result.score
            
            # Calculate average scores
            final_results = []
            for player_id, data in all_results.items():
                if data["player"]:
                    avg_score = sum(data["vector_scores"].values()) / len(data["vector_scores"])
                    final_results.append(MultiVectorSearchResult(
                        player=data["player"],
                        combined_score=avg_score,
                        vector_scores=data["vector_scores"],
                        vector_id=player_id,
                        collection_type=MultiVectorCollectionType.MULTI_VECTOR_REGULAR,
                        search_strategy="avg_score"
                    ))
            
            # Sort by average score and limit
            final_results.sort(key=lambda x: x.combined_score, reverse=True)
            return final_results[:search_request.limit]
            
        except Exception as e:
            self.logger.error(f"Failed to perform average score search: {e}")
            return []
    
    @performance_timer('selective_vector_search')
    async def selective_vector_search(
        self,
        query_vector: List[float],
        vector_type: str,
        collection_type: MultiVectorCollectionType = MultiVectorCollectionType.MULTI_VECTOR_REGULAR,
        limit: int = 10,
        score_threshold: float = 0.3,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[MultiVectorSearchResult]:
        """Search specific vector type in multi-vector collection."""
        try:
            collection_name = (
                self.multi_vector_regular_collection if collection_type == MultiVectorCollectionType.MULTI_VECTOR_REGULAR
                else self.multi_vector_quantized_collection
            )
            
            # Execute search with specific vector type
            search_results = await self.async_client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
                vector_name=vector_type  # Use the specified vector type
            )
            
            # Convert to MultiVectorSearchResult objects
            results = []
            for result in search_results:
                player = self._payload_to_player(result.payload)
                if player:
                    results.append(MultiVectorSearchResult(
                        player=player,
                        combined_score=result.score,
                        vector_scores={vector_type: result.score},
                        vector_id=str(result.id),
                        collection_type=collection_type,
                        search_strategy="selective"
                    ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to perform selective vector search: {e}")
            return []
    
    @performance_timer('compare_multi_vector_performance')
    async def compare_multi_vector_performance(
        self,
        test_queries: List[MultiVectorSearchRequest],
        collection_type: MultiVectorCollectionType = MultiVectorCollectionType.MULTI_VECTOR_REGULAR
    ) -> Dict[str, Any]:
        """Compare performance across different multi-vector search strategies."""
        try:
            performance_results = {
                "weighted_combination": {"times": [], "results_count": [], "avg_scores": []},
                "max_score": {"times": [], "results_count": [], "avg_scores": []},
                "avg_score": {"times": [], "results_count": [], "avg_scores": []}
            }
            
            strategies = ["weighted_combination", "max_score", "avg_score"]
            
            for strategy in strategies:
                for query in test_queries:
                    query.search_strategy = strategy
                    
                    start_time = time.time()
                    results = await self.search_multi_vector(query, collection_type)
                    end_time = time.time()
                    
                    performance_results[strategy]["times"].append(end_time - start_time)
                    performance_results[strategy]["results_count"].append(len(results))
                    
                    if results:
                        avg_score = sum(r.combined_score for r in results) / len(results)
                        performance_results[strategy]["avg_scores"].append(avg_score)
                    else:
                        performance_results[strategy]["avg_scores"].append(0.0)
            
            # Calculate summary statistics
            summary = {}
            for strategy, data in performance_results.items():
                summary[strategy] = {
                    "avg_time_ms": sum(data["times"]) * 1000 / len(data["times"]),
                    "avg_results_count": sum(data["results_count"]) / len(data["results_count"]),
                    "avg_score": sum(data["avg_scores"]) / len(data["avg_scores"]),
                    "total_queries": len(data["times"])
                }
            
            return {
                "detailed_results": performance_results,
                "summary": summary,
                "recommendation": self._get_performance_recommendation(summary)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to compare multi-vector performance: {e}")
            return {}
    
    @performance_timer('ab_test_vector_combinations')
    async def ab_test_vector_combinations(
        self,
        test_queries: List[MultiVectorSearchRequest],
        vector_combinations: List[Dict[str, float]],
        collection_type: MultiVectorCollectionType = MultiVectorCollectionType.MULTI_VECTOR_REGULAR
    ) -> Dict[str, Any]:
        """A/B test different vector weight combinations."""
        try:
            ab_test_results = {}
            
            for i, combination in enumerate(vector_combinations):
                combination_name = f"combination_{i+1}"
                ab_test_results[combination_name] = {
                    "weights": combination,
                    "times": [],
                    "results_count": [],
                    "avg_scores": [],
                    "top_scores": []
                }
                
                for query in test_queries:
                    query.vector_weights = combination
                    query.search_strategy = "weighted_combination"
                    
                    start_time = time.time()
                    results = await self.search_multi_vector(query, collection_type)
                    end_time = time.time()
                    
                    ab_test_results[combination_name]["times"].append(end_time - start_time)
                    ab_test_results[combination_name]["results_count"].append(len(results))
                    
                    if results:
                        avg_score = sum(r.combined_score for r in results) / len(results)
                        top_score = max(r.combined_score for r in results)
                        ab_test_results[combination_name]["avg_scores"].append(avg_score)
                        ab_test_results[combination_name]["top_scores"].append(top_score)
                    else:
                        ab_test_results[combination_name]["avg_scores"].append(0.0)
                        ab_test_results[combination_name]["top_scores"].append(0.0)
            
            # Calculate summary statistics
            summary = {}
            for combination_name, data in ab_test_results.items():
                summary[combination_name] = {
                    "weights": data["weights"],
                    "avg_time_ms": sum(data["times"]) * 1000 / len(data["times"]),
                    "avg_results_count": sum(data["results_count"]) / len(data["results_count"]),
                    "avg_score": sum(data["avg_scores"]) / len(data["avg_scores"]),
                    "avg_top_score": sum(data["top_scores"]) / len(data["top_scores"]),
                    "total_queries": len(data["times"])
                }
            
            return {
                "detailed_results": ab_test_results,
                "summary": summary,
                "best_combination": self._get_best_combination(summary)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to A/B test vector combinations: {e}")
            return {}
    
    def _payload_to_player(self, payload: Dict[str, Any]) -> Optional[Player]:
        """Convert payload to Player object."""
        try:
            from ..data.models import PlayerBase, PlayerStats, PlayerDFS, Position, Team
            
            # Create PlayerBase
            base = PlayerBase(
                player_id=payload.get("player_id", ""),
                name=payload.get("name", ""),
                position=Position(payload.get("position", "QB")),
                team=Team(payload.get("team", "ARI")),
                season=payload.get("season", 2024),
                week=payload.get("week", 1)
            )
            
            # Create PlayerStats if available
            stats = None
            if any(key in payload for key in ["fantasy_points", "passing_yards", "rushing_yards", "receiving_yards"]):
                stats = PlayerStats(
                    fantasy_points=payload.get("fantasy_points"),
                    passing_yards=payload.get("passing_yards"),
                    rushing_yards=payload.get("rushing_yards"),
                    receiving_yards=payload.get("receiving_yards")
                )
            
            # Create PlayerDFS if available
            dfs = None
            if any(key in payload for key in ["salary", "projected_points", "ownership_percentage"]):
                dfs = PlayerDFS(
                    salary=payload.get("salary"),
                    projected_points=payload.get("projected_points"),
                    ownership_percentage=payload.get("ownership_percentage")
                )
            
            return Player(base=base, stats=stats, dfs=dfs)
            
        except Exception as e:
            self.logger.error(f"Failed to convert payload to Player: {e}")
            return None
    
    def _build_filter(self, filters: Dict[str, Any]) -> rest.Filter:
        """Build Qdrant filter from dictionary."""
        try:
            conditions = []
            
            for key, value in filters.items():
                if key == "position" and value:
                    conditions.append(
                        rest.FieldCondition(
                            key="position",
                            match=rest.MatchValue(value=value)
                        )
                    )
                elif key == "team" and value:
                    conditions.append(
                        rest.FieldCondition(
                            key="team",
                            match=rest.MatchValue(value=value)
                        )
                    )
                elif key == "season" and value:
                    conditions.append(
                        rest.FieldCondition(
                            key="season",
                            match=rest.MatchValue(value=value)
                        )
                    )
                elif key == "week" and value:
                    conditions.append(
                        rest.FieldCondition(
                            key="week",
                            match=rest.MatchValue(value=value)
                        )
                    )
                elif key == "min_fantasy_points" and value:
                    conditions.append(
                        rest.FieldCondition(
                            key="fantasy_points",
                            range=rest.DatetimeRange(
                                gte=value
                            )
                        )
                    )
                elif key == "max_salary" and value:
                    conditions.append(
                        rest.FieldCondition(
                            key="salary",
                            range=rest.DatetimeRange(
                                lte=value
                            )
                        )
                    )
            
            return rest.Filter(must=conditions) if conditions else None
            
        except Exception as e:
            self.logger.error(f"Failed to build filter: {e}")
            return None
    
    def _get_performance_recommendation(self, summary: Dict[str, Any]) -> str:
        """Get performance recommendation based on summary statistics."""
        try:
            best_strategy = None
            best_score = -1
            
            for strategy, data in summary.items():
                if data["avg_score"] > best_score:
                    best_score = data["avg_score"]
                    best_strategy = strategy
            
            if best_strategy:
                return f"Recommended strategy: {best_strategy} (avg score: {best_score:.3f})"
            else:
                return "No clear recommendation available"
                
        except Exception as e:
            self.logger.error(f"Failed to get performance recommendation: {e}")
            return "Unable to determine recommendation"
    
    def _get_best_combination(self, summary: Dict[str, Any]) -> str:
        """Get best vector combination based on A/B test results."""
        try:
            best_combination = None
            best_score = -1
            
            for combination_name, data in summary.items():
                if data["avg_score"] > best_score:
                    best_score = data["avg_score"]
                    best_combination = combination_name
            
            if best_combination:
                weights = summary[best_combination]["weights"]
                return f"Best combination: {best_combination} with weights {weights} (avg score: {best_score:.3f})"
            else:
                return "No clear best combination available"
                
        except Exception as e:
            self.logger.error(f"Failed to get best combination: {e}")
            return "Unable to determine best combination"
    
    @performance_timer('compare_collections_performance')
    async def compare_collections_performance(
        self,
        query_vector: List[float],
        test_queries: int = 10
    ) -> QuantizationComparison:
        """Compare performance between regular and binary quantized collections."""
        try:
            self.logger.info("Starting performance comparison between collections...")
            
            # Test regular collection
            regular_times = []
            for _ in range(test_queries):
                start_time = time.time()
                await self.search_vectors(
                    query_vector,
                    CollectionType.REGULAR,
                    limit=10
                )
                regular_times.append((time.time() - start_time) * 1000)  # Convert to ms
            
            # Test quantized collection
            quantized_times = []
            for _ in range(test_queries):
                start_time = time.time()
                await self.search_vectors(
                    query_vector,
                    CollectionType.BINARY_QUANTIZED,
                    limit=10
                )
                quantized_times.append((time.time() - start_time) * 1000)  # Convert to ms
            
            # Calculate metrics
            avg_regular_time = np.mean(regular_times)
            avg_quantized_time = np.mean(quantized_times)
            
            # Get collection info
            regular_info = await self.async_client.get_collection(self.regular_collection)
            quantized_info = await self.async_client.get_collection(self.quantized_collection)
            
            regular_size = regular_info.points_count * self.settings.vector_db.vector_dimensions * 4  # 4 bytes per float
            quantized_size = quantized_info.points_count * self.settings.vector_db.vector_dimensions * 1  # 1 byte per quantized value
            
            # Calculate improvements
            speed_improvement = ((avg_regular_time - avg_quantized_time) / avg_regular_time) * 100 if avg_regular_time > 0 else 0
            memory_savings = ((regular_size - quantized_size) / regular_size) * 100 if regular_size > 0 else 0
            
            comparison = QuantizationComparison(
                original_size_mb=regular_size / (1024 * 1024),
                quantized_size_mb=quantized_size / (1024 * 1024),
                compression_ratio=quantized_size / regular_size,
                memory_savings_percent=memory_savings,
                search_speed_original_ms=avg_regular_time,
                search_speed_quantized_ms=avg_quantized_time,
                speed_improvement_percent=speed_improvement,
                vector_count=regular_info.points_count,
                test_queries=test_queries
            )
            
            self.logger.info(f"Performance comparison completed: {speed_improvement:.1f}% speed improvement, {memory_savings:.1f}% memory savings")
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Failed to compare collections performance: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on Qdrant connection and collections."""
        try:
            start_time = time.time()
            
            # Test connection
            collections = await self.async_client.get_collections()
            
            # Check collections exist
            collection_names = [col.name for col in collections.collections]
            regular_exists = self.regular_collection in collection_names
            quantized_exists = self.quantized_collection in collection_names
            
            # Get collection info
            regular_info = None
            quantized_info = None
            
            if regular_exists:
                regular_info = await self.async_client.get_collection(self.regular_collection)
            
            if quantized_exists:
                quantized_info = await self.async_client.get_collection(self.quantized_collection)
            
            health_status = {
                "connection_healthy": True,
                "response_time_ms": (time.time() - start_time) * 1000,
                "collections": {
                    "regular": {
                        "exists": regular_exists,
                        "points_count": regular_info.points_count if regular_info else 0,
                        "status": regular_info.status if regular_info else "not_found"
                    },
                    "quantized": {
                        "exists": quantized_exists,
                        "points_count": quantized_info.points_count if quantized_info else 0,
                        "status": quantized_info.status if quantized_info else "not_found"
                    }
                },
                "total_collections": len(collections.collections),
                "timestamp": time.time()
            }
            
            self.connection_health = True
            self.last_health_check = time.time()
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self.connection_health = False
            
            return {
                "connection_healthy": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def get_collection_stats(self, collection_type: Union[CollectionType, MultiVectorCollectionType]) -> Dict[str, Any]:
        """Get detailed statistics for a collection."""
        try:
            if collection_type == CollectionType.REGULAR:
                collection_name = self.regular_collection
            elif collection_type == CollectionType.BINARY_QUANTIZED:
                collection_name = self.quantized_collection
            elif collection_type == MultiVectorCollectionType.MULTI_VECTOR_REGULAR:
                collection_name = self.multi_vector_regular_collection
            elif collection_type == MultiVectorCollectionType.MULTI_VECTOR_QUANTIZED:
                collection_name = self.multi_vector_quantized_collection
            else:
                raise ValueError(f"Unsupported collection type: {collection_type}")
            
            info = await self.async_client.get_collection(collection_name)
            
            # Calculate memory usage
            vector_size = info.points_count * self.settings.vector_db.vector_dimensions
            if collection_type in [CollectionType.REGULAR, MultiVectorCollectionType.MULTI_VECTOR_REGULAR]:
                memory_usage_mb = (vector_size * 4) / (1024 * 1024)  # 4 bytes per float
            else:
                memory_usage_mb = (vector_size * 1) / (1024 * 1024)  # 1 byte per quantized value
            
            return {
                "collection_name": collection_name,
                "collection_type": str(collection_type),
                "points_count": info.points_count,
                "vector_dimensions": self.settings.vector_db.vector_dimensions,
                "memory_usage_mb": memory_usage_mb,
                "status": info.status,
                "payload_indexes": getattr(info, 'payload_indexes', [])
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {}
    
    async def clear_collection(self, collection_type: CollectionType) -> bool:
        """Clear all vectors from a collection."""
        try:
            if collection_type == CollectionType.REGULAR:
                collection_name = self.regular_collection
            elif collection_type == CollectionType.BINARY_QUANTIZED:
                collection_name = self.quantized_collection
            else:
                raise ValueError(f"Unsupported collection type: {collection_type}")
            
            await self.async_client.delete(
                collection_name=collection_name,
                points_selector=rest.PointIdsList(
                    points=[]  # Empty list deletes all points
                )
            )
            
            self.logger.info(f"Cleared collection: {collection_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear collection: {e}")
            return False
    
    async def close(self) -> None:
        """Close Qdrant client connections."""
        try:
            if self._async_client:
                await self._async_client.close()
            if self._sync_client:
                self._sync_client.close()
            
            self.logger.info("Vector engine connections closed")
            
        except Exception as e:
            self.logger.error(f"Error closing vector engine: {e}")


# Global vector engine instance
_vector_engine: Optional[VectorEngine] = None


def get_vector_engine() -> VectorEngine:
    """Get the global vector engine instance."""
    global _vector_engine
    if _vector_engine is None:
        _vector_engine = VectorEngine()
    return _vector_engine



