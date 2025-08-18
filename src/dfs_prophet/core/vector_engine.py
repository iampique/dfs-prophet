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
        self.connection_health = True
        self.last_health_check = 0
        
        # Collection names
        self.regular_collection = f"{self.settings.vector_db.players_collection}_regular"
        self.quantized_collection = f"{self.settings.vector_db.players_collection}_quantized"
    
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
    
    @performance_timer('upsert_player_vector')
    async def upsert_player_vector(
        self,
        player: Player,
        vector: List[float],
        collection_type: CollectionType = CollectionType.REGULAR
    ) -> bool:
        """Upsert a player vector to the specified collection."""
        try:
            collection_name = (
                self.regular_collection if collection_type == CollectionType.REGULAR 
                else self.quantized_collection
            )
            
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
            collection_name = (
                self.regular_collection if collection_type == CollectionType.REGULAR 
                else self.quantized_collection
            )
            
            total_upserted = 0
            
            # Process in batches
            for i in range(0, len(players), batch_size):
                batch_players = players[i:i + batch_size]
                batch_vectors = vectors[i:i + batch_size]
                
                # Prepare batch points
                points = []
                for player, vector in zip(batch_players, batch_vectors):
                    payload = {
                        "player_id": player.player_id,
                        "name": player.name,
                        "position": player.position.value,
                        "team": player.team.value,
                        "season": player.base.season,
                        "week": player.base.week,
                        "timestamp": time.time()
                    }
                    
                    if player.stats:
                        payload.update({
                            "fantasy_points": player.stats.fantasy_points,
                            "passing_yards": player.stats.passing_yards,
                            "rushing_yards": player.stats.rushing_yards,
                            "receiving_yards": player.stats.receiving_yards,
                        })
                    
                    if player.dfs:
                        payload.update({
                            "salary": player.dfs.salary,
                            "projected_points": player.dfs.projected_points,
                            "ownership_percentage": player.dfs.ownership_percentage,
                        })
                    
                    points.append(rest.PointStruct(
                        id=hash(player.player_id) % (2**63),  # Convert to integer ID
                        vector=vector,
                        payload=payload
                    ))
                
                # Upsert batch
                await self.async_client.upsert(
                    collection_name=collection_name,
                    points=points
                )
                
                total_upserted += len(points)
            
            log_vector_operation(
                "BATCH_UPSERT",
                collection_name,
                total_upserted,
                time.time()
            )
            
            return total_upserted
            
        except Exception as e:
            self.logger.error(f"Failed to batch upsert vectors: {e}")
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
            collection_name = (
                self.regular_collection if collection_type == CollectionType.REGULAR 
                else self.quantized_collection
            )
            
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
            if filters:
                search_results = await self.async_client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    query_filter=search_request.filter
                )
            else:
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
                # Reconstruct Player object from payload
                player = self._reconstruct_player_from_payload(result.payload)
                
                search_result = SearchResult(
                    player=player,
                    score=result.score,
                    vector_id=result.id,
                    collection_type=collection_type
                )
                results.append(search_result)
            
            log_vector_operation(
                "SEARCH",
                collection_name,
                len(results),
                time.time()
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search vectors: {e}")
            return []
    
    def _build_filter(self, filters: Dict[str, Any]) -> rest.Filter:
        """Build Qdrant filter from dictionary."""
        conditions = []
        
        for field, value in filters.items():
            if isinstance(value, list):
                conditions.append(rest.FieldCondition(
                    key=field,
                    match=rest.MatchAny(any=value)
                ))
            else:
                conditions.append(rest.FieldCondition(
                    key=field,
                    match=rest.MatchValue(value=value)
                ))
        
        return rest.Filter(must=conditions)
    
    def _reconstruct_player_from_payload(self, payload: Dict[str, Any]) -> Player:
        """Reconstruct Player object from Qdrant payload."""
        from ..data.models import PlayerBase, PlayerStats, PlayerDFS, Position, Team
        
        # Reconstruct base
        base = PlayerBase(
            player_id=payload["player_id"],
            name=payload["name"],
            position=Position(payload["position"]),
            team=Team(payload["team"]),
            season=payload["season"],
            week=payload.get("week")
        )
        
        # Reconstruct stats if available
        stats = None
        if "fantasy_points" in payload:
            stats = PlayerStats(
                fantasy_points=payload.get("fantasy_points"),
                passing_yards=payload.get("passing_yards"),
                rushing_yards=payload.get("rushing_yards"),
                receiving_yards=payload.get("receiving_yards")
            )
        
        # Reconstruct DFS data if available
        dfs = None
        if "salary" in payload:
            dfs = PlayerDFS(
                salary=payload["salary"],
                projected_points=payload["projected_points"],
                ownership_percentage=payload.get("ownership_percentage")
            )
        
        return Player(base=base, stats=stats, dfs=dfs)
    
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
    
    async def get_collection_stats(self, collection_type: CollectionType) -> Dict[str, Any]:
        """Get detailed statistics for a collection."""
        try:
            collection_name = (
                self.regular_collection if collection_type == CollectionType.REGULAR 
                else self.quantized_collection
            )
            
            info = await self.async_client.get_collection(collection_name)
            
            # Calculate memory usage
            vector_size = info.points_count * self.settings.vector_db.vector_dimensions
            if collection_type == CollectionType.REGULAR:
                memory_usage_mb = (vector_size * 4) / (1024 * 1024)  # 4 bytes per float
            else:
                memory_usage_mb = (vector_size * 1) / (1024 * 1024)  # 1 byte per quantized value
            
            return {
                "collection_name": collection_name,
                "collection_type": collection_type.value,
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
            collection_name = (
                self.regular_collection if collection_type == CollectionType.REGULAR 
                else self.quantized_collection
            )
            
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



