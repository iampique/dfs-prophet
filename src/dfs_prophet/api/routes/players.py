"""
Player Search API for DFS Prophet.

Endpoints:
- GET /players/search - Configurable search (regular vs binary)
- GET /players/search/binary - Binary quantized search (speed optimized)
- GET /players/search/regular - Regular vector search (accuracy optimized)
- GET /players/compare - Performance comparison between methods
- POST /players/batch-search - Batch search for performance testing

Features:
- Query parameter validation
- Performance timing
- Response formatting with similarity scores
- Error handling and proper HTTP status codes
- OpenAPI documentation
- Binary quantization speed showcase
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query, Body, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from ...config import get_settings
from ...utils import get_logger, performance_timer
from ...core import get_vector_engine, get_embedding_generator, CollectionType
from ...data.models import SearchRequest, SearchResponse, QuantizationComparison, Player

router = APIRouter(prefix="/players", tags=["players"])

logger = get_logger(__name__)


class PlayerSearchQuery(BaseModel):
    """Query model for player search requests."""
    
    query: str = Field(
        ..., 
        description="Search query (player name, position, team, or description)",
        example="Patrick Mahomes quarterback Kansas City Chiefs"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return"
    )
    score_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold"
    )
    position: Optional[str] = Field(
        default=None,
        description="Filter by player position (QB, RB, WR, TE, etc.)"
    )
    team: Optional[str] = Field(
        default=None,
        description="Filter by team abbreviation (KC, NE, etc.)"
    )
    season: Optional[int] = Field(
        default=None,
        ge=2020,
        le=2024,
        description="Filter by season year"
    )


class BatchSearchRequest(BaseModel):
    """Request model for batch search operations."""
    
    queries: List[str] = Field(
        ...,
        min_items=1,
        max_items=50,
        description="List of search queries to process"
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results per query"
    )
    score_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold"
    )
    collection_type: str = Field(
        default="binary",
        pattern="^(regular|binary)$",
        description="Collection type to use for search"
    )


class SearchResult(BaseModel):
    """Individual search result model."""
    
    player_id: str = Field(..., description="Unique player identifier")
    name: str = Field(..., description="Player name")
    position: str = Field(..., description="Player position")
    team: str = Field(..., description="Player team")
    season: int = Field(..., description="Season year")
    week: int = Field(..., description="Week number")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    fantasy_points: float = Field(..., description="Fantasy points")
    salary: Optional[int] = Field(None, description="DFS salary")
    projected_points: Optional[float] = Field(None, description="Projected fantasy points")
    stats: Dict[str, Any] = Field(..., description="Player statistics")


class SearchResponseModel(BaseModel):
    """Response model for search endpoints."""
    
    query: str = Field(..., description="Original search query")
    collection_type: str = Field(..., description="Collection type used")
    results: List[SearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")
    embedding_time_ms: float = Field(..., description="Embedding generation time in milliseconds")
    total_time_ms: float = Field(..., description="Total request time in milliseconds")
    timestamp: str = Field(..., description="Request timestamp")


class ComparisonResult(BaseModel):
    """Performance comparison result model."""
    
    query: str = Field(..., description="Search query used for comparison")
    regular_search: SearchResponseModel = Field(..., description="Regular search results")
    binary_search: SearchResponseModel = Field(..., description="Binary search results")
    speed_improvement: float = Field(..., description="Speed improvement percentage")
    accuracy_comparison: Dict[str, Any] = Field(..., description="Accuracy comparison metrics")
    memory_usage: Dict[str, float] = Field(..., description="Memory usage comparison")
    timestamp: str = Field(..., description="Comparison timestamp")


class BatchSearchResponse(BaseModel):
    """Response model for batch search operations."""
    
    queries: List[str] = Field(..., description="Original queries")
    results: List[SearchResponseModel] = Field(..., description="Results for each query")
    total_queries: int = Field(..., description="Total number of queries processed")
    total_time_ms: float = Field(..., description="Total batch processing time")
    avg_time_per_query_ms: float = Field(..., description="Average time per query")
    successful_queries: int = Field(..., description="Number of successful queries")
    failed_queries: int = Field(..., description="Number of failed queries")
    timestamp: str = Field(..., description="Batch processing timestamp")


@router.get("/search", summary="Configurable Player Search")
async def search_players(
    query: str = Query(..., description="Search query (player name, position, team, or description)"),
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of results"),
    score_threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimum similarity score"),
    collection_type: str = Query(default="binary", pattern="^(regular|binary)$", description="Collection type to use"),
    strategy: str = Query(default="TEXT_ONLY", pattern="^(TEXT_ONLY|HYBRID|CONTEXTUAL|STATISTICAL)$", description="Embedding strategy to use"),
    position: Optional[str] = Query(default=None, description="Filter by player position"),
    team: Optional[str] = Query(default=None, description="Filter by team abbreviation"),
    season: Optional[int] = Query(default=None, ge=2020, le=2024, description="Filter by season year")
) -> SearchResponseModel:
    """
    Search for players using configurable collection type.
    
    This endpoint allows you to choose between regular vector search (accuracy optimized)
    and binary quantized search (speed optimized). Binary quantization typically provides
    40x speed improvement with minimal accuracy loss.
    
    Args:
        query: Search query describing the player you're looking for
        limit: Maximum number of results to return (1-100)
        score_threshold: Minimum similarity score (0.0-1.0)
        collection_type: Use "regular" for accuracy or "binary" for speed
        position: Optional position filter (QB, RB, WR, TE, etc.)
        team: Optional team filter (KC, NE, etc.)
        season: Optional season filter (2020-2024)
    
    Returns:
        Search results with performance metrics and timing information.
    """
    try:
        start_time = time.time()
        
        # Validate collection type
        collection = CollectionType.REGULAR if collection_type == "regular" else CollectionType.BINARY_QUANTIZED
        
        # Generate embedding
        embedding_start = time.time()
        generator = get_embedding_generator()
        query_embedding = await generator.generate_query_embedding(query)
        embedding_time = (time.time() - embedding_start) * 1000
        
        # Build search filters
        filters = {}
        if position:
            filters["position"] = position.upper()
        if team:
            filters["team"] = team.upper()
        if season:
            filters["season"] = season
        
        # Perform search
        search_start = time.time()
        engine = get_vector_engine()
        search_results = await engine.search_vectors(
            query_embedding,
            collection,
            limit=limit,
            score_threshold=score_threshold,
            filters=filters
        )
        search_time = (time.time() - search_start) * 1000
        
        # Format results
        results = []
        for result in search_results:
            player = result.player
            results.append(SearchResult(
                player_id=player.player_id,
                name=player.name,
                position=player.position.value,
                team=player.team.value,
                season=player.base.season,
                week=player.base.week,
                similarity_score=result.score,
                fantasy_points=player.stats.fantasy_points,
                salary=player.dfs.salary,
                projected_points=player.dfs.projected_points,
                stats={
                    "passing_yards": player.stats.passing_yards,
                    "rushing_yards": player.stats.rushing_yards,
                    "receiving_yards": player.stats.receiving_yards,
                    "total_touchdowns": getattr(player.stats, 'total_touchdowns', 0),
                    "total_yards": getattr(player.stats, 'total_yards', 0)
                }
            ))
        
        total_time = (time.time() - start_time) * 1000
        
        logger.info(f"Player search completed: {len(results)} results in {total_time:.2f}ms")
        
        return SearchResponseModel(
            query=query,
            collection_type=collection_type,
            results=results,
            total_results=len(results),
            search_time_ms=round(search_time, 2),
            embedding_time_ms=round(embedding_time, 2),
            total_time_ms=round(total_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Player search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/search/binary", summary="Binary Quantized Search (Speed Optimized)")
async def search_players_binary(
    query: str = Query(..., description="Search query (player name, position, team, or description)"),
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of results"),
    score_threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimum similarity score"),
    position: Optional[str] = Query(default=None, description="Filter by player position"),
    team: Optional[str] = Query(default=None, description="Filter by team abbreviation"),
    season: Optional[int] = Query(default=None, ge=2020, le=2024, description="Filter by season year")
) -> SearchResponseModel:
    """
    Search for players using binary quantized vectors for maximum speed.
    
    This endpoint uses binary quantization which provides approximately 40x speed improvement
    compared to regular vector search with minimal accuracy loss. Ideal for real-time
    applications and high-throughput scenarios.
    
    Args:
        query: Search query describing the player you're looking for
        limit: Maximum number of results to return (1-100)
        score_threshold: Minimum similarity score (0.0-1.0)
        position: Optional position filter (QB, RB, WR, TE, etc.)
        team: Optional team filter (KC, NE, etc.)
        season: Optional season filter (2020-2024)
    
    Returns:
        Search results with performance metrics highlighting speed optimization.
    """
    return await search_players(
        query=query,
        limit=limit,
        score_threshold=score_threshold,
        collection_type="binary",
        position=position,
        team=team,
        season=season
    )


@router.get("/search/regular", summary="Regular Vector Search (Accuracy Optimized)")
async def search_players_regular(
    query: str = Query(..., description="Search query (player name, position, team, or description)"),
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of results"),
    score_threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimum similarity score"),
    position: Optional[str] = Query(default=None, description="Filter by player position"),
    team: Optional[str] = Query(default=None, description="Filter by team abbreviation"),
    season: Optional[int] = Query(default=None, ge=2020, le=2024, description="Filter by season year")
) -> SearchResponseModel:
    """
    Search for players using regular vectors for maximum accuracy.
    
    This endpoint uses full-precision vectors which provide the highest accuracy
    but slower search times compared to binary quantization. Ideal for applications
    where accuracy is more important than speed.
    
    Args:
        query: Search query describing the player you're looking for
        limit: Maximum number of results to return (1-100)
        score_threshold: Minimum similarity score (0.0-1.0)
        position: Optional position filter (QB, RB, WR, TE, etc.)
        team: Optional team filter (KC, NE, etc.)
        season: Optional season filter (2020-2024)
    
    Returns:
        Search results with performance metrics highlighting accuracy optimization.
    """
    return await search_players(
        query=query,
        limit=limit,
        score_threshold=score_threshold,
        collection_type="regular",
        position=position,
        team=team,
        season=season
    )


@router.get("/compare", summary="Performance Comparison")
async def compare_search_methods(
    query: str = Query(..., description="Search query for comparison"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum number of results"),
    score_threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimum similarity score"),
    position: Optional[str] = Query(default=None, description="Filter by player position"),
    team: Optional[str] = Query(default=None, description="Filter by team abbreviation"),
    season: Optional[int] = Query(default=None, ge=2020, le=2024, description="Filter by season year")
) -> ComparisonResult:
    """
    Compare performance between regular and binary quantized search methods.
    
    This endpoint runs the same search query using both regular vectors and binary
    quantized vectors, then compares their performance metrics including speed,
    accuracy, and memory usage. This demonstrates the 40x speed improvement
    achievable with binary quantization.
    
    Args:
        query: Search query to use for comparison
        limit: Maximum number of results to return (1-50)
        score_threshold: Minimum similarity score (0.0-1.0)
        position: Optional position filter (QB, RB, WR, TE, etc.)
        team: Optional team filter (KC, NE, etc.)
        season: Optional season filter (2020-2024)
    
    Returns:
        Detailed comparison of both search methods with performance metrics.
    """
    try:
        start_time = time.time()
        
        # Run both searches concurrently
        regular_task = search_players(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
            collection_type="regular",
            position=position,
            team=team,
            season=season
        )
        
        binary_task = search_players(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
            collection_type="binary",
            position=position,
            team=team,
            season=season
        )
        
        regular_result, binary_result = await asyncio.gather(regular_task, binary_task)
        
        # Calculate speed improvement
        speed_improvement = 0
        if regular_result.search_time_ms > 0:
            speed_improvement = ((regular_result.search_time_ms - binary_result.search_time_ms) / regular_result.search_time_ms) * 100
        
        # Calculate accuracy comparison
        accuracy_comparison = _calculate_accuracy_comparison(regular_result, binary_result)
        
        # Get memory usage comparison
        engine = get_vector_engine()
        regular_stats = await engine.get_collection_stats(CollectionType.REGULAR)
        binary_stats = await engine.get_collection_stats(CollectionType.BINARY_QUANTIZED)
        
        memory_usage = {
            "regular_mb": regular_stats.get("memory_usage_mb", 0),
            "binary_mb": binary_stats.get("memory_usage_mb", 0),
            "compression_ratio": binary_stats.get("memory_usage_mb", 0) / max(regular_stats.get("memory_usage_mb", 1), 1)
        }
        
        total_time = (time.time() - start_time) * 1000
        
        logger.info(f"Performance comparison completed in {total_time:.2f}ms")
        
        return ComparisonResult(
            query=query,
            regular_search=regular_result,
            binary_search=binary_result,
            speed_improvement=round(speed_improvement, 2),
            accuracy_comparison=accuracy_comparison,
            memory_usage=memory_usage,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Performance comparison failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comparison failed: {str(e)}"
        )


@router.post("/batch-search", summary="Batch Search for Performance Testing")
async def batch_search_players(
    request: BatchSearchRequest = Body(..., description="Batch search request")
) -> BatchSearchResponse:
    """
    Perform batch search operations for performance testing.
    
    This endpoint processes multiple search queries in parallel, making it ideal
    for performance testing and benchmarking. You can specify whether to use
    regular or binary quantized vectors for the entire batch.
    
    Args:
        request: Batch search request containing queries and parameters
    
    Returns:
        Batch search results with aggregate performance metrics.
    """
    try:
        start_time = time.time()
        
        # Validate collection type
        collection = CollectionType.REGULAR if request.collection_type == "regular" else CollectionType.BINARY_QUANTIZED
        
        # Process queries concurrently
        tasks = []
        for query in request.queries:
            task = _process_single_search(
                query=query,
                limit=request.limit,
                score_threshold=request.score_threshold,
                collection=collection
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_count += 1
                logger.error(f"Query {i} failed: {result}")
            else:
                successful_results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / len(request.queries) if request.queries else 0
        
        logger.info(f"Batch search completed: {len(successful_results)}/{len(request.queries)} successful in {total_time:.2f}ms")
        
        return BatchSearchResponse(
            queries=request.queries,
            results=successful_results,
            total_queries=len(request.queries),
            total_time_ms=round(total_time, 2),
            avg_time_per_query_ms=round(avg_time, 2),
            successful_queries=len(successful_results),
            failed_queries=failed_count,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Batch search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch search failed: {str(e)}"
        )


async def _process_single_search(
    query: str,
    limit: int,
    score_threshold: float,
    collection: CollectionType
) -> SearchResponseModel:
    """Process a single search query for batch operations."""
    try:
        start_time = time.time()
        
        # Generate embedding
        embedding_start = time.time()
        generator = get_embedding_generator()
        query_embedding = await generator.generate_query_embedding(query)
        embedding_time = (time.time() - embedding_start) * 1000
        
        # Perform search
        search_start = time.time()
        engine = get_vector_engine()
        search_results = await engine.search_vectors(
            query_embedding,
            collection,
            limit=limit,
            score_threshold=score_threshold
        )
        search_time = (time.time() - search_start) * 1000
        
        # Format results
        results = []
        for result in search_results:
            player = result.player
            results.append(SearchResult(
                player_id=player.player_id,
                name=player.name,
                position=player.position.value,
                team=player.team.value,
                season=player.base.season,
                week=player.base.week,
                similarity_score=result.score,
                fantasy_points=player.stats.fantasy_points,
                salary=player.dfs.salary,
                projected_points=player.dfs.projected_points,
                stats={
                    "passing_yards": player.stats.passing_yards,
                    "rushing_yards": player.stats.rushing_yards,
                    "receiving_yards": player.stats.receiving_yards,
                    "total_touchdowns": getattr(player.stats, 'total_touchdowns', 0),
                    "total_yards": getattr(player.stats, 'total_yards', 0)
                }
            ))
        
        total_time = (time.time() - start_time) * 1000
        
        return SearchResponseModel(
            query=query,
            collection_type="regular" if collection == CollectionType.REGULAR else "binary",
            results=results,
            total_results=len(results),
            search_time_ms=round(search_time, 2),
            embedding_time_ms=round(embedding_time, 2),
            total_time_ms=round(total_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise Exception(f"Search failed for query '{query}': {str(e)}")


def _calculate_accuracy_comparison(regular_result: SearchResponseModel, binary_result: SearchResponseModel) -> Dict[str, Any]:
    """Calculate accuracy comparison between regular and binary search results."""
    try:
        # Compare top results overlap
        regular_ids = {r.player_id for r in regular_result.results[:5]}
        binary_ids = {r.player_id for r in binary_result.results[:5]}
        
        overlap = len(regular_ids.intersection(binary_ids))
        overlap_percentage = (overlap / 5) * 100 if regular_ids else 0
        
        # Compare average similarity scores
        regular_avg_score = sum(r.similarity_score for r in regular_result.results) / len(regular_result.results) if regular_result.results else 0
        binary_avg_score = sum(r.similarity_score for r in binary_result.results) / len(binary_result.results) if binary_result.results else 0
        
        score_difference = regular_avg_score - binary_avg_score
        
        return {
            "top_5_overlap": overlap,
            "overlap_percentage": round(overlap_percentage, 2),
            "regular_avg_score": round(regular_avg_score, 4),
            "binary_avg_score": round(binary_avg_score, 4),
            "score_difference": round(score_difference, 4),
            "accuracy_preserved": overlap_percentage >= 80  # Consider good if 80%+ overlap
        }
    except Exception as e:
        return {
            "error": str(e),
            "accuracy_preserved": False
        }


