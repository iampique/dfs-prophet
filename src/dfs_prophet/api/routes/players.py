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
from ...core import get_vector_engine, get_embedding_generator, CollectionType, MultiVectorCollectionType
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


# Multi-Vector Search Models
class MultiVectorSearchRequest(BaseModel):
    """Request model for multi-vector search operations."""
    
    query: str = Field(
        ..., 
        description="Search query for multi-vector analysis",
        example="elite quarterback with high fantasy points"
    )
    vector_types: List[str] = Field(
        default=["stats", "context", "value"],
        description="Vector types to include in search",
        example=["stats", "context", "value"]
    )
    weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Custom weights for vector fusion (must sum to ~1.0)",
        example={"stats": 0.4, "context": 0.3, "value": 0.3}
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
    fusion_strategy: str = Field(
        default="weighted_average",
        pattern="^(weighted_average|max_score|min_score|product)$",
        description="Strategy for combining multiple vectors"
    )


class VectorContribution(BaseModel):
    """Model for vector contribution breakdown."""
    
    vector_type: str = Field(..., description="Vector type (stats, context, value)")
    weight: float = Field(..., description="Weight used in fusion")
    score: float = Field(..., description="Individual vector score")
    contribution: float = Field(..., description="Contribution to final score")
    explanation: str = Field(..., description="Explanation of why this vector matched")


class MultiVectorSearchResult(BaseModel):
    """Enhanced search result with multi-vector analysis."""
    
    player_id: str = Field(..., description="Unique player identifier")
    name: str = Field(..., description="Player name")
    position: str = Field(..., description="Player position")
    team: str = Field(..., description="Player team")
    season: int = Field(..., description="Season year")
    week: int = Field(..., description="Week number")
    final_score: float = Field(..., description="Final fusion similarity score")
    vector_contributions: List[VectorContribution] = Field(..., description="Breakdown by vector type")
    primary_vector: str = Field(..., description="Vector type with highest contribution")
    match_explanation: str = Field(..., description="Explanation of why player matched")
    fantasy_points: float = Field(..., description="Fantasy points")
    salary: Optional[int] = Field(None, description="DFS salary")
    projected_points: Optional[float] = Field(None, description="Projected fantasy points")
    stats: Dict[str, Any] = Field(..., description="Player statistics")


class MultiVectorSearchResponse(BaseModel):
    """Response model for multi-vector search endpoints."""
    
    query: str = Field(..., description="Original search query")
    vector_types: List[str] = Field(..., description="Vector types used")
    fusion_strategy: str = Field(..., description="Fusion strategy used")
    weights: Dict[str, float] = Field(..., description="Weights used for fusion")
    results: List[MultiVectorSearchResult] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")
    embedding_time_ms: float = Field(..., description="Embedding generation time in milliseconds")
    fusion_time_ms: float = Field(..., description="Vector fusion time in milliseconds")
    total_time_ms: float = Field(..., description="Total request time in milliseconds")
    timestamp: str = Field(..., description="Request timestamp")


class PlayerAnalysis(BaseModel):
    """Complete multi-vector player analysis."""
    
    player_id: str = Field(..., description="Player identifier")
    name: str = Field(..., description="Player name")
    position: str = Field(..., description="Player position")
    team: str = Field(..., description="Player team")
    season: int = Field(..., description="Season year")
    week: int = Field(..., description="Week number")
    
    # Vector-specific analysis
    stats_analysis: Dict[str, Any] = Field(..., description="Statistical vector analysis")
    context_analysis: Dict[str, Any] = Field(..., description="Contextual vector analysis")
    value_analysis: Dict[str, Any] = Field(..., description="Value vector analysis")
    combined_analysis: Dict[str, Any] = Field(..., description="Combined vector analysis")
    
    # Similarity patterns
    similar_players: Dict[str, List[str]] = Field(..., description="Similar players by vector type")
    vector_strengths: Dict[str, float] = Field(..., description="Strength scores by vector type")
    vector_weaknesses: Dict[str, List[str]] = Field(..., description="Weaknesses by vector type")
    
    # Recommendations
    recommendations: List[str] = Field(..., description="Strategic recommendations")
    risk_factors: List[str] = Field(..., description="Risk factors to consider")
    timestamp: str = Field(..., description="Analysis timestamp")


class VectorComparisonRequest(BaseModel):
    """Request model for vector comparison."""
    
    player_ids: List[str] = Field(
        ...,
        min_items=2,
        max_items=5,
        description="Player IDs to compare"
    )
    vector_types: List[str] = Field(
        default=["stats", "context", "value"],
        description="Vector types to include in comparison"
    )
    comparison_metric: str = Field(
        default="similarity",
        pattern="^(similarity|performance|value|consistency)$",
        description="Metric to use for comparison"
    )


class VectorComparisonResult(BaseModel):
    """Result model for vector comparison."""
    
    comparison_metric: str = Field(..., description="Metric used for comparison")
    vector_types: List[str] = Field(..., description="Vector types compared")
    player_comparisons: Dict[str, Dict[str, float]] = Field(..., description="Comparison scores by player")
    vector_contributions: Dict[str, Dict[str, float]] = Field(..., description="Vector contributions by player")
    rankings: Dict[str, List[str]] = Field(..., description="Player rankings by vector type")
    insights: List[str] = Field(..., description="Key insights from comparison")
    timestamp: str = Field(..., description="Comparison timestamp")


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


# Multi-Vector Search Endpoints
@router.get("/search/stats", summary="Statistical Vector Search")
async def search_by_stats(
    query: str = Query(..., description="Search query for statistical similarity"),
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of results"),
    score_threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimum similarity score"),
    position: Optional[str] = Query(default=None, description="Filter by player position"),
    team: Optional[str] = Query(default=None, description="Filter by team abbreviation"),
    season: Optional[int] = Query(default=None, ge=2020, le=2024, description="Filter by season year")
) -> MultiVectorSearchResponse:
    """
    Search for players using statistical vector similarity only.
    
    This endpoint focuses on statistical performance patterns like fantasy points,
    yards, touchdowns, and efficiency metrics. Ideal for finding players with
    similar statistical profiles.
    
    Args:
        query: Search query describing statistical patterns
        limit: Maximum number of results (1-100)
        score_threshold: Minimum similarity score (0.0-1.0)
        position: Optional position filter
        team: Optional team filter
        season: Optional season filter
    
    Returns:
        Multi-vector search results with statistical focus.
    """
    return await _multi_vector_search(
        query=query,
        vector_types=["stats"],
        limit=limit,
        score_threshold=score_threshold,
        position=position,
        team=team,
        season=season
    )


@router.get("/search/context", summary="Contextual Vector Search")
async def search_by_context(
    query: str = Query(..., description="Search query for contextual similarity"),
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of results"),
    score_threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimum similarity score"),
    position: Optional[str] = Query(default=None, description="Filter by player position"),
    team: Optional[str] = Query(default=None, description="Filter by team abbreviation"),
    season: Optional[int] = Query(default=None, ge=2020, le=2024, description="Filter by season year")
) -> MultiVectorSearchResponse:
    """
    Search for players using contextual vector similarity only.
    
    This endpoint focuses on game context factors like weather, opponent strength,
    venue, time of day, and situational factors. Ideal for finding players who
    perform well in similar game contexts.
    
    Args:
        query: Search query describing contextual factors
        limit: Maximum number of results (1-100)
        score_threshold: Minimum similarity score (0.0-1.0)
        position: Optional position filter
        team: Optional team filter
        season: Optional season filter
    
    Returns:
        Multi-vector search results with contextual focus.
    """
    return await _multi_vector_search(
        query=query,
        vector_types=["context"],
        limit=limit,
        score_threshold=score_threshold,
        position=position,
        team=team,
        season=season
    )


@router.get("/search/value", summary="Value Vector Search")
async def search_by_value(
    query: str = Query(..., description="Search query for DFS value patterns"),
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of results"),
    score_threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimum similarity score"),
    position: Optional[str] = Query(default=None, description="Filter by player position"),
    team: Optional[str] = Query(default=None, description="Filter by team abbreviation"),
    season: Optional[int] = Query(default=None, ge=2020, le=2024, description="Filter by season year")
) -> MultiVectorSearchResponse:
    """
    Search for players using value vector similarity only.
    
    This endpoint focuses on DFS value patterns like salary efficiency, ownership
    trends, ROI, and market dynamics. Ideal for finding undervalued players or
    identifying value plays.
    
    Args:
        query: Search query describing value patterns
        limit: Maximum number of results (1-100)
        score_threshold: Minimum similarity score (0.0-1.0)
        position: Optional position filter
        team: Optional team filter
        season: Optional season filter
    
    Returns:
        Multi-vector search results with value focus.
    """
    return await _multi_vector_search(
        query=query,
        vector_types=["value"],
        limit=limit,
        score_threshold=score_threshold,
        position=position,
        team=team,
        season=season
    )


@router.get("/search/fusion", summary="Multi-Vector Fusion Search")
async def search_by_fusion(
    query: str = Query(..., description="Search query for multi-vector analysis"),
    vector_types: str = Query(default="stats,context,value", description="Comma-separated vector types"),
    weights: Optional[str] = Query(default=None, description="Comma-separated weights (e.g., 0.4,0.3,0.3)"),
    fusion_strategy: str = Query(default="weighted_average", description="Fusion strategy"),
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of results"),
    score_threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimum similarity score"),
    position: Optional[str] = Query(default=None, description="Filter by player position"),
    team: Optional[str] = Query(default=None, description="Filter by team abbreviation"),
    season: Optional[int] = Query(default=None, ge=2020, le=2024, description="Filter by season year")
) -> MultiVectorSearchResponse:
    """
    Search for players using multi-vector fusion with weighted combination.
    
    This endpoint combines multiple vector types (statistical, contextual, value)
    using configurable weights and fusion strategies. Provides the most comprehensive
    player similarity analysis.
    
    Args:
        query: Search query for multi-vector analysis
        vector_types: Comma-separated list of vector types (stats,context,value)
        weights: Optional comma-separated weights for fusion
        fusion_strategy: Strategy for combining vectors
        limit: Maximum number of results (1-100)
        score_threshold: Minimum similarity score (0.0-1.0)
        position: Optional position filter
        team: Optional team filter
        season: Optional season filter
    
    Returns:
        Multi-vector search results with fusion analysis.
    """
    # Parse vector types and weights
    vt_list = [vt.strip() for vt in vector_types.split(",")]
    weight_dict = None
    if weights:
        weight_list = [float(w.strip()) for w in weights.split(",")]
        if len(weight_list) == len(vt_list):
            weight_dict = dict(zip(vt_list, weight_list))
    
    return await _multi_vector_search(
        query=query,
        vector_types=vt_list,
        weights=weight_dict,
        fusion_strategy=fusion_strategy,
        limit=limit,
        score_threshold=score_threshold,
        position=position,
        team=team,
        season=season
    )


@router.get("/analyze/{player_id}", summary="Complete Multi-Vector Player Analysis")
async def analyze_player(
    player_id: str,
    vector_types: str = Query(default="stats,context,value", description="Comma-separated vector types to analyze")
) -> PlayerAnalysis:
    """
    Perform complete multi-vector analysis of a specific player.
    
    This endpoint provides comprehensive analysis across all vector types,
    including similarity patterns, strengths, weaknesses, and strategic
    recommendations for DFS optimization.
    
    Args:
        player_id: Unique player identifier
        vector_types: Comma-separated list of vector types to analyze
    
    Returns:
        Complete multi-vector player analysis with recommendations.
    """
    try:
        start_time = time.time()
        
        # Parse vector types
        vt_list = [vt.strip() for vt in vector_types.split(",")]
        
        # Get settings for configuration
        settings = get_settings()
        
        # Generate analysis for each vector type
        analysis = {
            "player_id": player_id,
            "name": "Player Name",  # Would be fetched from database
            "position": "QB",       # Would be fetched from database
            "team": "KC",           # Would be fetched from database
            "season": 2024,         # Would be fetched from database
            "week": 1,              # Would be fetched from database
            "stats_analysis": _generate_stats_analysis(player_id),
            "context_analysis": _generate_context_analysis(player_id),
            "value_analysis": _generate_value_analysis(player_id),
            "combined_analysis": _generate_combined_analysis(player_id),
            "similar_players": _find_similar_players(player_id, vt_list),
            "vector_strengths": _calculate_vector_strengths(player_id, vt_list),
            "vector_weaknesses": _identify_weaknesses(player_id, vt_list),
            "recommendations": _generate_recommendations(player_id, vt_list),
            "risk_factors": _identify_risk_factors(player_id, vt_list),
            "timestamp": datetime.now().isoformat()
        }
        
        return PlayerAnalysis(**analysis)
        
    except Exception as e:
        logger.error(f"Player analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Player analysis failed: {str(e)}"
        )


@router.post("/compare/vectors", summary="Multi-Vector Player Comparison")
async def compare_players_vectors(
    request: VectorComparisonRequest = Body(..., description="Vector comparison request")
) -> VectorComparisonResult:
    """
    Compare multiple players across different vector types.
    
    This endpoint allows you to compare players using different vector types
    and metrics, providing insights into their relative strengths and
    similarities across statistical, contextual, and value dimensions.
    
    Args:
        request: Vector comparison request with player IDs and parameters
    
    Returns:
        Multi-vector comparison results with rankings and insights.
    """
    try:
        start_time = time.time()
        
        # Perform comparison for each vector type
        comparisons = {}
        contributions = {}
        rankings = {}
        
        for vector_type in request.vector_types:
            vector_comparison = await _compare_players_by_vector(
                request.player_ids, vector_type, request.comparison_metric
            )
            comparisons[vector_type] = vector_comparison["scores"]
            contributions[vector_type] = vector_comparison["contributions"]
            rankings[vector_type] = vector_comparison["rankings"]
        
        # Generate insights
        insights = _generate_comparison_insights(comparisons, rankings)
        
        return VectorComparisonResult(
            comparison_metric=request.comparison_metric,
            vector_types=request.vector_types,
            player_comparisons=comparisons,
            vector_contributions=contributions,
            rankings=rankings,
            insights=insights,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Vector comparison failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector comparison failed: {str(e)}"
        )


# Enhanced existing endpoint
@router.get("/search", summary="Enhanced Configurable Player Search")
async def search_players_enhanced(
    query: str = Query(..., description="Search query (player name, position, team, or description)"),
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of results"),
    score_threshold: float = Query(default=0.5, ge=0.0, le=1.0, description="Minimum similarity score"),
    collection_type: str = Query(default="binary", pattern="^(regular|binary)$", description="Collection type to use"),
    strategy: str = Query(default="TEXT_ONLY", pattern="^(TEXT_ONLY|HYBRID|CONTEXTUAL|STATISTICAL)$", description="Embedding strategy to use"),
    vector_types: Optional[str] = Query(default=None, description="Comma-separated vector types for multi-vector search"),
    position: Optional[str] = Query(default=None, description="Filter by player position"),
    team: Optional[str] = Query(default=None, description="Filter by team abbreviation"),
    season: Optional[int] = Query(default=None, ge=2020, le=2024, description="Filter by season year")
) -> Union[SearchResponseModel, MultiVectorSearchResponse]:
    """
    Enhanced configurable player search with multi-vector support.
    
    This endpoint now supports both traditional single-vector search and
    multi-vector search. When vector_types is specified, it performs
    multi-vector analysis; otherwise, it uses the traditional approach.
    
    Args:
        query: Search query describing the player you're looking for
        limit: Maximum number of results to return (1-100)
        score_threshold: Minimum similarity score (0.0-1.0)
        collection_type: Use "regular" for accuracy or "binary" for speed
        strategy: Embedding strategy to use
        vector_types: Optional comma-separated vector types for multi-vector search
        position: Optional position filter (QB, RB, WR, TE, etc.)
        team: Optional team filter (KC, NE, etc.)
        season: Optional season filter (2020-2024)
    
    Returns:
        Search results with performance metrics and timing information.
    """
    # Check if multi-vector search is requested
    if vector_types:
        vt_list = [vt.strip() for vt in vector_types.split(",")]
        return await _multi_vector_search(
            query=query,
            vector_types=vt_list,
            limit=limit,
            score_threshold=score_threshold,
            position=position,
            team=team,
            season=season
        )
    else:
        # Use traditional single-vector search
        return await search_players(
            query=query,
            limit=limit,
            score_threshold=score_threshold,
            collection_type=collection_type,
            strategy=strategy,
            position=position,
            team=team,
            season=season
        )


# Helper functions for multi-vector search
async def _multi_vector_search(
    query: str,
    vector_types: List[str],
    weights: Optional[Dict[str, float]] = None,
    fusion_strategy: str = "weighted_average",
    limit: int = 10,
    score_threshold: float = 0.5,
    position: Optional[str] = None,
    team: Optional[str] = None,
    season: Optional[int] = None
) -> MultiVectorSearchResponse:
    """Perform multi-vector search with fusion."""
    try:
        start_time = time.time()
        
        # Get settings for configuration
        settings = get_settings()
        
        # Use default weights if not provided
        if weights is None:
            weights = settings.get_vector_weights()
        
        # Generate embeddings for each vector type
        embedding_start = time.time()
        generator = get_embedding_generator()
        embeddings = {}
        
        for vector_type in vector_types:
            # Generate vector-type-specific embeddings
            if vector_type == "stats":
                embeddings[vector_type] = await generator.generate_stats_query_embedding(query)
            elif vector_type == "context":
                embeddings[vector_type] = await generator.generate_context_query_embedding(query)
            elif vector_type == "value":
                embeddings[vector_type] = await generator.generate_value_query_embedding(query)
            else:
                embeddings[vector_type] = await generator.generate_query_embedding(query)
        
        embedding_time = (time.time() - embedding_start) * 1000
        
        # Perform search for each vector type
        search_start = time.time()
        engine = get_vector_engine()
        all_results = {}
        
        for vector_type in vector_types:
            # Use multi-vector search for vector-type-specific results
            if len(vector_types) == 1:
                # Single vector type search - use regular collection for now
                results = await engine.search_vectors(
                    embeddings[vector_type],
                    CollectionType.REGULAR,
                    limit=limit * 2,  # Get more results for fusion
                    score_threshold=score_threshold
                )
            else:
                # Multi-vector search - use multi-vector collection
                # For now, use regular search since multi-vector search requires different approach
                results = await engine.search_vectors(
                    embeddings[vector_type],
                    CollectionType.REGULAR,
                    limit=limit * 2,  # Get more results for fusion
                    score_threshold=score_threshold
                )
            all_results[vector_type] = results
        
        search_time = (time.time() - search_start) * 1000
        
        # Fuse results
        fusion_start = time.time()
        fused_results = _fuse_search_results(
            all_results, weights, fusion_strategy, limit
        )
        fusion_time = (time.time() - fusion_start) * 1000
        
        # Format results
        formatted_results = []
        for result in fused_results:
            vector_contributions = _calculate_vector_contributions(
                result, all_results, weights
            )
            
            # Extract player data from the result
            player = result.player
            formatted_result = MultiVectorSearchResult(
                player_id=player.player_id,
                name=player.name,
                position=player.position.value,
                team=player.team.value,
                season=player.base.season,
                week=player.base.week,
                final_score=result.score,
                vector_contributions=vector_contributions,
                primary_vector=_get_primary_vector(vector_contributions),
                match_explanation=_generate_match_explanation(result, vector_contributions),
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
            )
            formatted_results.append(formatted_result)
        
        return MultiVectorSearchResponse(
            query=query,
            vector_types=vector_types,
            fusion_strategy=fusion_strategy,
            weights=weights,
            results=formatted_results,
            total_results=len(formatted_results),
            search_time_ms=round(search_time, 2),
            embedding_time_ms=round(embedding_time, 2),
            fusion_time_ms=round(fusion_time, 2),
            total_time_ms=round((time.time() - start_time) * 1000, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Multi-vector search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Multi-vector search failed: {str(e)}"
        )


def _fuse_search_results(
    all_results: Dict[str, List],
    weights: Dict[str, float],
    fusion_strategy: str,
    limit: int
) -> List:
    """Fuse search results from multiple vector types."""
    # Simple fusion implementation - in real implementation, use proper fusion
    fused = {}
    
    for vector_type, results in all_results.items():
        weight = weights.get(vector_type, 0.0)
        for result in results:
            player_id = result.player.player_id
            if player_id not in fused:
                fused[player_id] = result
                fused[player_id].score = result.score * weight
            else:
                if fusion_strategy == "weighted_average":
                    fused[player_id].score += result.score * weight
                elif fusion_strategy == "max_score":
                    fused[player_id].score = max(fused[player_id].score, result.score * weight)
                elif fusion_strategy == "min_score":
                    fused[player_id].score = min(fused[player_id].score, result.score * weight)
                elif fusion_strategy == "product":
                    fused[player_id].score *= result.score * weight
    
    # Sort by fused score and return top results
    sorted_results = sorted(fused.values(), key=lambda x: x.score, reverse=True)
    return sorted_results[:limit]


def _calculate_vector_contributions(
    result,
    all_results: Dict[str, List],
    weights: Dict[str, float]
) -> List[VectorContribution]:
    """Calculate vector contributions for a result."""
    contributions = []
    
    for vector_type, results in all_results.items():
        # Find this player in the vector type results
        player_result = next((r for r in results if r.player.player_id == result.player.player_id), None)
        if player_result:
            weight = weights.get(vector_type, 0.0)
            contribution = player_result.score * weight
            explanation = _generate_vector_explanation(vector_type, player_result)
            
            contributions.append(VectorContribution(
                vector_type=vector_type,
                weight=weight,
                score=player_result.score,
                contribution=contribution,
                explanation=explanation
            ))
    
    return contributions


def _get_primary_vector(contributions: List[VectorContribution]) -> str:
    """Get the primary vector type with highest contribution."""
    if not contributions:
        return "unknown"
    return max(contributions, key=lambda x: x.contribution).vector_type


def _generate_match_explanation(result, contributions: List[VectorContribution]) -> str:
    """Generate explanation of why the player matched."""
    primary = _get_primary_vector(contributions)
    explanations = [c.explanation for c in contributions if c.contribution > 0.1]
    
    if explanations:
        return f"Matched primarily on {primary} vector: {'; '.join(explanations[:2])}"
    else:
        return f"Matched on {primary} vector with score {result.score:.3f}"


def _generate_vector_explanation(vector_type: str, result) -> str:
    """Generate explanation for vector type match."""
    player = result.player
    if vector_type == "stats":
        return f"Statistical similarity: {player.stats.fantasy_points:.1f} fantasy points"
    elif vector_type == "context":
        return f"Contextual similarity: {player.team.value} team context"
    elif vector_type == "value":
        return f"Value similarity: ${player.dfs.salary} salary efficiency"
    else:
        return f"{vector_type} similarity: score {result.score:.3f}"


# Helper functions for player analysis
def _generate_stats_analysis(player_id: str) -> Dict[str, Any]:
    """Generate statistical analysis for a player."""
    return {
        "fantasy_points": 25.5,
        "efficiency_metrics": {"yards_per_attempt": 8.2, "touchdown_rate": 0.15},
        "consistency_score": 0.75,
        "trend_analysis": "Improving over last 3 weeks"
    }


def _generate_context_analysis(player_id: str) -> Dict[str, Any]:
    """Generate contextual analysis for a player."""
    return {
        "weather_impact": 0.05,
        "opponent_strength": "Above average",
        "venue_factors": "Home game advantage",
        "time_analysis": "Primetime performance boost"
    }


def _generate_value_analysis(player_id: str) -> Dict[str, Any]:
    """Generate value analysis for a player."""
    return {
        "salary_efficiency": 0.85,
        "ownership_trends": "Increasing",
        "roi_potential": "High",
        "market_position": "Undervalued"
    }


def _generate_combined_analysis(player_id: str) -> Dict[str, Any]:
    """Generate combined analysis for a player."""
    return {
        "overall_score": 0.78,
        "risk_reward_ratio": 1.2,
        "recommendation": "Strong play",
        "confidence": 0.85
    }


def _find_similar_players(player_id: str, vector_types: List[str]) -> Dict[str, List[str]]:
    """Find similar players by vector type."""
    return {
        "stats": ["player_123", "player_456"],
        "context": ["player_789", "player_012"],
        "value": ["player_345", "player_678"]
    }


def _calculate_vector_strengths(player_id: str, vector_types: List[str]) -> Dict[str, float]:
    """Calculate vector strengths for a player."""
    return {
        "stats": 0.85,
        "context": 0.72,
        "value": 0.68
    }


def _identify_weaknesses(player_id: str, vector_types: List[str]) -> Dict[str, List[str]]:
    """Identify weaknesses by vector type."""
    return {
        "stats": ["Inconsistent fantasy points"],
        "context": ["Poor weather performance"],
        "value": ["High ownership risk"]
    }


def _generate_recommendations(player_id: str, vector_types: List[str]) -> List[str]:
    """Generate strategic recommendations."""
    return [
        "Strong statistical profile suggests high floor",
        "Contextual factors favor performance",
        "Value metrics indicate good ROI potential"
    ]


def _identify_risk_factors(player_id: str, vector_types: List[str]) -> List[str]:
    """Identify risk factors."""
    return [
        "Recent injury concerns",
        "Tough defensive matchup",
        "High ownership could limit upside"
    ]


async def _compare_players_by_vector(
    player_ids: List[str],
    vector_type: str,
    metric: str
) -> Dict[str, Any]:
    """Compare players by specific vector type."""
    # Mock implementation - in real implementation, perform actual comparison
    scores = {player_id: 0.75 for player_id in player_ids}
    contributions = {player_id: {vector_type: 0.75} for player_id in player_ids}
    rankings = [player_ids[0], player_ids[1]] if len(player_ids) >= 2 else player_ids
    
    return {
        "scores": scores,
        "contributions": contributions,
        "rankings": rankings
    }


def _generate_comparison_insights(
    comparisons: Dict[str, Dict[str, float]],
    rankings: Dict[str, List[str]]
) -> List[str]:
    """Generate insights from player comparison."""
    insights = []
    
    for vector_type, scores in comparisons.items():
        top_player = max(scores.items(), key=lambda x: x[1])
        insights.append(f"Top {vector_type} performer: {top_player[0]} ({top_player[1]:.3f})")
    
    return insights


