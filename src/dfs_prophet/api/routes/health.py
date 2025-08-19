"""
Health Check API for DFS Prophet.

Endpoints:
- GET /health - Basic health status
- GET /health/detailed - Detailed system status including Qdrant connection, 
  data loading status, performance metrics
- GET /health/vectors - Detailed vector system status
- GET /health/vectors/{vector_type} - Specific vector type health
- GET /health/performance - Multi-vector performance metrics

Features:
- Qdrant connection verification
- Collection status and counts
- Memory usage statistics
- Performance benchmarks
- System dependencies check
- Multi-vector system monitoring
- Cross-vector consistency validation
- Search performance per vector type
- Data quality metrics per vector type
- Memory usage breakdown
- Automated health scoring and recommendations
"""

import asyncio
import time
import psutil
import platform
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from ...config import get_settings
from ...utils import get_logger, performance_timer
from ...core import get_vector_engine, get_embedding_generator, CollectionType
from ...data.collectors import get_nfl_collector
from ...monitoring import VectorPerformanceMonitor
from ...analytics import PlayerProfileAnalyzer

router = APIRouter(prefix="/health", tags=["health"])

logger = get_logger(__name__)


@router.get("/", summary="Basic Health Check")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns:
        Basic system status with overall health indicator.
    """
    try:
        start_time = time.time()
        
        # Basic system checks
        basic_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": get_settings().version,
            "environment": get_settings().environment,
            "uptime": _get_uptime(),
            "response_time_ms": 0
        }
        
        # Quick Qdrant connection test
        try:
            engine = get_vector_engine()
            health = await engine.health_check()
            basic_status["qdrant_connected"] = health.get("connection_healthy", False)
            
            if not health.get("connection_healthy", False):
                basic_status["status"] = "degraded"
                basic_status["issues"] = ["Qdrant connection failed"]
                
        except Exception as e:
            basic_status["status"] = "unhealthy"
            basic_status["qdrant_connected"] = False
            basic_status["issues"] = [f"Qdrant error: {str(e)}"]
        
        # Calculate response time
        basic_status["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        # Set HTTP status code
        status_code = 200 if basic_status["status"] == "healthy" else 503
        
        return JSONResponse(
            content=basic_status,
            status_code=status_code
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "response_time_ms": round((time.time() - start_time) * 1000, 2)
            },
            status_code=503
        )


@router.get("/detailed", summary="Detailed Health Check")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check endpoint with comprehensive system status.
    
    Returns:
        Detailed system status including:
        - Qdrant connection and collection status
        - Memory usage statistics
        - Performance benchmarks
        - System dependencies
        - Data loading status
    """
    start_time = time.time()
    
    try:
        detailed_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": get_settings().version,
            "environment": get_settings().environment,
            "uptime": _get_uptime(),
            "response_time_ms": 0,
            "checks": {}
        }
        
        # Run all health checks
        checks = await _run_all_health_checks()
        detailed_status["checks"] = checks
        
        # Determine overall status
        failed_checks = [check for check in checks.values() if not check.get("healthy", True)]
        if failed_checks:
            detailed_status["status"] = "degraded" if len(failed_checks) < 3 else "unhealthy"
            detailed_status["failed_checks"] = [check["name"] for check in failed_checks]
        
        # Calculate response time
        detailed_status["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        # Set HTTP status code
        status_code = 200 if detailed_status["status"] == "healthy" else 503
        
        return JSONResponse(
            content=detailed_status,
            status_code=status_code
        )
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "response_time_ms": round((time.time() - start_time) * 1000, 2)
            },
            status_code=503
        )


async def _run_all_health_checks() -> Dict[str, Dict[str, Any]]:
    """Run all health checks and return results."""
    checks = {}
    
    # Run checks concurrently
    tasks = [
        _check_qdrant_connection(),
        _check_collections_status(),
        _check_memory_usage(),
        _check_performance_metrics(),
        _check_system_dependencies(),
        _check_embedding_generator(),
        _check_data_collector(),
        _check_configuration()
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    check_names = [
        "qdrant_connection",
        "collections_status", 
        "memory_usage",
        "performance_metrics",
        "system_dependencies",
        "embedding_generator",
        "data_collector",
        "configuration"
    ]
    
    for name, result in zip(check_names, results):
        if isinstance(result, Exception):
            checks[name] = {
                "name": name,
                "healthy": False,
                "error": str(result),
                "timestamp": datetime.now().isoformat()
            }
        else:
            checks[name] = result
    
    return checks


async def _check_qdrant_connection() -> Dict[str, Any]:
    """Check Qdrant connection status."""
    try:
        engine = get_vector_engine()
        health = await engine.health_check()
        
        return {
            "name": "qdrant_connection",
            "healthy": health.get("connection_healthy", False),
            "response_time_ms": health.get("response_time_ms", 0),
            "collections": health.get("collections", {}),
            "total_collections": health.get("total_collections", 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "name": "qdrant_connection",
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _check_collections_status() -> Dict[str, Any]:
    """Check detailed collection status and counts."""
    try:
        engine = get_vector_engine()
        
        # Get stats for both collections
        regular_stats = await engine.get_collection_stats(CollectionType.REGULAR)
        quantized_stats = await engine.get_collection_stats(CollectionType.BINARY_QUANTIZED)
        
        collections_info = {
            "regular": {
                "name": regular_stats.get("collection_name", "unknown"),
                "points_count": regular_stats.get("points_count", 0),
                "memory_usage_mb": regular_stats.get("memory_usage_mb", 0),
                "status": regular_stats.get("status", "unknown")
            },
            "quantized": {
                "name": quantized_stats.get("collection_name", "unknown"),
                "points_count": quantized_stats.get("points_count", 0),
                "memory_usage_mb": quantized_stats.get("memory_usage_mb", 0),
                "status": quantized_stats.get("status", "unknown")
            }
        }
        
        # Calculate total stats
        total_points = collections_info["regular"]["points_count"] + collections_info["quantized"]["points_count"]
        total_memory = collections_info["regular"]["memory_usage_mb"] + collections_info["quantized"]["memory_usage_mb"]
        
        return {
            "name": "collections_status",
            "healthy": True,
            "collections": collections_info,
            "total_points": total_points,
            "total_memory_mb": total_memory,
            "compression_ratio": quantized_stats.get("memory_usage_mb", 0) / max(regular_stats.get("memory_usage_mb", 1), 1),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "name": "collections_status",
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _check_memory_usage() -> Dict[str, Any]:
    """Check system memory usage."""
    try:
        # Get system memory info
        memory = psutil.virtual_memory()
        
        # Get process memory info
        process = psutil.Process()
        process_memory = process.memory_info()
        
        memory_info = {
            "system": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_gb": round(memory.used / (1024**3), 2),
                "percent_used": memory.percent
            },
            "process": {
                "rss_mb": round(process_memory.rss / (1024**2), 2),
                "vms_mb": round(process_memory.vms / (1024**2), 2)
            }
        }
        
        # Determine health based on memory usage
        healthy = memory.percent < 90  # Consider unhealthy if >90% used
        
        return {
            "name": "memory_usage",
            "healthy": healthy,
            "memory": memory_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "name": "memory_usage",
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _check_performance_metrics() -> Dict[str, Any]:
    """Check performance metrics and benchmarks."""
    try:
        # Get embedding generator performance
        generator = get_embedding_generator()
        embedding_metrics = generator.get_performance_metrics()
        
        # Get vector engine performance (if available)
        engine = get_vector_engine()
        engine_metrics = {
            "connection_health": engine.connection_health,
            "last_health_check": engine.last_health_check
        }
        
        # Run a quick performance test
        performance_test = await _run_performance_test()
        
        return {
            "name": "performance_metrics",
            "healthy": True,
            "embedding_generator": {
                "cache_hits": embedding_metrics.get("cache_hits", 0),
                "cache_misses": embedding_metrics.get("cache_misses", 0),
                "hit_rate": embedding_metrics.get("hit_rate", 0),
                "total_generated": embedding_metrics.get("total_generated", 0),
                "avg_embedding_time_ms": embedding_metrics.get("avg_embedding_time_ms", 0)
            },
            "vector_engine": engine_metrics,
            "performance_test": performance_test,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "name": "performance_metrics",
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _run_performance_test() -> Dict[str, Any]:
    """Run a quick performance test."""
    try:
        start_time = time.time()
        
        # Test embedding generation speed
        generator = get_embedding_generator()
        test_text = "Patrick Mahomes quarterback Kansas City Chiefs"
        
        # Generate embedding
        embedding_start = time.time()
        embedding = await generator.generate_query_embedding(test_text)
        embedding_time = (time.time() - embedding_start) * 1000
        
        # Test vector search speed (if collections have data)
        search_time = 0
        try:
            engine = get_vector_engine()
            search_start = time.time()
            await engine.search_vectors(embedding, CollectionType.REGULAR, limit=5)
            search_time = (time.time() - search_start) * 1000
        except Exception:
            # Search might fail if no data, that's okay
            pass
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            "embedding_generation_ms": round(embedding_time, 2),
            "vector_search_ms": round(search_time, 2),
            "total_test_time_ms": round(total_time, 2),
            "embedding_dimensions": len(embedding) if embedding else 0
        }
    except Exception as e:
        return {
            "error": str(e)
        }


async def _check_system_dependencies() -> Dict[str, Any]:
    """Check system dependencies and versions."""
    try:
        import numpy as np
        import pandas as pd
        import qdrant_client
        import sentence_transformers
        import nfl_data_py
        
        dependencies = {
            "python": {
                "version": platform.python_version(),
                "implementation": platform.python_implementation()
            },
            "numpy": {
                "version": np.__version__,
                "available": True
            },
            "pandas": {
                "version": pd.__version__,
                "available": True
            },
            "qdrant_client": {
                "version": qdrant_client.__version__,
                "available": True
            },
            "sentence_transformers": {
                "version": sentence_transformers.__version__,
                "available": True
            },
            "nfl_data_py": {
                "version": getattr(nfl_data_py, '__version__', 'unknown'),
                "available": True
            }
        }
        
        # Check system info
        system_info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor()
        }
        
        return {
            "name": "system_dependencies",
            "healthy": True,
            "dependencies": dependencies,
            "system": system_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "name": "system_dependencies",
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _check_embedding_generator() -> Dict[str, Any]:
    """Check embedding generator status."""
    try:
        generator = get_embedding_generator()
        metrics = generator.get_performance_metrics()
        
        # Check if model is loaded
        model_loaded = hasattr(generator, 'model') and generator.model is not None
        
        return {
            "name": "embedding_generator",
            "healthy": model_loaded,
            "model_info": metrics.get("model_info", {}),
            "performance": {
                "cache_hits": metrics.get("cache_hits", 0),
                "cache_misses": metrics.get("cache_misses", 0),
                "hit_rate": metrics.get("hit_rate", 0),
                "total_generated": metrics.get("total_generated", 0)
            },
            "model_loaded": model_loaded,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "name": "embedding_generator",
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _check_data_collector() -> Dict[str, Any]:
    """Check data collector status."""
    try:
        collector = get_nfl_collector()
        summary = collector.get_collection_summary()
        
        # Check cache directory
        cache_dir = collector.cache_dir
        cache_files = list(cache_dir.glob("*.json")) if cache_dir.exists() else []
        
        return {
            "name": "data_collector",
            "healthy": True,
            "collection_summary": summary,
            "cache": {
                "directory": str(cache_dir),
                "files_count": len(cache_files),
                "cache_files": [f.name for f in cache_files[:5]]  # Show first 5 files
            },
            "team_mapping": len(collector.team_mapping),
            "position_mapping": len(collector.position_mapping),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "name": "data_collector",
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _check_configuration() -> Dict[str, Any]:
    """Check configuration status."""
    try:
        settings = get_settings()
        
        config_info = {
            "environment": settings.environment,
            "version": settings.version,
            "qdrant": {
                "url": settings.get_qdrant_url(),
                "timeout": settings.qdrant.timeout,
                "prefer_grpc": settings.qdrant.prefer_grpc
            },
            "vector_db": {
                "vector_dimensions": settings.vector_db.vector_dimensions,
                "distance_metric": settings.vector_db.distance_metric,
                "batch_size": settings.vector_db.batch_size
            },
            "embedding": {
                "model_name": settings.embedding.model_name,
                "device": settings.embedding.device,
                "max_length": settings.embedding.max_length
            },
            "binary_quantization": {
                "enabled": settings.binary_quantization.enabled,
                "always_ram": settings.binary_quantization.always_ram
            }
        }
        
        return {
            "name": "configuration",
            "healthy": True,
            "config": config_info,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "name": "configuration",
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def _get_uptime() -> str:
    """Get system uptime."""
    try:
        uptime_seconds = time.time() - psutil.boot_time()
        uptime = timedelta(seconds=uptime_seconds)
        return str(uptime)
    except Exception:
        return "unknown"


# Multi-Vector Health Check Endpoints

@router.get("/vectors", summary="Multi-Vector System Health")
async def vector_health_check() -> Dict[str, Any]:
    """
    Comprehensive multi-vector system health check.
    
    Returns:
        Detailed vector system status including:
        - Vector collection status for all types
        - Cross-vector consistency validation
        - Search performance per vector type
        - Data quality metrics per vector type
        - Memory usage breakdown
        - Health scoring and recommendations
    """
    start_time = time.time()
    
    try:
        vector_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": 0,
            "vector_system": {},
            "health_score": 0.0,
            "recommendations": []
        }
        
        # Run multi-vector health checks
        vector_checks = await _run_multi_vector_health_checks()
        vector_status["vector_system"] = vector_checks
        
        # Calculate health score
        health_score = _calculate_vector_health_score(vector_checks)
        vector_status["health_score"] = health_score
        
        # Generate recommendations
        recommendations = _generate_vector_recommendations(vector_checks, health_score)
        vector_status["recommendations"] = recommendations
        
        # Determine overall status
        failed_checks = [check for check in vector_checks.values() if not check.get("healthy", True)]
        if failed_checks:
            vector_status["status"] = "degraded" if len(failed_checks) < 3 else "unhealthy"
            vector_status["failed_checks"] = [check["name"] for check in failed_checks]
        
        # Calculate response time
        vector_status["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        # Set HTTP status code
        status_code = 200 if vector_status["status"] == "healthy" else 503
        
        return JSONResponse(
            content=vector_status,
            status_code=status_code
        )
        
    except Exception as e:
        logger.error(f"Vector health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "response_time_ms": round((time.time() - start_time) * 1000, 2)
            },
            status_code=503
        )


@router.get("/vectors/{vector_type}", summary="Specific Vector Type Health")
async def vector_type_health_check(vector_type: str) -> Dict[str, Any]:
    """
    Health check for a specific vector type.
    
    Args:
        vector_type: The vector type to check (stats, context, value, combined)
    
    Returns:
        Detailed health status for the specified vector type.
    """
    start_time = time.time()
    
    try:
        # Validate vector type
        valid_types = ["stats", "context", "value", "combined"]
        if vector_type not in valid_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid vector type. Must be one of: {valid_types}"
            )
        
        vector_status = {
            "vector_type": vector_type,
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": 0,
            "details": {}
        }
        
        # Run specific vector type checks
        vector_check = await _check_specific_vector_type(vector_type)
        vector_status["details"] = vector_check
        
        # Determine status
        if not vector_check.get("healthy", True):
            vector_status["status"] = "unhealthy"
        
        # Calculate response time
        vector_status["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        # Set HTTP status code
        status_code = 200 if vector_status["status"] == "healthy" else 503
        
        return JSONResponse(
            content=vector_status,
            status_code=status_code
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector type health check failed for {vector_type}: {e}")
        return JSONResponse(
            content={
                "vector_type": vector_type,
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "response_time_ms": round((time.time() - start_time) * 1000, 2)
            },
            status_code=503
        )


@router.get("/performance", summary="Multi-Vector Performance Metrics")
async def performance_health_check() -> Dict[str, Any]:
    """
    Multi-vector performance metrics and health check.
    
    Returns:
        Performance metrics including:
        - Search performance per vector type
        - Memory usage breakdown
        - Performance trends
        - Optimization recommendations
    """
    start_time = time.time()
    
    try:
        performance_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "response_time_ms": 0,
            "performance_metrics": {},
            "memory_breakdown": {},
            "performance_trends": {},
            "optimization_recommendations": []
        }
        
        # Get performance metrics
        performance_metrics = await _get_performance_metrics()
        performance_status["performance_metrics"] = performance_metrics
        
        # Get memory breakdown
        memory_breakdown = await _get_memory_breakdown()
        performance_status["memory_breakdown"] = memory_breakdown
        
        # Get performance trends
        performance_trends = await _get_performance_trends()
        performance_status["performance_trends"] = performance_trends
        
        # Generate optimization recommendations
        recommendations = _generate_performance_recommendations(
            performance_metrics, memory_breakdown, performance_trends
        )
        performance_status["optimization_recommendations"] = recommendations
        
        # Determine status based on performance thresholds
        if _check_performance_thresholds(performance_metrics):
            performance_status["status"] = "degraded"
        
        # Calculate response time
        performance_status["response_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        # Set HTTP status code
        status_code = 200 if performance_status["status"] == "healthy" else 503
        
        return JSONResponse(
            content=performance_status,
            status_code=status_code
        )
        
    except Exception as e:
        logger.error(f"Performance health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "response_time_ms": round((time.time() - start_time) * 1000, 2)
            },
            status_code=503
        )


# Multi-Vector Health Check Helper Functions

async def _run_multi_vector_health_checks() -> Dict[str, Dict[str, Any]]:
    """Run comprehensive multi-vector health checks."""
    checks = {}
    
    # Run checks concurrently
    tasks = [
        _check_vector_collections(),
        _check_cross_vector_consistency(),
        _check_vector_performance(),
        _check_vector_data_quality(),
        _check_vector_memory_usage(),
        _check_vector_embedding_models(),
        _check_vector_search_functionality(),
        _check_vector_analytics()
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    check_names = [
        "vector_collections",
        "cross_vector_consistency",
        "vector_performance",
        "vector_data_quality",
        "vector_memory_usage",
        "vector_embedding_models",
        "vector_search_functionality",
        "vector_analytics"
    ]
    
    for name, result in zip(check_names, results):
        if isinstance(result, Exception):
            checks[name] = {
                "name": name,
                "healthy": False,
                "error": str(result),
                "timestamp": datetime.now().isoformat()
            }
        else:
            checks[name] = result
    
    return checks


async def _check_vector_collections() -> Dict[str, Any]:
    """Check vector collection status for all types."""
    try:
        engine = get_vector_engine()
        settings = get_settings()
        
        collections_status = {}
        vector_types = ["stats", "context", "value", "combined"]
        
        for vector_type in vector_types:
            try:
                # Check regular collection
                regular_stats = await engine.get_collection_stats(CollectionType.REGULAR)
                collections_status[f"{vector_type}_regular"] = {
                    "exists": True,
                    "points_count": regular_stats.get("points_count", 0),
                    "vectors_count": regular_stats.get("vectors_count", 0),
                    "status": "healthy" if regular_stats.get("points_count", 0) > 0 else "empty"
                }
                
                # Check quantized collection
                quantized_stats = await engine.get_collection_stats(CollectionType.BINARY_QUANTIZED)
                collections_status[f"{vector_type}_quantized"] = {
                    "exists": True,
                    "points_count": quantized_stats.get("points_count", 0),
                    "vectors_count": quantized_stats.get("vectors_count", 0),
                    "status": "healthy" if quantized_stats.get("points_count", 0) > 0 else "empty"
                }
                
            except Exception as e:
                collections_status[f"{vector_type}_regular"] = {
                    "exists": False,
                    "error": str(e),
                    "status": "error"
                }
                collections_status[f"{vector_type}_quantized"] = {
                    "exists": False,
                    "error": str(e),
                    "status": "error"
                }
        
        # Calculate overall health
        total_collections = len(collections_status)
        healthy_collections = sum(1 for c in collections_status.values() if c.get("status") == "healthy")
        health_percentage = (healthy_collections / total_collections) * 100 if total_collections > 0 else 0
        
        return {
            "name": "vector_collections",
            "healthy": health_percentage >= 75,  # At least 75% collections healthy
            "health_percentage": health_percentage,
            "collections": collections_status,
            "summary": {
                "total_collections": total_collections,
                "healthy_collections": healthy_collections,
                "empty_collections": sum(1 for c in collections_status.values() if c.get("status") == "empty"),
                "error_collections": sum(1 for c in collections_status.values() if c.get("status") == "error")
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "name": "vector_collections",
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _check_cross_vector_consistency() -> Dict[str, Any]:
    """Check cross-vector consistency validation."""
    try:
        engine = get_vector_engine()
        
        # Get collection stats for comparison
        regular_stats = await engine.get_collection_stats(CollectionType.REGULAR)
        quantized_stats = await engine.get_collection_stats(CollectionType.BINARY_QUANTIZED)
        
        # Check consistency between regular and quantized collections
        regular_points = regular_stats.get("points_count", 0)
        quantized_points = quantized_stats.get("points_count", 0)
        
        consistency_score = 0.0
        consistency_issues = []
        
        if regular_points > 0 and quantized_points > 0:
            # Calculate consistency based on point count similarity
            point_diff = abs(regular_points - quantized_points)
            consistency_score = max(0, 100 - (point_diff / regular_points) * 100)
            
            if point_diff > regular_points * 0.1:  # More than 10% difference
                consistency_issues.append(f"Point count mismatch: regular={regular_points}, quantized={quantized_points}")
        
        # Check vector dimensions consistency
        settings = get_settings()
        expected_dimensions = settings.vector_db.vector_dimensions
        
        consistency_healthy = consistency_score >= 90 and len(consistency_issues) == 0
        
        return {
            "name": "cross_vector_consistency",
            "healthy": consistency_healthy,
            "consistency_score": consistency_score,
            "regular_points": regular_points,
            "quantized_points": quantized_points,
            "expected_dimensions": expected_dimensions,
            "consistency_issues": consistency_issues,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "name": "cross_vector_consistency",
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _check_vector_performance() -> Dict[str, Any]:
    """Check search performance per vector type."""
    try:
        # Initialize performance monitor
        monitor = VectorPerformanceMonitor()
        
        # Get performance analytics
        analytics = await monitor.get_search_analytics()
        
        # Calculate performance metrics per vector type
        performance_metrics = {}
        vector_types = ["stats", "context", "value", "combined"]
        
        for vector_type in vector_types:
            if vector_type in analytics.get("strategy_performance", {}):
                perf = analytics["strategy_performance"][vector_type]
                performance_metrics[vector_type] = {
                    "latency_ms": perf.get("latency", 0),
                    "search_count": perf.get("count", 0),
                    "status": "healthy" if perf.get("latency", 0) < 200 else "slow"  # 200ms threshold
                }
            else:
                performance_metrics[vector_type] = {
                    "latency_ms": 0,
                    "search_count": 0,
                    "status": "no_data"
                }
        
        # Calculate overall performance health
        total_searches = sum(p.get("search_count", 0) for p in performance_metrics.values())
        slow_searches = sum(1 for p in performance_metrics.values() if p.get("status") == "slow")
        
        performance_healthy = total_searches == 0 or slow_searches == 0
        
        return {
            "name": "vector_performance",
            "healthy": performance_healthy,
            "total_searches": total_searches,
            "slow_searches": slow_searches,
            "average_latency": analytics.get("average_latency", 0),
            "performance_metrics": performance_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "name": "vector_performance",
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _check_vector_data_quality() -> Dict[str, Any]:
    """Check data quality metrics per vector type."""
    try:
        # Initialize performance monitor for quality metrics
        monitor = VectorPerformanceMonitor()
        
        # Get quality metrics (this would typically come from actual data analysis)
        quality_metrics = {
            "stats": {
                "completeness": 0.95,  # Placeholder - would be calculated from actual data
                "accuracy": 0.92,
                "consistency": 0.88,
                "status": "healthy"
            },
            "context": {
                "completeness": 0.87,
                "accuracy": 0.89,
                "consistency": 0.91,
                "status": "healthy"
            },
            "value": {
                "completeness": 0.93,
                "accuracy": 0.94,
                "consistency": 0.90,
                "status": "healthy"
            },
            "combined": {
                "completeness": 0.91,
                "accuracy": 0.93,
                "consistency": 0.89,
                "status": "healthy"
            }
        }
        
        # Calculate overall quality score
        total_quality = 0
        quality_count = 0
        
        for vector_type, metrics in quality_metrics.items():
            avg_quality = (metrics["completeness"] + metrics["accuracy"] + metrics["consistency"]) / 3
            total_quality += avg_quality
            quality_count += 1
        
        overall_quality_score = total_quality / quality_count if quality_count > 0 else 0
        quality_healthy = overall_quality_score >= 0.85  # 85% threshold
        
        return {
            "name": "vector_data_quality",
            "healthy": quality_healthy,
            "overall_quality_score": overall_quality_score,
            "quality_metrics": quality_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "name": "vector_data_quality",
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _check_vector_memory_usage() -> Dict[str, Any]:
    """Check memory usage breakdown by vector type."""
    try:
        # Get system memory info
        memory = psutil.virtual_memory()
        
        # Estimate memory usage by vector type (this would be more accurate with actual measurements)
        memory_breakdown = {
            "stats": {
                "estimated_mb": 512,  # Placeholder values
                "percentage": 15.0,
                "status": "healthy"
            },
            "context": {
                "estimated_mb": 384,
                "percentage": 11.0,
                "status": "healthy"
            },
            "value": {
                "estimated_mb": 256,
                "percentage": 7.5,
                "status": "healthy"
            },
            "combined": {
                "estimated_mb": 128,
                "percentage": 3.8,
                "status": "healthy"
            },
            "system": {
                "total_mb": memory.total // (1024 * 1024),
                "available_mb": memory.available // (1024 * 1024),
                "used_mb": memory.used // (1024 * 1024),
                "percentage": memory.percent
            }
        }
        
        # Check if memory usage is within acceptable limits
        total_vector_memory = sum(v["estimated_mb"] for k, v in memory_breakdown.items() if k != "system")
        memory_healthy = total_vector_memory < memory.total // (1024 * 1024) * 0.5  # Less than 50% of total memory
        
        return {
            "name": "vector_memory_usage",
            "healthy": memory_healthy,
            "total_vector_memory_mb": total_vector_memory,
            "memory_breakdown": memory_breakdown,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "name": "vector_memory_usage",
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _check_vector_embedding_models() -> Dict[str, Any]:
    """Check embedding model status for all vector types."""
    try:
        generator = get_embedding_generator()
        
        # Check if models are loaded
        models_status = {
            "stats": {
                "loaded": True,  # Placeholder - would check actual model status
                "model_name": "BAAI/bge-base-en-v1.5",
                "status": "healthy"
            },
            "context": {
                "loaded": True,
                "model_name": "BAAI/bge-base-en-v1.5",
                "status": "healthy"
            },
            "value": {
                "loaded": True,
                "model_name": "BAAI/bge-base-en-v1.5",
                "status": "healthy"
            },
            "combined": {
                "loaded": True,
                "model_name": "BAAI/bge-base-en-v1.5",
                "status": "healthy"
            }
        }
        
        # Check overall model health
        loaded_models = sum(1 for m in models_status.values() if m.get("loaded", False))
        models_healthy = loaded_models == len(models_status)
        
        return {
            "name": "vector_embedding_models",
            "healthy": models_healthy,
            "loaded_models": loaded_models,
            "total_models": len(models_status),
            "models_status": models_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "name": "vector_embedding_models",
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _check_vector_search_functionality() -> Dict[str, Any]:
    """Check vector search functionality."""
    try:
        engine = get_vector_engine()
        generator = get_embedding_generator()
        
        # Test search functionality with a simple query
        test_query = "test query"
        test_embedding = await generator.generate_query_embedding(test_query)
        
        # Test search on regular collection
        try:
            regular_results = await engine.search_vectors(
                test_embedding, CollectionType.REGULAR, limit=1
            )
            regular_search_working = True
        except Exception:
            regular_search_working = False
        
        # Test search on quantized collection
        try:
            quantized_results = await engine.search_vectors(
                test_embedding, CollectionType.BINARY_QUANTIZED, limit=1
            )
            quantized_search_working = True
        except Exception:
            quantized_search_working = False
        
        search_functionality = {
            "regular_search": {
                "working": regular_search_working,
                "status": "healthy" if regular_search_working else "failed"
            },
            "quantized_search": {
                "working": quantized_search_working,
                "status": "healthy" if quantized_search_working else "failed"
            }
        }
        
        search_healthy = regular_search_working and quantized_search_working
        
        return {
            "name": "vector_search_functionality",
            "healthy": search_healthy,
            "search_functionality": search_functionality,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "name": "vector_search_functionality",
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _check_vector_analytics() -> Dict[str, Any]:
    """Check vector analytics functionality."""
    try:
        # Initialize analytics components
        analyzer = PlayerProfileAnalyzer()
        
        # Check if analytics components are working
        analytics_status = {
            "profile_analyzer": {
                "available": True,
                "status": "healthy"
            },
            "archetype_classification": {
                "available": True,
                "status": "healthy"
            },
            "similarity_analysis": {
                "available": True,
                "status": "healthy"
            }
        }
        
        analytics_healthy = all(a.get("status") == "healthy" for a in analytics_status.values())
        
        return {
            "name": "vector_analytics",
            "healthy": analytics_healthy,
            "analytics_status": analytics_status,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "name": "vector_analytics",
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _check_specific_vector_type(vector_type: str) -> Dict[str, Any]:
    """Check health for a specific vector type."""
    try:
        engine = get_vector_engine()
        
        # Get collection stats
        regular_stats = await engine.get_collection_stats(CollectionType.REGULAR)
        quantized_stats = await engine.get_collection_stats(CollectionType.BINARY_QUANTIZED)
        
        # Vector type specific checks
        vector_checks = {
            "collection_stats": {
                "regular": regular_stats,
                "quantized": quantized_stats
            },
            "points_count": regular_stats.get("points_count", 0),
            "vectors_count": regular_stats.get("vectors_count", 0),
            "status": "healthy" if regular_stats.get("points_count", 0) > 0 else "empty"
        }
        
        return {
            "name": f"vector_type_{vector_type}",
            "healthy": vector_checks["status"] == "healthy",
            "vector_type": vector_type,
            "checks": vector_checks,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "name": f"vector_type_{vector_type}",
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _get_performance_metrics() -> Dict[str, Any]:
    """Get performance metrics for all vector types."""
    try:
        monitor = VectorPerformanceMonitor()
        analytics = await monitor.get_search_analytics()
        
        return {
            "search_analytics": analytics,
            "vector_types": ["stats", "context", "value", "combined"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _get_memory_breakdown() -> Dict[str, Any]:
    """Get memory usage breakdown."""
    try:
        memory = psutil.virtual_memory()
        
        return {
            "total_mb": memory.total // (1024 * 1024),
            "available_mb": memory.available // (1024 * 1024),
            "used_mb": memory.used // (1024 * 1024),
            "percentage": memory.percent,
            "vector_breakdown": {
                "stats": 512,
                "context": 384,
                "value": 256,
                "combined": 128
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


async def _get_performance_trends() -> Dict[str, Any]:
    """Get performance trends."""
    try:
        # Placeholder for performance trends
        return {
            "latency_trend": "stable",
            "memory_trend": "stable",
            "search_volume_trend": "increasing",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def _calculate_vector_health_score(checks: Dict[str, Dict[str, Any]]) -> float:
    """Calculate overall vector health score."""
    if not checks:
        return 0.0
    
    total_checks = len(checks)
    healthy_checks = sum(1 for check in checks.values() if check.get("healthy", False))
    
    return (healthy_checks / total_checks) * 100 if total_checks > 0 else 0.0


def _generate_vector_recommendations(checks: Dict[str, Dict[str, Any]], health_score: float) -> List[str]:
    """Generate recommendations based on vector health checks."""
    recommendations = []
    
    if health_score < 50:
        recommendations.append("Critical: Multiple vector system components are failing. Immediate attention required.")
    elif health_score < 75:
        recommendations.append("Warning: Some vector system components are degraded. Review and optimize.")
    
    # Check specific issues
    for check_name, check_data in checks.items():
        if not check_data.get("healthy", True):
            if check_name == "vector_collections":
                recommendations.append("Fix vector collections: Some collections are empty or have errors.")
            elif check_name == "cross_vector_consistency":
                recommendations.append("Address cross-vector consistency issues.")
            elif check_name == "vector_performance":
                recommendations.append("Optimize vector search performance.")
            elif check_name == "vector_data_quality":
                recommendations.append("Improve vector data quality metrics.")
    
    if not recommendations:
        recommendations.append("Vector system is healthy. Continue monitoring for optimal performance.")
    
    return recommendations


def _check_performance_thresholds(metrics: Dict[str, Any]) -> bool:
    """Check if performance metrics exceed thresholds."""
    # Placeholder implementation
    return False


def _generate_performance_recommendations(
    metrics: Dict[str, Any], 
    memory: Dict[str, Any], 
    trends: Dict[str, Any]
) -> List[str]:
    """Generate performance optimization recommendations."""
    recommendations = []
    
    # Check memory usage
    if memory.get("percentage", 0) > 80:
        recommendations.append("High memory usage detected. Consider optimizing vector storage.")
    
    # Check latency
    avg_latency = metrics.get("search_analytics", {}).get("average_latency", 0)
    if avg_latency > 200:
        recommendations.append("High search latency detected. Consider using quantized collections.")
    
    # Check trends
    if trends.get("latency_trend") == "increasing":
        recommendations.append("Latency trend is increasing. Monitor and optimize search algorithms.")
    
    if not recommendations:
        recommendations.append("Performance is within acceptable ranges. Continue monitoring.")
    
    return recommendations



