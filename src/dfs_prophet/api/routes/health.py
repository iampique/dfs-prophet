"""
Health Check API for DFS Prophet.

Endpoints:
- GET /health - Basic health status
- GET /health/detailed - Detailed system status including Qdrant connection, 
  data loading status, performance metrics

Features:
- Qdrant connection verification
- Collection status and counts
- Memory usage statistics
- Performance benchmarks
- System dependencies check
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



