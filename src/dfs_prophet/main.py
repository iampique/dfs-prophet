"""
Main FastAPI application for DFS Prophet.

This module provides the main FastAPI application with:
- CORS middleware configuration
- Startup event handlers for Qdrant initialization
- Shutdown event handlers for cleanup
- Route registration
- Global exception handling
- Performance monitoring middleware
- OpenAPI configuration with proper metadata
"""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi

from .config import get_settings
from .utils import get_logger
from .utils.logger import setup_logging
from .api.routes import health_router, players_router
from .core import get_vector_engine, get_embedding_generator

# Setup logging
setup_logging()
logger = get_logger(__name__)


def _validate_environment() -> None:
    """Validate critical environment and configuration before starting.

    Raises RuntimeError with helpful messages when validation fails.
    """
    settings = get_settings()

    # Basic configuration sanity checks
    if settings.vector_db.vector_dimensions <= 0:
        raise RuntimeError("VECTOR_DB_VECTOR_DIMENSIONS must be > 0")

    if not settings.embedding.model_name or not str(settings.embedding.model_name).strip():
        raise RuntimeError("EMBEDDING_MODEL_MODEL_NAME must be set")

    # Qdrant URL format check
    q_url = settings.get_qdrant_url()
    if not (q_url.startswith("http://") or q_url.startswith("https://")):
        raise RuntimeError("QDRANT_URL must start with http:// or https://")

    # Optional: advise when API key is missing in non-local setups
    if "localhost" not in q_url and not settings.qdrant.api_key:
        logger.warning("QDRANT_API_KEY is not set; required for Qdrant Cloud.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    Handles:
    - Qdrant connection initialization
    - Embedding model loading
    - Graceful shutdown and cleanup
    """
    # Startup
    logger.info("🚀 Starting DFS Prophet application...")
    
    try:
        # Initialize Qdrant connection and collections
        logger.info("📊 Initializing Qdrant vector database...")
        _validate_environment()
        engine = get_vector_engine()
        await engine.initialize_collections()
        
        # Test Qdrant health
        health = await engine.health_check()
        if health.get("connection_healthy", False):
            logger.info("✅ Qdrant connection established successfully")
        else:
            logger.warning("⚠️ Qdrant connection issues detected")
        
        # Initialize embedding generator (lazy loading)
        logger.info("🧠 Initializing embedding generator...")
        generator = get_embedding_generator()
        # The model will be loaded on first use
        
        logger.info("✅ DFS Prophet application started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down DFS Prophet application...")
    
    try:
        # Close Qdrant connections
        engine = get_vector_engine()
        await engine.close()
        logger.info("✅ Qdrant connections closed")
        
        # Clear embedding generator cache
        generator = get_embedding_generator()
        generator.clear_cache()
        logger.info("✅ Embedding cache cleared")
        
        logger.info("✅ DFS Prophet application shutdown complete")
        
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance.
    """
    settings = get_settings()
    
    # Create FastAPI app with lifespan
    app = FastAPI(
        title="DFS Prophet API",
        description="AI-Powered DFS Lineup Optimization using Qdrant Binary Quantization",
        version=settings.version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.app.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    
    # Add performance monitoring middleware
    @app.middleware("http")
    async def performance_middleware(request: Request, call_next):
        """Middleware to track request performance and logging."""
        start_time = time.time()
        
        # Log request
        logger.info(f"📥 {request.method} {request.url.path} - Client: {request.client.host if request.client else 'unknown'}")
        
        try:
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            
            # Log response
            logger.info(f"📤 {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.3f}s")
            
            return response
            
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(f"❌ {request.method} {request.url.path} - Error: {e} - Time: {process_time:.3f}s")
            raise
    
    # Register routes
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(players_router, prefix="/api/v1")
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Global exception handler for unhandled errors."""
        logger.error(f"Unhandled exception in {request.method} {request.url.path}: {exc}")
        
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "timestamp": time.time(),
                "path": str(request.url.path)
            }
        )
    
    # Custom OpenAPI schema
    def custom_openapi():
        """Custom OpenAPI schema with additional metadata."""
        if app.openapi_schema:
            return app.openapi_schema
        
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
        
        # Add additional metadata
        openapi_schema["info"]["contact"] = {
            "name": "DFS Prophet API",
            "url": "https://github.com/your-repo/dfs-prophet"
        }
        
        openapi_schema["info"]["license"] = {
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT"
        }
        
        # Add server information
        openapi_schema["servers"] = [
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            },
            {
                "url": "https://api.dfsprophet.com",
                "description": "Production server"
            }
        ]
        
        # Add tags metadata
        openapi_schema["tags"] = [
            {
                "name": "health",
                "description": "Health check and system monitoring endpoints"
            },
            {
                "name": "players",
                "description": "Player search and similarity matching endpoints"
            }
        ]
        
        app.openapi_schema = openapi_schema
        return app.openapi_schema
    
    app.openapi = custom_openapi
    
    # Root endpoint
    @app.get("/", summary="API Overview")
    async def root():
        """API overview and available endpoints."""
        return {
            "message": "DFS Prophet API - AI-Powered DFS Lineup Optimization",
            "version": settings.version,
            "environment": settings.environment,
            "endpoints": {
                "health": {
                    "GET /api/v1/health": "Basic health check",
                    "GET /api/v1/health/detailed": "Detailed system status"
                },
                "players": {
                    "GET /api/v1/players/search": "Configurable player search",
                    "GET /api/v1/players/search/binary": "Binary quantized search (40x faster)",
                    "GET /api/v1/players/search/regular": "Regular vector search (max accuracy)",
                    "GET /api/v1/players/compare": "Performance comparison",
                    "POST /api/v1/players/batch-search": "Batch search for testing"
                }
            },
            "documentation": {
                "swagger": "/docs",
                "redoc": "/redoc",
                "openapi": "/openapi.json"
            },
            "features": [
                "AI-powered player similarity search",
                "Binary quantization for 40x speed improvement",
                "Real-time fantasy sports data integration",
                "Comprehensive health monitoring",
                "Performance benchmarking tools"
            ]
        }
    
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    
    print("🚀 Starting DFS Prophet API...")
    print(f"📊 Environment: {settings.environment}")
    print(f"🌐 Host: {settings.app.host}")
    print(f"🔌 Port: {settings.app.port}")
    print(f"👥 Workers: {settings.app.workers}")
    print("📖 API Documentation:")
    print("   Swagger UI: http://localhost:8000/docs")
    print("   ReDoc: http://localhost:8000/redoc")
    print("   OpenAPI JSON: http://localhost:8000/openapi.json")
    
    uvicorn.run(
        "src.dfs_prophet.main:app",
        host=settings.app.host,
        port=settings.app.port,
        workers=settings.app.workers,
        reload=settings.app.debug,
        log_level=settings.app.log_level.lower()
    )



