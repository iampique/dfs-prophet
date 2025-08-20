"""
DFS Prophet - Daily Fantasy Sports API

A comprehensive API for Daily Fantasy Sports analysis, predictions, and data management.
This package provides tools for collecting sports data, generating embeddings,
and building vector-based recommendation systems for DFS optimization.
"""

__version__ = "0.1.0"
__author__ = "Pavan Vemuri"

# Public exports for convenient imports
from .config import get_settings
from .core import (
    get_embedding_generator,
    get_vector_engine,
)
from .data.collectors import get_nfl_collector
from .data.models import (
    Player,
    PlayerBase,
    PlayerStats,
    PlayerDFS,
    SearchRequest,
    SearchResponse,
)
from .api.routes import health_router, players_router

__all__ = [
    "__version__",
    "__author__",
    "get_settings",
    "get_embedding_generator",
    "get_vector_engine",
    "get_nfl_collector",
    "Player",
    "PlayerBase",
    "PlayerStats",
    "PlayerDFS",
    "SearchRequest",
    "SearchResponse",
    "health_router",
    "players_router",
]
