"""
API Routes Module

This module contains all API route definitions and endpoint handlers
for the DFS Prophet API, including health checks, player data, and predictions.
"""

from .health import router as health_router
from .players import router as players_router

__all__ = ["health", "players", "health_router", "players_router"]
