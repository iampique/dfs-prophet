"""
Core Module for DFS Prophet

This module contains the core business logic including vector engines,
embedding generation, and prediction algorithms for DFS analysis.
"""

from .embedding_generator import (
    EmbeddingStrategy,
    FeatureNormalizer,
    EmbeddingCache,
    EmbeddingGenerator,
    get_embedding_generator,
)
from .vector_engine import (
    CollectionType,
    SearchResult,
    PerformanceMetrics,
    VectorEngine,
    get_vector_engine,
)

__all__ = [
    "vector_engine", 
    "embedding_generator",
    "EmbeddingStrategy",
    "FeatureNormalizer",
    "EmbeddingCache",
    "EmbeddingGenerator",
    "get_embedding_generator",
    "CollectionType",
    "SearchResult",
    "PerformanceMetrics",
    "VectorEngine",
    "get_vector_engine",
]
