"""
Core Module for DFS Prophet

This module contains the core business logic including vector engines,
embedding generation, and prediction algorithms for DFS analysis.
"""

from .embedding_generator import (
    EmbeddingStrategy,
    MultiVectorEmbeddingStrategy,
    FeatureNormalizer,
    EmbeddingCache,
    StatisticalEmbedder,
    ContextualEmbedder,
    ValueEmbedder,
    MultiVectorEmbeddingGenerator,
    EmbeddingGenerator,
    get_embedding_generator,
)
from .vector_engine import (
    CollectionType,
    MultiVectorCollectionType,
    VectorType,
    NamedVectorConfig,
    MultiVectorSearchRequest,
    MultiVectorSearchResult,
    SearchResult,
    PerformanceMetrics,
    MultiVectorPerformanceMetrics,
    VectorEngine,
    get_vector_engine,
)

__all__ = [
    "vector_engine", 
    "embedding_generator",
    "EmbeddingStrategy",
    "MultiVectorEmbeddingStrategy",
    "FeatureNormalizer",
    "EmbeddingCache",
    "StatisticalEmbedder",
    "ContextualEmbedder",
    "ValueEmbedder",
    "MultiVectorEmbeddingGenerator",
    "EmbeddingGenerator",
    "get_embedding_generator",
    "CollectionType",
    "MultiVectorCollectionType",
    "VectorType",
    "NamedVectorConfig",
    "MultiVectorSearchRequest",
    "MultiVectorSearchResult",
    "SearchResult",
    "PerformanceMetrics",
    "MultiVectorPerformanceMetrics",
    "VectorEngine",
    "get_vector_engine",
]
