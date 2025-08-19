"""
Configuration management for DFS Prophet using Pydantic Settings.

This module provides centralized configuration management with environment variable
support, validation, and type safety for all application settings.
"""

from typing import List, Optional
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantSettings(BaseSettings):
    """Qdrant vector database configuration settings."""
    
    url: str = Field(
        default="http://localhost:6333",
        description="Qdrant server URL"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="Qdrant API key for authentication"
    )
    timeout: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Request timeout in seconds"
    )
    prefer_grpc: bool = Field(
        default=True,
        description="Prefer gRPC over HTTP for better performance"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="QDRANT_",
        protected_namespaces=('settings_',)
    )


class ApplicationSettings(BaseSettings):
    """Application-level configuration settings."""
    
    debug: bool = Field(
        default=True,
        description="Enable debug mode"
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins"
    )
    host: str = Field(
        default="0.0.0.0",
        description="Application host"
    )
    port: int = Field(
        default=8000,
        ge=1024,
        le=65535,
        description="Application port"
    )
    workers: int = Field(
        default=1,
        ge=1,
        le=16,
        description="Number of worker processes"
    )
    
    @validator("log_level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is supported."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    model_config = SettingsConfigDict(
        env_prefix="",
        protected_namespaces=('settings_',)
    )


class VectorDatabaseSettings(BaseSettings):
    """Vector database collection and configuration settings."""
    
    # Collection names
    players_collection: str = Field(
        default="dfs_players",
        description="Collection name for player vectors"
    )
    lineups_collection: str = Field(
        default="dfs_lineups",
        description="Collection name for lineup vectors"
    )
    games_collection: str = Field(
        default="dfs_games",
        description="Collection name for game vectors"
    )
    
    # Vector configuration
    vector_dimensions: int = Field(
        default=768,
        ge=64,
        le=2048,
        description="Vector dimensions for embeddings"
    )
    distance_metric: str = Field(
        default="Cosine",
        description="Distance metric for vector similarity"
    )
    on_disk_payload: bool = Field(
        default=True,
        description="Store payload on disk for memory efficiency"
    )
    
    # Performance settings
    batch_size: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Batch size for vector operations"
    )
    search_limit: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Default search result limit"
    )
    
    @validator("distance_metric")
    def validate_distance_metric(cls, v: str) -> str:
        """Validate distance metric is supported by Qdrant."""
        valid_metrics = ["Cosine", "Euclid", "Dot"]
        if v not in valid_metrics:
            raise ValueError(f"Distance metric must be one of: {valid_metrics}")
        return v
    
    model_config = SettingsConfigDict(
        env_prefix="VECTOR_",
        protected_namespaces=('settings_',)
    )


class EmbeddingModelSettings(BaseSettings):
    """Embedding model configuration settings."""
    
    model_name: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="HuggingFace model name for embeddings"
    )
    max_length: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Maximum sequence length for tokenization"
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="Normalize embeddings for cosine similarity"
    )
    device: str = Field(
        default="cpu",
        description="Device for model inference (cpu, cuda, mps)"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=128,
        description="Batch size for embedding generation"
    )
    
    # Model-specific settings
    use_fast_tokenizer: bool = Field(
        default=True,
        description="Use fast tokenizer for better performance"
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Trust remote code from HuggingFace models"
    )
    
    @validator("device")
    def validate_device(cls, v: str) -> str:
        """Validate device specification."""
        valid_devices = ["cpu", "cuda", "mps"]
        if v not in valid_devices:
            raise ValueError(f"Device must be one of: {valid_devices}")
        return v
    
    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        protected_namespaces=('settings_',)
    )


class BinaryQuantizationSettings(BaseSettings):
    """Binary quantization configuration for efficient vector storage."""
    
    enabled: bool = Field(
        default=True,
        description="Enable binary quantization for memory efficiency"
    )
    always_ram: bool = Field(
        default=False,
        description="Keep quantized vectors in RAM"
    )
    compression_ratio: float = Field(
        default=0.25,
        ge=0.1,
        le=1.0,
        description="Compression ratio for binary quantization"
    )
    
    # Quantization parameters
    quantile_threshold: float = Field(
        default=0.95,
        ge=0.5,
        le=0.99,
        description="Quantile threshold for binary quantization"
    )
    use_always_ram: bool = Field(
        default=False,
        description="Always keep quantized vectors in RAM"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="BINARY_QUANTIZATION_",
        protected_namespaces=('settings_',)
    )


class VectorTypeSettings(BaseSettings):
    """Configuration for different vector types in multi-vector setup."""
    
    # Vector type definitions
    stats_vector_name: str = Field(
        default="stats",
        description="Name for statistical vector type"
    )
    context_vector_name: str = Field(
        default="context",
        description="Name for contextual vector type"
    )
    value_vector_name: str = Field(
        default="value",
        description="Name for value-based vector type"
    )
    combined_vector_name: str = Field(
        default="combined",
        description="Name for combined vector type"
    )
    
    # Vector dimensions per type
    stats_vector_dimensions: int = Field(
        default=768,
        ge=64,
        le=2048,
        description="Dimensions for statistical vectors"
    )
    context_vector_dimensions: int = Field(
        default=768,
        ge=64,
        le=2048,
        description="Dimensions for contextual vectors"
    )
    value_vector_dimensions: int = Field(
        default=768,
        ge=64,
        le=2048,
        description="Dimensions for value-based vectors"
    )
    combined_vector_dimensions: int = Field(
        default=768,
        ge=64,
        le=2048,
        description="Dimensions for combined vectors"
    )
    
    # Collection naming conventions
    multi_vector_regular_collection: str = Field(
        default="dfs_players_multi_regular",
        description="Collection name for regular multi-vector storage"
    )
    multi_vector_quantized_collection: str = Field(
        default="dfs_players_multi_quantized",
        description="Collection name for quantized multi-vector storage"
    )
    
    @validator("stats_vector_dimensions", "context_vector_dimensions", 
               "value_vector_dimensions", "combined_vector_dimensions")
    def validate_vector_dimensions(cls, v: int) -> int:
        """Validate vector dimensions are consistent."""
        if v % 64 != 0:
            raise ValueError("Vector dimensions must be divisible by 64 for optimal performance")
        return v
    
    model_config = SettingsConfigDict(
        env_prefix="VECTOR_TYPE_",
        protected_namespaces=('settings_',)
    )


class MultiVectorSearchSettings(BaseSettings):
    """Configuration for multi-vector search and fusion strategies."""
    
    # Vector fusion weights
    vector_weight_stats: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for statistical vectors in fusion"
    )
    vector_weight_context: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for contextual vectors in fusion"
    )
    vector_weight_value: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for value-based vectors in fusion"
    )
    
    # Search configuration
    enable_vector_fusion: bool = Field(
        default=True,
        description="Enable vector fusion for multi-vector search"
    )
    max_vectors_per_search: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of vectors to use per search"
    )
    
    # Fusion strategies
    fusion_strategy: str = Field(
        default="weighted_average",
        description="Strategy for combining multiple vectors"
    )
    min_fusion_score: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold for fusion results"
    )
    
    # Performance thresholds
    stats_performance_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Performance threshold for statistical vectors"
    )
    context_performance_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Performance threshold for contextual vectors"
    )
    value_performance_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Performance threshold for value-based vectors"
    )
    
    @validator("fusion_strategy")
    def validate_fusion_strategy(cls, v: str) -> str:
        """Validate fusion strategy is supported."""
        valid_strategies = ["weighted_average", "max_score", "min_score", "product"]
        if v not in valid_strategies:
            raise ValueError(f"Fusion strategy must be one of: {valid_strategies}")
        return v
    
    @validator("vector_weight_stats", "vector_weight_context", "vector_weight_value")
    def validate_weights_sum(cls, v: float, values: dict) -> float:
        """Validate that weights sum to approximately 1.0."""
        if "vector_weight_stats" in values and "vector_weight_context" in values and "vector_weight_value" in values:
            total_weight = values.get("vector_weight_stats", 0) + values.get("vector_weight_context", 0) + values.get("vector_weight_value", 0)
            if abs(total_weight - 1.0) > 0.1:
                raise ValueError("Vector weights should sum to approximately 1.0")
        return v
    
    model_config = SettingsConfigDict(
        env_prefix="MULTI_VECTOR_",
        protected_namespaces=('settings_',)
    )


class EmbeddingModelPerTypeSettings(BaseSettings):
    """Embedding model configuration per vector type."""
    
    # Model settings for different vector types
    stats_model_name: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="Model for statistical vector embeddings"
    )
    context_model_name: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="Model for contextual vector embeddings"
    )
    value_model_name: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="Model for value-based vector embeddings"
    )
    combined_model_name: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="Model for combined vector embeddings"
    )
    
    # Model-specific parameters
    stats_model_max_length: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Max sequence length for statistical model"
    )
    context_model_max_length: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Max sequence length for contextual model"
    )
    value_model_max_length: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Max sequence length for value model"
    )
    combined_model_max_length: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Max sequence length for combined model"
    )
    
    # Cache settings per model
    enable_model_caching: bool = Field(
        default=True,
        description="Enable model caching for performance"
    )
    cache_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Cache TTL in hours"
    )
    
    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_MODEL_",
        protected_namespaces=('settings_',)
    )


class Settings(BaseSettings):
    """Main application settings combining all configuration sections."""
    
    # Configuration sections
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    app: ApplicationSettings = Field(default_factory=ApplicationSettings)
    vector_db: VectorDatabaseSettings = Field(default_factory=VectorDatabaseSettings)
    embedding: EmbeddingModelSettings = Field(default_factory=EmbeddingModelSettings)
    binary_quantization: BinaryQuantizationSettings = Field(
        default_factory=BinaryQuantizationSettings
    )
    
    # Multi-vector configuration sections
    vector_types: VectorTypeSettings = Field(default_factory=VectorTypeSettings)
    multi_vector_search: MultiVectorSearchSettings = Field(default_factory=MultiVectorSearchSettings)
    embedding_models: EmbeddingModelPerTypeSettings = Field(default_factory=EmbeddingModelPerTypeSettings)
    
    # Additional settings
    environment: str = Field(
        default="development",
        description="Application environment"
    )
    version: str = Field(
        default="0.1.0",
        description="Application version"
    )
    
    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting."""
        valid_environments = ["development", "staging", "production"]
        if v not in valid_environments:
            raise ValueError(f"Environment must be one of: {valid_environments}")
        return v
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=('settings_',)
    )
    
    def get_qdrant_url(self) -> str:
        """Get formatted Qdrant URL."""
        return self.qdrant.url.rstrip("/")
    
    def get_collection_name(self, collection_type: str) -> str:
        """Get collection name by type."""
        collection_map = {
            "players": self.vector_db.players_collection,
            "lineups": self.vector_db.lineups_collection,
            "games": self.vector_db.games_collection,
        }
        return collection_map.get(collection_type, f"dfs_{collection_type}")
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"
    
    # Multi-vector configuration helpers
    def get_vector_names(self) -> dict:
        """Get all vector type names."""
        return {
            "stats": self.vector_types.stats_vector_name,
            "context": self.vector_types.context_vector_name,
            "value": self.vector_types.value_vector_name,
            "combined": self.vector_types.combined_vector_name
        }
    
    def get_vector_dimensions(self) -> dict:
        """Get dimensions for each vector type."""
        return {
            "stats": self.vector_types.stats_vector_dimensions,
            "context": self.vector_types.context_vector_dimensions,
            "value": self.vector_types.value_vector_dimensions,
            "combined": self.vector_types.combined_vector_dimensions
        }
    
    def get_vector_weights(self) -> dict:
        """Get fusion weights for each vector type."""
        return {
            "stats": self.multi_vector_search.vector_weight_stats,
            "context": self.multi_vector_search.vector_weight_context,
            "value": self.multi_vector_search.vector_weight_value
        }
    
    def get_performance_thresholds(self) -> dict:
        """Get performance thresholds for each vector type."""
        return {
            "stats": self.multi_vector_search.stats_performance_threshold,
            "context": self.multi_vector_search.context_performance_threshold,
            "value": self.multi_vector_search.value_performance_threshold
        }
    
    def get_model_configs(self) -> dict:
        """Get model configurations for each vector type."""
        return {
            "stats": {
                "model_name": self.embedding_models.stats_model_name,
                "max_length": self.embedding_models.stats_model_max_length
            },
            "context": {
                "model_name": self.embedding_models.context_model_name,
                "max_length": self.embedding_models.context_model_max_length
            },
            "value": {
                "model_name": self.embedding_models.value_model_name,
                "max_length": self.embedding_models.value_model_max_length
            },
            "combined": {
                "model_name": self.embedding_models.combined_model_name,
                "max_length": self.embedding_models.combined_model_max_length
            }
        }
    
    def validate_multi_vector_config(self) -> bool:
        """Validate multi-vector configuration consistency."""
        try:
            # Check vector dimensions consistency
            dimensions = self.get_vector_dimensions()
            for vector_type, dim in dimensions.items():
                if dim % 64 != 0:
                    raise ValueError(f"Vector dimensions for {vector_type} must be divisible by 64")
            
            # Check weights sum to approximately 1.0
            weights = self.get_vector_weights()
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.1:
                raise ValueError(f"Vector weights sum to {total_weight}, should be approximately 1.0")
            
            # Check performance thresholds are valid
            thresholds = self.get_performance_thresholds()
            for vector_type, threshold in thresholds.items():
                if not 0.0 <= threshold <= 1.0:
                    raise ValueError(f"Performance threshold for {vector_type} must be between 0.0 and 1.0")
            
            return True
        except Exception as e:
            raise ValueError(f"Multi-vector configuration validation failed: {e}")
    
    def get_multi_vector_collection_name(self, collection_type: str, quantized: bool = False) -> str:
        """Get multi-vector collection name by type."""
        if quantized:
            return self.vector_types.multi_vector_quantized_collection
        else:
            return self.vector_types.multi_vector_regular_collection


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def reload_settings() -> Settings:
    """Reload settings from environment variables."""
    global settings
    settings = Settings()
    return settings



