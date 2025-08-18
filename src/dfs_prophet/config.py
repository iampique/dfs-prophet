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



