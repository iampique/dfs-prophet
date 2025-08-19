"""
Player data models for DFS Prophet.

This module contains comprehensive Pydantic models for player information,
statistics, DFS data, vector representations, and API request/response models.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
import numpy as np


class Position(str, Enum):
    """NFL player positions."""
    QB = "QB"
    RB = "RB"
    WR = "WR"
    TE = "TE"
    K = "K"
    DEF = "DEF"
    FLEX = "FLEX"


class Team(str, Enum):
    """NFL teams."""
    ARI = "ARI"
    ATL = "ATL"
    BAL = "BAL"
    BUF = "BUF"
    CAR = "CAR"
    CHI = "CHI"
    CIN = "CIN"
    CLE = "CLE"
    DAL = "DAL"
    DEN = "DEN"
    DET = "DET"
    GB = "GB"
    HOU = "HOU"
    IND = "IND"
    JAX = "JAX"
    KC = "KC"
    LV = "LV"
    LAC = "LAC"
    LAR = "LAR"
    MIA = "MIA"
    MIN = "MIN"
    NE = "NE"
    NO = "NO"
    NYG = "NYG"
    NYJ = "NYJ"
    PHI = "PHI"
    PIT = "PIT"
    SEA = "SEA"
    SF = "SF"
    TB = "TB"
    TEN = "TEN"
    WAS = "WAS"



class PlayerBase(BaseModel):
    """Core player information model."""
    
    player_id: str = Field(
        ...,
        description="Unique player identifier",
        example="12345"
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Player's full name",
        example="Patrick Mahomes"
    )
    position: Position = Field(
        ...,
        description="Player's primary position"
    )
    team: Team = Field(
        ...,
        description="Player's current team"
    )
    season: int = Field(
        ...,
        ge=2020,
        le=2030,
        description="NFL season year",
        example=2024
    )
    week: Optional[int] = Field(
        None,
        ge=1,
        le=25,  # Increased to handle playoffs and special weeks
        description="Week number in the season",
        example=1
    )
    age: Optional[int] = Field(
        None,
        ge=18,
        le=50,
        description="Player's age",
        example=28
    )
    experience: Optional[int] = Field(
        None,
        ge=0,
        le=20,
        description="Years of NFL experience",
        example=6
    )
    height: Optional[str] = Field(
        None,
        description="Player's height in feet-inches format",
        example="6'3\""
    )
    weight: Optional[int] = Field(
        None,
        ge=150,
        le=400,
        description="Player's weight in pounds",
        example=225
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "player_id": "12345",
                "name": "Patrick Mahomes",
                "position": "QB",
                "team": "KC",
                "season": 2024,
                "week": 1,
                "age": 28,
                "experience": 6,
                "height": "6'3\"",
                "weight": 225
            }
        }


class PlayerStats(BaseModel):
    """Statistical features for embedding generation."""
    
    # Passing statistics (QB)
    passing_yards: Optional[float] = Field(
        None,
        ge=-50,  # Allow small negative values for data issues
        description="Passing yards",
        example=312.5
    )
    passing_touchdowns: Optional[float] = Field(
        None,
        ge=0,
        description="Passing touchdowns",
        example=2.1
    )
    passing_interceptions: Optional[float] = Field(
        None,
        ge=0,
        description="Passing interceptions",
        example=0.8
    )
    completion_percentage: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Completion percentage",
        example=65.2
    )
    qb_rating: Optional[float] = Field(
        None,
        ge=0,
        le=158.3,
        description="Quarterback rating",
        example=98.7
    )
    
    # Rushing statistics (RB, QB, WR)
    rushing_yards: Optional[float] = Field(
        None,
        ge=-50,  # Allow small negative values for data issues
        description="Rushing yards",
        example=85.3
    )
    rushing_touchdowns: Optional[float] = Field(
        None,
        ge=0,
        description="Rushing touchdowns",
        example=0.8
    )
    rushing_attempts: Optional[float] = Field(
        None,
        ge=0,
        description="Rushing attempts",
        example=18.5
    )
    
    # Receiving statistics (WR, TE, RB)
    receiving_yards: Optional[float] = Field(
        None,
        ge=-50,  # Allow small negative values for data issues
        description="Receiving yards",
        example=125.8
    )
    receiving_touchdowns: Optional[float] = Field(
        None,
        ge=0,
        description="Receiving touchdowns",
        example=1.2
    )
    receptions: Optional[float] = Field(
        None,
        ge=0,
        description="Receptions",
        example=8.5
    )
    targets: Optional[float] = Field(
        None,
        ge=0,
        description="Targets",
        example=12.3
    )
    
    # Combined statistics
    total_touchdowns: Optional[float] = Field(
        None,
        ge=0,
        description="Total touchdowns (passing + rushing + receiving)",
        example=3.1
    )
    total_yards: Optional[float] = Field(
        None,
        ge=-50,
        description="Total yards (passing + rushing + receiving)",
        example=523.6
    )
    
    # Additional passing statistics
    passing_attempts: Optional[float] = Field(
        None,
        ge=0,
        description="Passing attempts",
        example=35.2
    )
    passing_completions: Optional[float] = Field(
        None,
        ge=0,
        description="Passing completions",
        example=23.8
    )
    
    # Additional receiving statistics
    receiving_receptions: Optional[float] = Field(
        None,
        ge=0,
        description="Receptions",
        example=8.5
    )
    receiving_targets: Optional[float] = Field(
        None,
        ge=0,
        description="Targets",
        example=12.3
    )
    yards_per_carry: Optional[float] = Field(
        None,
        ge=0,
        description="Yards per carry",
        example=4.6
    )
    
    # Receiving statistics (WR, TE, RB)
    receiving_yards: Optional[float] = Field(
        None,
        ge=-50,  # Allow small negative values for data issues
        description="Receiving yards",
        example=67.8
    )
    receiving_touchdowns: Optional[float] = Field(
        None,
        ge=0,
        description="Receiving touchdowns",
        example=0.5
    )
    receptions: Optional[float] = Field(
        None,
        ge=0,
        description="Receptions",
        example=5.2
    )
    targets: Optional[float] = Field(
        None,
        ge=0,
        description="Targets",
        example=7.8
    )
    catch_percentage: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Catch percentage",
        example=66.7
    )
    
    # Kicking statistics (K)
    field_goals_made: Optional[float] = Field(
        None,
        ge=0,
        description="Field goals made",
        example=2.1
    )
    field_goals_attempted: Optional[float] = Field(
        None,
        ge=0,
        description="Field goals attempted",
        example=2.5
    )
    extra_points_made: Optional[float] = Field(
        None,
        ge=0,
        description="Extra points made",
        example=3.2
    )
    extra_points_attempted: Optional[float] = Field(
        None,
        ge=0,
        description="Extra points attempted",
        example=3.2
    )
    
    # Defense statistics (DEF)
    sacks: Optional[float] = Field(
        None,
        ge=0,
        description="Sacks",
        example=2.8
    )
    interceptions: Optional[float] = Field(
        None,
        ge=0,
        description="Interceptions",
        example=1.2
    )
    fumbles_recovered: Optional[float] = Field(
        None,
        ge=0,
        description="Fumbles recovered",
        example=0.5
    )
    points_allowed: Optional[float] = Field(
        None,
        ge=0,
        description="Points allowed",
        example=21.3
    )
    
    # Fantasy points
    fantasy_points: Optional[float] = Field(
        None,
        ge=0,
        description="Fantasy points scored",
        example=18.7
    )
    fantasy_points_ppr: Optional[float] = Field(
        None,
        ge=0,
        description="PPR fantasy points scored",
        example=21.2
    )
    
    # Advanced metrics
    snap_percentage: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Snap percentage",
        example=85.3
    )
    red_zone_targets: Optional[float] = Field(
        None,
        ge=0,
        description="Red zone targets",
        example=1.2
    )
    red_zone_touchdowns: Optional[float] = Field(
        None,
        ge=0,
        description="Red zone touchdowns",
        example=0.4
    )
    
    @validator('fantasy_points', 'fantasy_points_ppr')
    def validate_fantasy_points(cls, v):
        """Validate fantasy points are reasonable."""
        if v is not None and v > 500:  # Increased limit for season totals
            raise ValueError("Fantasy points cannot exceed 500")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "passing_yards": 312.5,
                "passing_touchdowns": 2.1,
                "passing_interceptions": 0.8,
                "completion_percentage": 65.2,
                "qb_rating": 98.7,
                "rushing_yards": 25.3,
                "rushing_touchdowns": 0.2,
                "fantasy_points": 18.7,
                "fantasy_points_ppr": 18.7
            }
        }


class PlayerDFS(BaseModel):
    """DFS-specific data model with historical trends."""
    
    salary: int = Field(
        ...,
        ge=1000,
        le=10000,
        description="Player's DFS salary",
        example=8500
    )
    projected_points: float = Field(
        ...,
        ge=0,
        le=500,  # Increased limit for season totals
        description="Projected fantasy points",
        example=18.7
    )
    ownership_percentage: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Projected ownership percentage",
        example=12.5
    )
    value_rating: Optional[float] = Field(
        None,
        ge=0,
        le=100,  # Increased limit for realistic value ratings
        description="Value rating (points per $1000 salary)",
        example=2.2
    )
    game_total: Optional[float] = Field(
        None,
        ge=20,
        le=80,
        description="Game total over/under",
        example=48.5
    )
    team_spread: Optional[float] = Field(
        None,
        description="Team point spread",
        example=-3.5
    )
    weather_conditions: Optional[str] = Field(
        None,
        description="Weather conditions",
        example="Clear, 72°F, 5mph wind"
    )
    injury_status: Optional[str] = Field(
        None,
        description="Player injury status",
        example="Questionable"
    )
    matchup_rating: Optional[float] = Field(
        None,
        ge=1,
        le=10,
        description="Matchup difficulty rating (1=easy, 10=hard)",
        example=3.5
    )
    
    # Historical salary and ownership data
    salary_history: Optional[List[int]] = Field(
        None,
        description="Historical salary data (last 5 weeks)",
        example=[8200, 8300, 8400, 8450, 8500]
    )
    ownership_history: Optional[List[float]] = Field(
        None,
        description="Historical ownership data (last 5 weeks)",
        example=[10.2, 11.8, 12.1, 12.3, 12.5]
    )
    salary_trend: Optional[float] = Field(
        None,
        description="Salary trend (ratio of current to average)",
        example=1.05
    )
    ownership_trend: Optional[float] = Field(
        None,
        description="Ownership trend (ratio of current to average)",
        example=0.95
    )
    salary_volatility: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Salary volatility (standard deviation / mean)",
        example=0.15
    )
    ownership_volatility: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Ownership volatility (standard deviation / mean)",
        example=0.25
    )
    
    # ROI and performance history
    roi_history: Optional[List[float]] = Field(
        None,
        description="Historical ROI data (last 5 weeks)",
        example=[1.2, 0.8, 1.5, 0.9, 1.1]
    )
    avg_roi: Optional[float] = Field(
        None,
        description="Average ROI over historical period",
        example=1.1
    )
    roi_volatility: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="ROI volatility (standard deviation / mean)",
        example=0.30
    )
    consistency_rating: Optional[float] = Field(
        None,
        ge=0,
        le=10,
        description="Performance consistency rating (1=volatile, 10=consistent)",
        example=7.5
    )
    
    @validator('value_rating', pre=True, always=True)
    def calculate_value_rating(cls, v, values):
        """Calculate value rating if not provided."""
        if v is None and 'salary' in values and 'projected_points' in values:
            if values['salary'] > 0:
                return (values['projected_points'] / values['salary']) * 1000
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "salary": 8500,
                "projected_points": 18.7,
                "ownership_percentage": 12.5,
                "value_rating": 2.2,
                "game_total": 48.5,
                "team_spread": -3.5,
                "weather_conditions": "Clear, 72°F, 5mph wind",
                "injury_status": "Questionable",
                "matchup_rating": 3.5
            }
        }


class PlayerVector(BaseModel):
    """Vector representation and metadata model."""
    
    vector_id: str = Field(
        ...,
        description="Unique vector identifier",
        example="vec_12345_2024_1"
    )
    embedding: List[float] = Field(
        ...,
        description="Player embedding vector",
        min_items=384,
        max_items=2048
    )
    vector_dimensions: int = Field(
        ...,
        ge=384,
        le=2048,
        description="Number of vector dimensions",
        example=768
    )
    embedding_model: str = Field(
        ...,
        description="Model used to generate embedding",
        example="BAAI/bge-base-en-v1.5"
    )
    embedding_timestamp: datetime = Field(
        ...,
        description="When the embedding was generated"
    )
    text_features: Optional[str] = Field(
        None,
        description="Text features used for embedding generation",
        example="Patrick Mahomes QB KC 312.5 passing yards 2.1 touchdowns"
    )
    similarity_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Similarity score from search",
        example=0.87
    )
    
    @validator('embedding')
    def validate_embedding_dimensions(cls, v, values):
        """Validate embedding dimensions match vector_dimensions."""
        if 'vector_dimensions' in values and len(v) != values['vector_dimensions']:
            raise ValueError(f"Embedding length {len(v)} must match vector_dimensions {values['vector_dimensions']}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "vector_id": "vec_12345_2024_1",
                "embedding": [0.1, 0.2, 0.3, ...],
                "vector_dimensions": 768,
                "embedding_model": "BAAI/bge-base-en-v1.5",
                "embedding_timestamp": "2024-08-18T10:30:00Z",
                "text_features": "Patrick Mahomes QB KC 312.5 passing yards 2.1 touchdowns",
                "similarity_score": 0.87
            }
        }


class PlayerStatVector(BaseModel):
    """Statistical performance vector for player statistics."""
    
    vector_id: str = Field(
        ...,
        description="Unique statistical vector identifier",
        example="stat_vec_12345_2024_1"
    )
    embedding: List[float] = Field(
        ...,
        description="Statistical performance embedding vector",
        min_items=384,
        max_items=2048
    )
    vector_dimensions: int = Field(
        ...,
        ge=384,
        le=2048,
        description="Number of vector dimensions",
        example=768
    )
    embedding_model: str = Field(
        ...,
        description="Model used to generate statistical embedding",
        example="BAAI/bge-base-en-v1.5"
    )
    embedding_timestamp: datetime = Field(
        ...,
        description="When the statistical embedding was generated"
    )
    stat_features: Dict[str, float] = Field(
        ...,
        description="Statistical features used for embedding",
        example={
            "passing_yards": 312.5,
            "rushing_yards": 25.3,
            "receiving_yards": 0.0,
            "total_touchdowns": 2.3,
            "fantasy_points": 18.7
        }
    )
    stat_summary: str = Field(
        ...,
        description="Text summary of statistical features",
        example="312.5 passing yards, 25.3 rushing yards, 2.3 total touchdowns, 18.7 fantasy points"
    )
    similarity_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Statistical similarity score from search",
        example=0.92
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "vector_id": "stat_vec_12345_2024_1",
                "embedding": [0.1, 0.2, 0.3, ...],
                "vector_dimensions": 768,
                "embedding_model": "BAAI/bge-base-en-v1.5",
                "embedding_timestamp": "2024-08-18T10:30:00Z",
                "stat_features": {
                    "passing_yards": 312.5,
                    "rushing_yards": 25.3,
                    "total_touchdowns": 2.3,
                    "fantasy_points": 18.7
                },
                "stat_summary": "312.5 passing yards, 25.3 rushing yards, 2.3 total touchdowns, 18.7 fantasy points",
                "similarity_score": 0.92
            }
        }


class PlayerContextVector(BaseModel):
    """Game context vector for situational factors."""
    
    vector_id: str = Field(
        ...,
        description="Unique context vector identifier",
        example="context_vec_12345_2024_1"
    )
    embedding: List[float] = Field(
        ...,
        description="Game context embedding vector",
        min_items=384,
        max_items=2048
    )
    vector_dimensions: int = Field(
        ...,
        ge=384,
        le=2048,
        description="Number of vector dimensions",
        example=768
    )
    embedding_model: str = Field(
        ...,
        description="Model used to generate context embedding",
        example="BAAI/bge-base-en-v1.5"
    )
    embedding_timestamp: datetime = Field(
        ...,
        description="When the context embedding was generated"
    )
    context_features: Dict[str, Any] = Field(
        ...,
        description="Contextual features used for embedding",
        example={
            "opponent_team": "BUF",
            "opponent_strength": 8.5,
            "home_away": "home",
            "weather_temp": 72.0,
            "weather_wind": 5.0,
            "weather_conditions": "clear",
            "game_total": 48.5,
            "team_spread": -3.5,
            "injury_status": "healthy"
        }
    )
    context_summary: str = Field(
        ...,
        description="Text summary of contextual features",
        example="Home game vs BUF (strength: 8.5), clear weather 72°F 5mph wind, game total 48.5, spread -3.5"
    )
    similarity_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Context similarity score from search",
        example=0.85
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "vector_id": "context_vec_12345_2024_1",
                "embedding": [0.1, 0.2, 0.3, ...],
                "vector_dimensions": 768,
                "embedding_model": "BAAI/bge-base-en-v1.5",
                "embedding_timestamp": "2024-08-18T10:30:00Z",
                "context_features": {
                    "opponent_team": "BUF",
                    "opponent_strength": 8.5,
                    "home_away": "home",
                    "weather_temp": 72.0,
                    "weather_wind": 5.0,
                    "weather_conditions": "clear",
                    "game_total": 48.5,
                    "team_spread": -3.5,
                    "injury_status": "healthy"
                },
                "context_summary": "Home game vs BUF (strength: 8.5), clear weather 72°F 5mph wind, game total 48.5, spread -3.5",
                "similarity_score": 0.85
            }
        }


class PlayerValueVector(BaseModel):
    """DFS value patterns vector for salary and ownership trends."""
    
    vector_id: str = Field(
        ...,
        description="Unique value vector identifier",
        example="value_vec_12345_2024_1"
    )
    embedding: List[float] = Field(
        ...,
        description="DFS value patterns embedding vector",
        min_items=384,
        max_items=2048
    )
    vector_dimensions: int = Field(
        ...,
        ge=384,
        le=2048,
        description="Number of vector dimensions",
        example=768
    )
    embedding_model: str = Field(
        ...,
        description="Model used to generate value embedding",
        example="BAAI/bge-base-en-v1.5"
    )
    embedding_timestamp: datetime = Field(
        ...,
        description="When the value embedding was generated"
    )
    value_features: Dict[str, Any] = Field(
        ...,
        description="Value features used for embedding",
        example={
            "current_salary": 8500,
            "salary_trend": 1.05,
            "ownership_percentage": 12.5,
            "ownership_trend": 0.95,
            "value_rating": 2.2,
            "roi_history": [1.2, 0.8, 1.5, 0.9, 1.1],
            "avg_roi": 1.1,
            "salary_volatility": 0.15,
            "ownership_volatility": 0.25
        }
    )
    value_summary: str = Field(
        ...,
        description="Text summary of value features",
        example="Salary $8500 (trending +5%), ownership 12.5% (trending -5%), value rating 2.2, avg ROI 1.1"
    )
    similarity_score: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Value similarity score from search",
        example=0.78
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "vector_id": "value_vec_12345_2024_1",
                "embedding": [0.1, 0.2, 0.3, ...],
                "vector_dimensions": 768,
                "embedding_model": "BAAI/bge-base-en-v1.5",
                "embedding_timestamp": "2024-08-18T10:30:00Z",
                "value_features": {
                    "current_salary": 8500,
                    "salary_trend": 1.05,
                    "ownership_percentage": 12.5,
                    "ownership_trend": 0.95,
                    "value_rating": 2.2,
                    "roi_history": [1.2, 0.8, 1.5, 0.9, 1.1],
                    "avg_roi": 1.1,
                    "salary_volatility": 0.15,
                    "ownership_volatility": 0.25
                },
                "value_summary": "Salary $8500 (trending +5%), ownership 12.5% (trending -5%), value rating 2.2, avg ROI 1.1",
                "similarity_score": 0.78
            }
        }


class PlayerMultiVector(BaseModel):
    """Container for all vector types with named vectors."""
    
    player_id: str = Field(
        ...,
        description="Player identifier",
        example="12345"
    )
    season: int = Field(
        ...,
        ge=2020,
        le=2030,
        description="NFL season year",
        example=2024
    )
    week: Optional[int] = Field(
        None,
        ge=1,
        le=25,
        description="Week number in the season",
        example=1
    )
    stat_vector: Optional[PlayerStatVector] = Field(
        None,
        description="Statistical performance vector"
    )
    context_vector: Optional[PlayerContextVector] = Field(
        None,
        description="Game context vector"
    )
    value_vector: Optional[PlayerValueVector] = Field(
        None,
        description="DFS value patterns vector"
    )
    combined_vector: Optional[PlayerVector] = Field(
        None,
        description="Combined multi-modal vector"
    )
    vector_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional vector metadata",
        example={
            "vector_count": 4,
            "embedding_models": ["BAAI/bge-base-en-v1.5"],
            "total_dimensions": 3072,
            "last_updated": "2024-08-18T10:30:00Z"
        }
    )
    
    @property
    def available_vectors(self) -> List[str]:
        """Get list of available vector types."""
        vectors = []
        if self.stat_vector:
            vectors.append("stat")
        if self.context_vector:
            vectors.append("context")
        if self.value_vector:
            vectors.append("value")
        if self.combined_vector:
            vectors.append("combined")
        return vectors
    
    @property
    def vector_count(self) -> int:
        """Get total number of available vectors."""
        return len(self.available_vectors)
    
    class Config:
        json_schema_extra = {
            "example": {
                "player_id": "12345",
                "season": 2024,
                "week": 1,
                "vector_metadata": {
                    "vector_count": 4,
                    "embedding_models": ["BAAI/bge-base-en-v1.5"],
                    "total_dimensions": 3072,
                    "last_updated": "2024-08-18T10:30:00Z"
                }
            }
        }


class Player(BaseModel):
    """Complete player model combining all player data with multi-vector support."""
    
    base: PlayerBase = Field(..., description="Core player information")
    stats: Optional[PlayerStats] = Field(None, description="Player statistics")
    dfs: Optional[PlayerDFS] = Field(None, description="DFS-specific data")
    vector: Optional[PlayerVector] = Field(None, description="Vector representation")
    multi_vector: Optional[PlayerMultiVector] = Field(None, description="Multi-modal vector representations")
    
    @property
    def player_id(self) -> str:
        """Get player ID from base data."""
        return self.base.player_id
    
    @property
    def name(self) -> str:
        """Get player name from base data."""
        return self.base.name
    
    @property
    def position(self) -> Position:
        """Get player position from base data."""
        return self.base.position
    
    @property
    def team(self) -> Team:
        """Get player team from base data."""
        return self.base.team
    
    class Config:
        json_schema_extra = {
            "example": {
                "base": {
                    "player_id": "12345",
                    "name": "Patrick Mahomes",
                    "position": "QB",
                    "team": "KC",
                    "season": 2024,
                    "week": 1
                },
                "stats": {
                    "passing_yards": 312.5,
                    "passing_touchdowns": 2.1,
                    "fantasy_points": 18.7
                },
                "dfs": {
                    "salary": 8500,
                    "projected_points": 18.7,
                    "ownership_percentage": 12.5
                }
            }
        }


class SearchRequest(BaseModel):
    """API request model for player search."""
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query text",
        example="high passing yards quarterback"
    )
    position: Optional[Position] = Field(
        None,
        description="Filter by player position"
    )
    team: Optional[Team] = Field(
        None,
        description="Filter by team"
    )
    season: Optional[int] = Field(
        None,
        ge=2020,
        le=2030,
        description="Filter by season"
    )
    week: Optional[int] = Field(
        None,
        ge=1,
        le=21,
        description="Filter by week"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return",
        example=10
    )
    min_similarity: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Minimum similarity score threshold",
        example=0.7
    )
    include_stats: bool = Field(
        default=True,
        description="Include player statistics in response"
    )
    include_dfs: bool = Field(
        default=True,
        description="Include DFS data in response"
    )
    include_vector: bool = Field(
        default=False,
        description="Include vector data in response"
    )
    vector_types: Optional[List[str]] = Field(
        None,
        description="Vector types to search across (for multi-vector search)",
        example=["stat", "context", "value", "combined"]
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "high passing yards quarterback",
                "position": "QB",
                "season": 2024,
                "limit": 10,
                "min_similarity": 0.7,
                "include_stats": True,
                "include_dfs": True
            }
        }


class SearchResponse(BaseModel):
    """API response model for player search with performance metrics."""
    
    query: str = Field(..., description="Original search query")
    results: List[Player] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    search_time_ms: float = Field(..., description="Search execution time in milliseconds")
    vector_search_time_ms: Optional[float] = Field(
        None,
        description="Vector search execution time in milliseconds"
    )
    embedding_time_ms: Optional[float] = Field(
        None,
        description="Query embedding generation time in milliseconds"
    )
    similarity_scores: List[float] = Field(
        default_factory=list,
        description="Similarity scores for results"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional search metadata"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "high passing yards quarterback",
                "total_results": 5,
                "search_time_ms": 125.5,
                "vector_search_time_ms": 45.2,
                "embedding_time_ms": 23.1,
                "similarity_scores": [0.87, 0.82, 0.79, 0.76, 0.73],
                "metadata": {
                    "collection": "dfs_players",
                    "vector_dimensions": 768,
                    "embedding_model": "BAAI/bge-base-en-v1.5"
                }
            }
        }


class MultiVectorSearchRequest(BaseModel):
    """API request model for multi-vector player search."""
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query text",
        example="high passing yards quarterback in good weather"
    )
    vector_types: List[str] = Field(
        default=["combined"],
        description="Vector types to search across",
        example=["stat", "context", "value", "combined"]
    )
    vector_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Weight for each vector type in combined search",
        example={"stat": 0.4, "context": 0.3, "value": 0.3}
    )
    position: Optional[Position] = Field(
        None,
        description="Filter by player position"
    )
    team: Optional[Team] = Field(
        None,
        description="Filter by team"
    )
    season: Optional[int] = Field(
        None,
        ge=2020,
        le=2030,
        description="Filter by season"
    )
    week: Optional[int] = Field(
        None,
        ge=1,
        le=25,
        description="Filter by week"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return",
        example=10
    )
    min_similarity: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Minimum similarity score threshold",
        example=0.7
    )
    include_stats: bool = Field(
        default=True,
        description="Include player statistics in response"
    )
    include_dfs: bool = Field(
        default=True,
        description="Include DFS data in response"
    )
    include_vectors: bool = Field(
        default=False,
        description="Include vector data in response"
    )
    search_strategy: str = Field(
        default="weighted_combination",
        description="Multi-vector search strategy",
        example="weighted_combination"
    )
    
    @validator('vector_types')
    def validate_vector_types(cls, v):
        """Validate vector types are supported."""
        valid_types = ["stat", "context", "value", "combined"]
        for vt in v:
            if vt not in valid_types:
                raise ValueError(f"Vector type '{vt}' not supported. Valid types: {valid_types}")
        return v
    
    @validator('vector_weights')
    def validate_vector_weights(cls, v, values):
        """Validate vector weights sum to 1.0."""
        if v is not None:
            total_weight = sum(v.values())
            if abs(total_weight - 1.0) > 0.01:
                raise ValueError(f"Vector weights must sum to 1.0, got {total_weight}")
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "high passing yards quarterback in good weather",
                "vector_types": ["stat", "context", "value"],
                "vector_weights": {"stat": 0.4, "context": 0.3, "value": 0.3},
                "position": "QB",
                "season": 2024,
                "limit": 10,
                "min_similarity": 0.7,
                "search_strategy": "weighted_combination"
            }
        }


class MultiVectorSearchResponse(BaseModel):
    """API response model for multi-vector player search with vector-specific scores."""
    
    query: str = Field(..., description="Original search query")
    results: List[Player] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results found")
    search_time_ms: float = Field(..., description="Total search execution time in milliseconds")
    vector_scores: Dict[str, List[float]] = Field(
        ...,
        description="Similarity scores for each vector type",
        example={
            "stat": [0.92, 0.88, 0.85, 0.82, 0.79],
            "context": [0.85, 0.82, 0.78, 0.75, 0.72],
            "value": [0.78, 0.75, 0.72, 0.69, 0.66],
            "combined": [0.87, 0.83, 0.79, 0.76, 0.73]
        }
    )
    combined_scores: List[float] = Field(
        ...,
        description="Combined similarity scores for results"
    )
    vector_search_times: Dict[str, float] = Field(
        ...,
        description="Search time for each vector type in milliseconds",
        example={
            "stat": 45.2,
            "context": 38.7,
            "value": 42.1,
            "combined": 125.5
        }
    )
    embedding_times: Dict[str, float] = Field(
        ...,
        description="Embedding generation time for each vector type in milliseconds",
        example={
            "stat": 23.1,
            "context": 21.5,
            "value": 22.8,
            "combined": 67.4
        }
    )
    search_strategy: str = Field(
        ...,
        description="Search strategy used",
        example="weighted_combination"
    )
    vector_weights: Optional[Dict[str, float]] = Field(
        None,
        description="Weights used for each vector type",
        example={"stat": 0.4, "context": 0.3, "value": 0.3}
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional search metadata",
        example={
            "vector_types_searched": ["stat", "context", "value"],
            "total_vectors_processed": 15000,
            "embedding_models": ["BAAI/bge-base-en-v1.5"],
            "collections_searched": ["dfs_players_stat", "dfs_players_context", "dfs_players_value"]
        }
    )
    
    @property
    def average_combined_score(self) -> float:
        """Calculate average combined similarity score."""
        if not self.combined_scores:
            return 0.0
        return sum(self.combined_scores) / len(self.combined_scores)
    
    @property
    def total_vector_search_time(self) -> float:
        """Calculate total vector search time."""
        return sum(self.vector_search_times.values())
    
    @property
    def total_embedding_time(self) -> float:
        """Calculate total embedding generation time."""
        return sum(self.embedding_times.values())
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "high passing yards quarterback in good weather",
                "total_results": 5,
                "search_time_ms": 125.5,
                "vector_scores": {
                    "stat": [0.92, 0.88, 0.85, 0.82, 0.79],
                    "context": [0.85, 0.82, 0.78, 0.75, 0.72],
                    "value": [0.78, 0.75, 0.72, 0.69, 0.66],
                    "combined": [0.87, 0.83, 0.79, 0.76, 0.73]
                },
                "combined_scores": [0.87, 0.83, 0.79, 0.76, 0.73],
                "search_strategy": "weighted_combination",
                "vector_weights": {"stat": 0.4, "context": 0.3, "value": 0.3}
            }
        }


class QuantizationComparison(BaseModel):
    """Performance comparison results for binary quantization."""
    
    original_size_mb: float = Field(
        ...,
        description="Original vector storage size in MB"
    )
    quantized_size_mb: float = Field(
        ...,
        description="Quantized vector storage size in MB"
    )
    compression_ratio: float = Field(
        ...,
        ge=0,
        le=1,
        description="Compression ratio achieved"
    )
    memory_savings_percent: float = Field(
        ...,
        ge=0,
        le=100,
        description="Percentage of memory saved"
    )
    search_speed_original_ms: float = Field(
        ...,
        description="Search speed with original vectors (ms)"
    )
    search_speed_quantized_ms: float = Field(
        ...,
        description="Search speed with quantized vectors (ms)"
    )
    speed_improvement_percent: float = Field(
        ...,
        description="Percentage improvement in search speed"
    )
    accuracy_loss_percent: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Percentage loss in search accuracy"
    )
    vector_count: int = Field(
        ...,
        description="Number of vectors in comparison"
    )
    test_queries: int = Field(
        ...,
        description="Number of test queries used"
    )
    
    @property
    def storage_efficiency(self) -> float:
        """Calculate storage efficiency."""
        return self.original_size_mb / self.quantized_size_mb
    
    @property
    def speed_efficiency(self) -> float:
        """Calculate speed efficiency."""
        return self.search_speed_original_ms / self.search_speed_quantized_ms
    
    class Config:
        json_schema_extra = {
            "example": {
                "original_size_mb": 1024.0,
                "quantized_size_mb": 256.0,
                "compression_ratio": 0.25,
                "memory_savings_percent": 75.0,
                "search_speed_original_ms": 150.0,
                "search_speed_quantized_ms": 45.0,
                "speed_improvement_percent": 70.0,
                "accuracy_loss_percent": 2.5,
                "vector_count": 10000,
                "test_queries": 100
            }
        }



