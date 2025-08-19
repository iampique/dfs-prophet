"""
Pytest configuration for DFS Prophet multi-vector integration tests.

Provides fixtures and configuration for comprehensive testing of the multi-vector system.
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Test configuration
pytest_plugins = []


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_settings():
    """Provide test-specific settings."""
    from dfs_prophet.config import get_settings
    
    # Override settings for testing
    os.environ["ENVIRONMENT"] = "test"
    os.environ["QDANT_URL"] = "http://localhost:6333"
    os.environ["LOG_LEVEL"] = "WARNING"
    
    return get_settings()


@pytest.fixture(scope="function")
async def clean_vector_collections():
    """Clean vector collections before each test."""
    from dfs_prophet.core.vector_engine import VectorEngine, CollectionType
    
    vector_engine = VectorEngine()
    
    # Clean collections
    try:
        await vector_engine.clear_collection(CollectionType.REGULAR)
        await vector_engine.clear_collection(CollectionType.BINARY_QUANTIZED)
    except Exception:
        pass  # Collections might not exist yet
    
    yield vector_engine
    
    # Clean up after test
    try:
        await vector_engine.clear_collection(CollectionType.REGULAR)
        await vector_engine.clear_collection(CollectionType.BINARY_QUANTIZED)
    except Exception:
        pass


@pytest.fixture(scope="function")
def mock_embedding_generator():
    """Provide a mock embedding generator for testing."""
    from unittest.mock import AsyncMock, MagicMock
    
    mock_generator = MagicMock()
    mock_generator.generate_player_embedding = AsyncMock(return_value=[0.1] * 768)
    mock_generator.generate_query_embedding = AsyncMock(return_value=[0.1] * 768)
    
    return mock_generator


@pytest.fixture(scope="function")
def sample_test_players():
    """Provide sample test players for testing."""
    from dfs_prophet.data.models.player import (
        Player, PlayerStats, PlayerContext, PlayerValue, Position, Team
    )
    
    players = []
    
    # QB
    players.append(Player(
        player_id="test_qb_1",
        name="Test QB 1",
        position=Position.QB,
        team=Team.KC,
        season=2024,
        week=1,
        stats=PlayerStats(
            passing_yards=4000,
            passing_touchdowns=30,
            passing_interceptions=10,
            rushing_yards=300,
            rushing_touchdowns=4,
            fantasy_points=320.0,
            games_played=17
        ),
        context=PlayerContext(
            opponent_team=Team.BUF,
            home_away="home",
            weather_conditions="clear",
            injury_status="healthy",
            team_offensive_rank=3,
            opponent_defensive_rank=12
        ),
        value=PlayerValue(
            salary=8500,
            ownership_percentage=20.0,
            value_rating=8.0,
            consistency_score=0.85,
            upside_potential=0.90
        )
    ))
    
    # RB
    players.append(Player(
        player_id="test_rb_1",
        name="Test RB 1",
        position=Position.RB,
        team=Team.SF,
        season=2024,
        week=1,
        stats=PlayerStats(
            rushing_yards=1200,
            rushing_touchdowns=12,
            receiving_yards=500,
            receiving_touchdowns=5,
            fantasy_points=280.0,
            games_played=17
        ),
        context=PlayerContext(
            opponent_team=Team.LAR,
            home_away="away",
            weather_conditions="clear",
            injury_status="healthy",
            team_offensive_rank=5,
            opponent_defensive_rank=18
        ),
        value=PlayerValue(
            salary=8000,
            ownership_percentage=25.0,
            value_rating=8.5,
            consistency_score=0.90,
            upside_potential=0.95
        )
    ))
    
    # WR
    players.append(Player(
        player_id="test_wr_1",
        name="Test WR 1",
        position=Position.WR,
        team=Team.LV,
        season=2024,
        week=1,
        stats=PlayerStats(
            receiving_yards=1100,
            receiving_touchdowns=10,
            targets=130,
            receptions=85,
            fantasy_points=250.0,
            games_played=17
        ),
        context=PlayerContext(
            opponent_team=Team.DEN,
            home_away="home",
            weather_conditions="clear",
            injury_status="healthy",
            team_offensive_rank=15,
            opponent_defensive_rank=8
        ),
        value=PlayerValue(
            salary=7500,
            ownership_percentage=18.0,
            value_rating=7.5,
            consistency_score=0.82,
            upside_potential=0.88
        )
    ))
    
    return players


@pytest.fixture(scope="function")
def performance_thresholds():
    """Provide performance thresholds for testing."""
    return {
        "max_ingestion_time": 30.0,  # seconds
        "max_search_time": 5.0,      # seconds
        "max_concurrent_time": 10.0,  # seconds
        "min_similarity_score": 0.0,
        "max_similarity_score": 1.0,
        "min_health_score": 0.0,
        "max_health_score": 100.0
    }


# Pytest configuration
def pytest_configure(config):
    """Configure pytest for multi-vector testing."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "slow: marks tests as slow running tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Add integration marker to multi-vector tests
        if "test_multi_vector" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Add performance marker to performance tests
        if "performance" in item.nodeid.lower():
            item.add_marker(pytest.mark.performance)
        
        # Add slow marker to tests that might take time
        if any(keyword in item.nodeid.lower() for keyword in ["large_scale", "concurrent", "benchmark"]):
            item.add_marker(pytest.mark.slow)
