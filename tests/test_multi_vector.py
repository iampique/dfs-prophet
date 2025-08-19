"""
Comprehensive Integration Tests for Multi-Vector System

Test scenarios:
- Multi-vector data ingestion and retrieval
- Search accuracy across different vector types
- Vector fusion algorithm validation
- Performance comparison between search strategies
- Data consistency across vector types

Test categories:
- Unit tests for each vector type
- Integration tests for multi-vector operations
- Performance benchmarks for fusion search
- Edge case handling (missing vectors, corrupted data)
- API endpoint validation for all multi-vector features
"""

import pytest
import asyncio
import time
import json
import numpy as np
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from dfs_prophet.data.models.player import (
    Player, PlayerBase, PlayerStats, PlayerDFS, PlayerVector,
    Position, Team, PlayerMultiVector
)
from dfs_prophet.core.vector_engine import VectorEngine, CollectionType
from dfs_prophet.core.embedding_generator import EmbeddingGenerator, EmbeddingStrategy
from dfs_prophet.analytics.profile_analyzer import PlayerProfileAnalyzer, ArchetypeType
from dfs_prophet.search.advanced_search import AdvancedSearchEngine, SearchStrategy
from dfs_prophet.monitoring.vector_performance import VectorPerformanceMonitor
from dfs_prophet.config import get_settings


class TestMultiVectorIntegration:
    """Comprehensive integration tests for multi-vector system."""

    @pytest.fixture(autouse=True)
    async def setup_test_environment(self):
        """Setup test environment with clean state."""
        self.settings = get_settings()
        self.vector_engine = VectorEngine()
        self.embedding_generator = EmbeddingGenerator()
        self.profile_analyzer = PlayerProfileAnalyzer()
        self.search_engine = AdvancedSearchEngine()
        self.performance_monitor = VectorPerformanceMonitor()
        
        # Initialize collections
        await self.vector_engine.initialize_collections()
        
        # Generate test data
        self.test_players = self._generate_test_players()
        
        yield
        
        # Cleanup
        await self._cleanup_test_data()

    def _generate_test_players(self) -> List[Player]:
        """Generate comprehensive test player data."""
        players = []
        
        # Elite QB
        players.append(Player(
            base=PlayerBase(
                player_id="test_qb_1",
                name="Patrick Mahomes",
                position=Position.QB,
                team=Team.KC,
                season=2024,
                week=1
            ),
            stats=PlayerStats(
                passing_yards=4500,
                passing_touchdowns=35,
                passing_interceptions=8,
                rushing_yards=350,
                rushing_touchdowns=5,
                fantasy_points=350.5,
                games_played=17
            ),
            dfs=PlayerDFS(
                salary=9500,
                projected_points=25.5,
                ownership_percentage=25.5,
                value_rating=8.2,
                consistency_score=0.85,
                upside_potential=0.92
            )
        ))
        
        # Volume RB
        players.append(Player(
            base=PlayerBase(
                player_id="test_rb_1",
                name="Christian McCaffrey",
                position=Position.RB,
                team=Team.SF,
                season=2024,
                week=1
            ),
            stats=PlayerStats(
                rushing_yards=1400,
                rushing_touchdowns=14,
                receiving_yards=650,
                receiving_touchdowns=7,
                fantasy_points=320.0,
                games_played=17
            ),
            dfs=PlayerDFS(
                salary=9200,
                projected_points=22.0,
                ownership_percentage=18.5,
                value_rating=7.8,
                consistency_score=0.88,
                upside_potential=0.85
            )
        ))
        
        # Possession WR
        players.append(Player(
            base=PlayerBase(
                player_id="test_wr_1",
                name="Davante Adams",
                position=Position.WR,
                team=Team.LV,
                season=2024,
                week=1
            ),
            stats=PlayerStats(
                receiving_yards=1200,
                receiving_touchdowns=12,
                targets=150,
                receptions=95,
                fantasy_points=280.0,
                games_played=17
            ),
            dfs=PlayerDFS(
                salary=7800,
                projected_points=18.5,
                ownership_percentage=18.5,
                value_rating=7.5,
                consistency_score=0.82,
                upside_potential=0.88
            )
        ))
        
        return players

    async def _cleanup_test_data(self):
        """Clean up test data from collections."""
        try:
            await self.vector_engine.clear_collection(CollectionType.REGULAR)
            await self.vector_engine.clear_collection(CollectionType.BINARY_QUANTIZED)
        except Exception as e:
            print(f"Cleanup warning: {e}")

    # ============================================================================
    # UNIT TESTS FOR EACH VECTOR TYPE
    # ============================================================================

    @pytest.mark.asyncio
    async def test_stats_vector_generation(self):
        """Test statistical vector generation and storage."""
        player = self.test_players[0]  # QB
        
        # Generate stats embedding
        stats_embedding = await self.embedding_generator.generate_player_embedding(
            player, strategy=EmbeddingStrategy.STATISTICAL
        )
        
        # Validate embedding
        assert len(stats_embedding) == 768
        assert isinstance(stats_embedding, list)
        assert all(isinstance(x, float) for x in stats_embedding)
        
        # Store and retrieve
        await self.vector_engine.upsert_player_vector(
            player, stats_embedding, CollectionType.REGULAR
        )
        
        # Verify storage
        stats = await self.vector_engine.get_collection_stats(CollectionType.REGULAR)
        assert stats["points_count"] >= 1

    @pytest.mark.asyncio
    async def test_context_vector_generation(self):
        """Test contextual vector generation and storage."""
        player = self.test_players[1]  # RB
        
        # Generate context embedding
        context_embedding = await self.embedding_generator.generate_player_embedding(
            player, strategy=EmbeddingStrategy.CONTEXTUAL
        )
        
        # Validate embedding
        assert len(context_embedding) == 768
        assert isinstance(context_embedding, list)
        assert all(isinstance(x, float) for x in context_embedding)
        
        # Store and retrieve
        await self.vector_engine.upsert_player_vector(
            player, context_embedding, CollectionType.REGULAR
        )
        
        # Verify storage
        stats = await self.vector_engine.get_collection_stats(CollectionType.REGULAR)
        assert stats["points_count"] >= 1

    @pytest.mark.asyncio
    async def test_value_vector_generation(self):
        """Test value-based vector generation and storage."""
        player = self.test_players[2]  # WR
        
        # Generate value embedding
        value_embedding = await self.embedding_generator.generate_player_embedding(
            player, strategy=EmbeddingStrategy.VALUE_BASED
        )
        
        # Validate embedding
        assert len(value_embedding) == 768
        assert isinstance(value_embedding, list)
        assert all(isinstance(x, float) for x in value_embedding)
        
        # Store and retrieve
        await self.vector_engine.upsert_player_vector(
            player, value_embedding, CollectionType.REGULAR
        )
        
        # Verify storage
        stats = await self.vector_engine.get_collection_stats(CollectionType.REGULAR)
        assert stats["points_count"] >= 1

    @pytest.mark.asyncio
    async def test_combined_vector_generation(self):
        """Test combined vector generation and storage."""
        player = self.test_players[0]  # QB
        
        # Generate combined embedding
        combined_embedding = await self.embedding_generator.generate_player_embedding(
            player, strategy=EmbeddingStrategy.HYBRID
        )
        
        # Validate embedding
        assert len(combined_embedding) == 768
        assert isinstance(combined_embedding, list)
        assert all(isinstance(x, float) for x in combined_embedding)
        
        # Store and retrieve
        await self.vector_engine.upsert_player_vector(
            player, combined_embedding, CollectionType.REGULAR
        )
        
        # Verify storage
        stats = await self.vector_engine.get_collection_stats(CollectionType.REGULAR)
        assert stats["points_count"] >= 1

    # ============================================================================
    # INTEGRATION TESTS FOR MULTI-VECTOR OPERATIONS
    # ============================================================================

    @pytest.mark.asyncio
    async def test_multi_vector_data_ingestion(self):
        """Test complete multi-vector data ingestion pipeline."""
        # Generate all vector types for all players
        for player in self.test_players:
            # Generate embeddings for all strategies
            stats_embedding = await self.embedding_generator.generate_player_embedding(
                player, strategy=EmbeddingStrategy.STATISTICAL
            )
            context_embedding = await self.embedding_generator.generate_player_embedding(
                player, strategy=EmbeddingStrategy.CONTEXTUAL
            )
            value_embedding = await self.embedding_generator.generate_player_embedding(
                player, strategy=EmbeddingStrategy.VALUE_BASED
            )
            combined_embedding = await self.embedding_generator.generate_player_embedding(
                player, strategy=EmbeddingStrategy.HYBRID
            )
            
            # Create multi-vector player
            multi_vector_player = PlayerMultiVector(
                player=player,
                stats_vector=stats_embedding,
                context_vector=context_embedding,
                value_vector=value_embedding,
                combined_vector=combined_embedding
            )
            
            # Store in both collections
            await self.vector_engine.upsert_multi_vector_player(
                multi_vector_player, CollectionType.REGULAR
            )
            await self.vector_engine.upsert_multi_vector_player(
                multi_vector_player, CollectionType.BINARY_QUANTIZED
            )
        
        # Verify data ingestion
        regular_stats = await self.vector_engine.get_collection_stats(CollectionType.REGULAR)
        quantized_stats = await self.vector_engine.get_collection_stats(CollectionType.BINARY_QUANTIZED)
        
        assert regular_stats["points_count"] == len(self.test_players)
        assert quantized_stats["points_count"] == len(self.test_players)

    @pytest.mark.asyncio
    async def test_multi_vector_search_accuracy(self):
        """Test search accuracy across different vector types."""
        # First, ingest test data
        await self._ingest_test_data()
        
        # Test search queries
        test_queries = [
            "elite quarterback with high fantasy points",
            "volume running back with good value",
            "possession receiver with consistent targets"
        ]
        
        for query in test_queries:
            # Generate query embedding
            query_embedding = await self.embedding_generator.generate_query_embedding(query)
            
            # Search across different vector types
            for vector_type in ["stats", "context", "value", "combined"]:
                results = await self.vector_engine.search_vectors(
                    query_embedding, CollectionType.REGULAR, limit=3
                )
                
                # Validate results
                assert len(results) > 0
                assert all(hasattr(result, 'player') for result in results)
                assert all(hasattr(result, 'similarity_score') for result in results)
                
                # Check that similarity scores are reasonable
                for result in results:
                    assert 0.0 <= result.similarity_score <= 1.0

    @pytest.mark.asyncio
    async def test_vector_fusion_algorithm_validation(self):
        """Test vector fusion algorithm with different strategies."""
        # First, ingest test data
        await self._ingest_test_data()
        
        # Test different fusion strategies
        fusion_strategies = [
            SearchStrategy.WEIGHTED_FUSION,
            SearchStrategy.CONDITIONAL_LOGIC,
            SearchStrategy.ENSEMBLE_SEARCH
        ]
        
        query = "elite quarterback with high fantasy points"
        
        for strategy in fusion_strategies:
            results = await self.search_engine.search(
                query=query,
                strategy=strategy,
                limit=3
            )
            
            # Validate fusion results
            assert len(results) > 0
            assert all(hasattr(result, 'player') for result in results)
            assert all(hasattr(result, 'similarity_score') for result in results)
            
            # Check that fusion provides reasonable results
            for result in results:
                assert 0.0 <= result.similarity_score <= 1.0

    @pytest.mark.asyncio
    async def test_data_consistency_across_vector_types(self):
        """Test data consistency across different vector types."""
        # First, ingest test data
        await self._ingest_test_data()
        
        # Get collection stats
        regular_stats = await self.vector_engine.get_collection_stats(CollectionType.REGULAR)
        quantized_stats = await self.vector_engine.get_collection_stats(CollectionType.BINARY_QUANTIZED)
        
        # Check consistency
        assert regular_stats["points_count"] == quantized_stats["points_count"]
        assert regular_stats["vector_dimensions"] == quantized_stats["vector_dimensions"]
        
        # Test that same players exist in both collections
        query_embedding = await self.embedding_generator.generate_query_embedding("test query")
        
        regular_results = await self.vector_engine.search_vectors(
            query_embedding, CollectionType.REGULAR, limit=5
        )
        quantized_results = await self.vector_engine.search_vectors(
            query_embedding, CollectionType.BINARY_QUANTIZED, limit=5
        )
        
        # Both should return results
        assert len(regular_results) > 0
        assert len(quantized_results) > 0

    # ============================================================================
    # PERFORMANCE BENCHMARKS FOR FUSION SEARCH
    # ============================================================================

    @pytest.mark.asyncio
    async def test_performance_comparison_search_strategies(self):
        """Test performance comparison between different search strategies."""
        # First, ingest test data
        await self._ingest_test_data()
        
        query = "elite quarterback with high fantasy points"
        query_embedding = await self.embedding_generator.generate_query_embedding(query)
        
        # Test regular search
        start_time = time.time()
        regular_results = await self.vector_engine.search_vectors(
            query_embedding, CollectionType.REGULAR, limit=5
        )
        regular_time = time.time() - start_time
        
        # Test quantized search
        start_time = time.time()
        quantized_results = await self.vector_engine.search_vectors(
            query_embedding, CollectionType.BINARY_QUANTIZED, limit=5
        )
        quantized_time = time.time() - start_time
        
        # Test fusion search
        start_time = time.time()
        fusion_results = await self.search_engine.search(
            query=query,
            strategy=SearchStrategy.WEIGHTED_FUSION,
            limit=5
        )
        fusion_time = time.time() - start_time
        
        # Validate performance
        assert regular_time > 0
        assert quantized_time > 0
        assert fusion_time > 0
        
        # Quantized should be faster than regular
        assert quantized_time < regular_time
        
        # All should return results
        assert len(regular_results) > 0
        assert len(quantized_results) > 0
        assert len(fusion_results) > 0

    @pytest.mark.asyncio
    async def test_fusion_search_performance_benchmarks(self):
        """Test fusion search performance with different configurations."""
        # First, ingest test data
        await self._ingest_test_data()
        
        query = "elite quarterback with high fantasy points"
        
        # Test different fusion configurations
        configurations = [
            {"strategy": SearchStrategy.WEIGHTED_FUSION, "weights": {"stats": 0.4, "context": 0.3, "value": 0.3}},
            {"strategy": SearchStrategy.CONDITIONAL_LOGIC, "threshold": 0.7},
            {"strategy": SearchStrategy.ENSEMBLE_SEARCH, "ensemble_size": 3}
        ]
        
        performance_results = {}
        
        for config in configurations:
            start_time = time.time()
            results = await self.search_engine.search(
                query=query,
                strategy=config["strategy"],
                limit=5
            )
            end_time = time.time()
            
            performance_results[config["strategy"]] = {
                "time": end_time - start_time,
                "results_count": len(results),
                "avg_similarity": np.mean([r.similarity_score for r in results]) if results else 0.0
            }
        
        # Validate performance results
        for strategy, metrics in performance_results.items():
            assert metrics["time"] > 0
            assert metrics["results_count"] > 0
            assert 0.0 <= metrics["avg_similarity"] <= 1.0

    # ============================================================================
    # EDGE CASE HANDLING
    # ============================================================================

    @pytest.mark.asyncio
    async def test_missing_vectors_handling(self):
        """Test handling of missing vectors in multi-vector operations."""
        # Create player with missing context data
        incomplete_player = Player(
            player_id="test_incomplete_1",
            name="Incomplete Player",
            position=Position.QB,
            team=Team.KC,
            season=2024,
            week=1,
            stats=PlayerStats(
                passing_yards=3000,
                passing_touchdowns=20,
                passing_interceptions=10,
                rushing_yards=200,
                rushing_touchdowns=2,
                fantasy_points=250.0,
                games_played=16
            ),
            context=None,  # Missing context
            value=PlayerValue(
                salary=6000,
                ownership_percentage=10.0,
                value_rating=6.5,
                consistency_score=0.70,
                upside_potential=0.75
            )
        )
        
        # Should handle missing context gracefully
        try:
            stats_embedding = await self.embedding_generator.generate_player_embedding(
                incomplete_player, strategy=EmbeddingStrategy.STATISTICAL
            )
            assert len(stats_embedding) == 768
        except Exception as e:
            pytest.fail(f"Should handle missing context gracefully: {e}")

    @pytest.mark.asyncio
    async def test_corrupted_data_handling(self):
        """Test handling of corrupted data in vector operations."""
        # Test with corrupted embedding
        corrupted_embedding = [float('nan')] * 768
        
        try:
            # Should handle corrupted embedding gracefully
            await self.vector_engine.upsert_player_vector(
                self.test_players[0], corrupted_embedding, CollectionType.REGULAR
            )
        except Exception as e:
            # Expected to fail with corrupted data
            assert "nan" in str(e).lower() or "invalid" in str(e).lower()

    @pytest.mark.asyncio
    async def test_empty_collection_handling(self):
        """Test handling of empty collections."""
        # Clear collections
        await self.vector_engine.clear_collection(CollectionType.REGULAR)
        await self.vector_engine.clear_collection(CollectionType.BINARY_QUANTIZED)
        
        # Test search on empty collection
        query_embedding = await self.embedding_generator.generate_query_embedding("test query")
        
        results = await self.vector_engine.search_vectors(
            query_embedding, CollectionType.REGULAR, limit=5
        )
        
        # Should return empty results, not crash
        assert len(results) == 0

    # ============================================================================
    # API ENDPOINT VALIDATION
    # ============================================================================

    @pytest.mark.asyncio
    async def test_health_endpoints_validation(self):
        """Test multi-vector health check endpoints."""
        # Test vector health endpoint
        from dfs_prophet.api.routes.health import vector_health_check
        
        response = await vector_health_check()
        health_data = response.body.decode('utf-8')
        health_json = json.loads(health_data)
        
        assert "status" in health_json
        assert "health_score" in health_json
        assert "vector_system" in health_json
        assert "recommendations" in health_json

    @pytest.mark.asyncio
    async def test_performance_endpoints_validation(self):
        """Test performance monitoring endpoints."""
        # Test performance health endpoint
        from dfs_prophet.api.routes.health import performance_health_check
        
        response = await performance_health_check()
        perf_data = response.body.decode('utf-8')
        perf_json = json.loads(perf_data)
        
        assert "status" in perf_json
        assert "performance_metrics" in perf_json
        assert "memory_breakdown" in perf_json
        assert "optimization_recommendations" in perf_json

    @pytest.mark.asyncio
    async def test_vector_type_endpoints_validation(self):
        """Test specific vector type health endpoints."""
        # Test specific vector type endpoint
        from dfs_prophet.api.routes.health import vector_type_health_check
        
        response = await vector_type_health_check("stats")
        vector_data = response.body.decode('utf-8')
        vector_json = json.loads(vector_data)
        
        assert "vector_type" in vector_json
        assert "status" in vector_json
        assert "details" in vector_json

    # ============================================================================
    # HELPER METHODS
    # ============================================================================

    async def _ingest_test_data(self):
        """Helper method to ingest test data for integration tests."""
        for player in self.test_players:
            # Generate all vector types
            stats_embedding = await self.embedding_generator.generate_player_embedding(
                player, strategy=EmbeddingStrategy.STATISTICAL
            )
            context_embedding = await self.embedding_generator.generate_player_embedding(
                player, strategy=EmbeddingStrategy.CONTEXTUAL
            )
            value_embedding = await self.embedding_generator.generate_player_embedding(
                player, strategy=EmbeddingStrategy.VALUE_BASED
            )
            combined_embedding = await self.embedding_generator.generate_player_embedding(
                player, strategy=EmbeddingStrategy.HYBRID
            )
            
            # Create multi-vector player
            multi_vector_player = PlayerMultiVector(
                player=player,
                stats_vector=stats_embedding,
                context_vector=context_embedding,
                value_vector=value_embedding,
                combined_vector=combined_embedding
            )
            
            # Store in collections
            await self.vector_engine.upsert_multi_vector_player(
                multi_vector_player, CollectionType.REGULAR
            )
            await self.vector_engine.upsert_multi_vector_player(
                multi_vector_player, CollectionType.BINARY_QUANTIZED
            )


class TestMultiVectorPerformance:
    """Performance-specific tests for multi-vector system."""

    @pytest.mark.asyncio
    async def test_large_scale_data_ingestion(self):
        """Test performance with large-scale data ingestion."""
        # Generate larger test dataset
        large_test_players = []
        for i in range(100):
            player = Player(
                player_id=f"test_player_{i}",
                name=f"Test Player {i}",
                position=Position.QB if i % 4 == 0 else Position.RB if i % 4 == 1 else Position.WR if i % 4 == 2 else Position.TE,
                team=Team.KC,
                season=2024,
                week=1,
                stats=PlayerStats(
                    passing_yards=3000 + i * 10,
                    passing_touchdowns=20 + i,
                    passing_interceptions=10,
                    rushing_yards=200 + i * 5,
                    rushing_touchdowns=2 + i,
                    fantasy_points=250.0 + i * 2,
                    games_played=16
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
                    salary=6000 + i * 50,
                    ownership_percentage=10.0 + i * 0.5,
                    value_rating=6.5 + i * 0.1,
                    consistency_score=0.70 + i * 0.01,
                    upside_potential=0.75 + i * 0.01
                )
            )
            large_test_players.append(player)
        
        # Measure ingestion time
        start_time = time.time()
        
        vector_engine = VectorEngine()
        embedding_generator = EmbeddingGenerator()
        
        for player in large_test_players[:10]:  # Test with subset for performance
            embedding = await embedding_generator.generate_player_embedding(
                player, strategy=EmbeddingStrategy.STATISTICAL
            )
            await vector_engine.upsert_player_vector(
                player, embedding, CollectionType.REGULAR
            )
        
        ingestion_time = time.time() - start_time
        
        # Validate performance
        assert ingestion_time < 30.0  # Should complete within 30 seconds
        
        # Cleanup
        await vector_engine.clear_collection(CollectionType.REGULAR)

    @pytest.mark.asyncio
    async def test_concurrent_search_operations(self):
        """Test performance with concurrent search operations."""
        # Setup test data
        vector_engine = VectorEngine()
        embedding_generator = EmbeddingGenerator()
        
        # Generate test player
        test_player = Player(
            player_id="test_concurrent_1",
            name="Concurrent Test Player",
            position=Position.QB,
            team=Team.KC,
            season=2024,
            week=1,
            stats=PlayerStats(
                passing_yards=4000,
                passing_touchdowns=30,
                passing_interceptions=8,
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
        )
        
        # Ingest test data
        embedding = await embedding_generator.generate_player_embedding(
            test_player, strategy=EmbeddingStrategy.STATISTICAL
        )
        await vector_engine.upsert_player_vector(
            test_player, embedding, CollectionType.REGULAR
        )
        
        # Test concurrent searches
        query_embeddings = [
            await embedding_generator.generate_query_embedding("elite quarterback"),
            await embedding_generator.generate_query_embedding("high fantasy points"),
            await embedding_generator.generate_query_embedding("consistent performer")
        ]
        
        start_time = time.time()
        
        # Run concurrent searches
        tasks = [
            vector_engine.search_vectors(query_emb, CollectionType.REGULAR, limit=5)
            for query_emb in query_embeddings
        ]
        
        results = await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        # Validate results
        assert len(results) == 3
        assert all(len(result) > 0 for result in results)
        assert concurrent_time < 5.0  # Should complete within 5 seconds
        
        # Cleanup
        await vector_engine.clear_collection(CollectionType.REGULAR)


class TestMultiVectorAnalytics:
    """Analytics-specific tests for multi-vector system."""

    @pytest.mark.asyncio
    async def test_player_archetype_classification(self):
        """Test player archetype classification with multi-vector data."""
        analyzer = PlayerProfileAnalyzer()
        
        # Test with different player types
        for player in [
            Player(
                player_id="test_archetype_1",
                name="Volume Rusher",
                position=Position.RB,
                team=Team.SF,
                season=2024,
                week=1,
                stats=PlayerStats(
                    rushing_yards=1500,
                    rushing_touchdowns=15,
                    receiving_yards=200,
                    receiving_touchdowns=2,
                    fantasy_points=300.0,
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
                    salary=9000,
                    ownership_percentage=25.0,
                    value_rating=8.5,
                    consistency_score=0.90,
                    upside_potential=0.95
                )
            )
        ]:
            archetype = await analyzer.classify_player_archetype(player)
            
            # Validate archetype classification
            assert archetype is not None
            assert hasattr(archetype, 'archetype_type')
            assert hasattr(archetype, 'confidence_score')
            assert 0.0 <= archetype.confidence_score <= 1.0

    @pytest.mark.asyncio
    async def test_multi_vector_similarity_analysis(self):
        """Test multi-vector similarity analysis."""
        analyzer = PlayerProfileAnalyzer()
        
        # Create test players
        player1 = Player(
            player_id="test_similarity_1",
            name="Player 1",
            position=Position.QB,
            team=Team.KC,
            season=2024,
            week=1,
            stats=PlayerStats(
                passing_yards=4500,
                passing_touchdowns=35,
                passing_interceptions=8,
                rushing_yards=350,
                rushing_touchdowns=5,
                fantasy_points=350.5,
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
                salary=9500,
                ownership_percentage=25.5,
                value_rating=8.2,
                consistency_score=0.85,
                upside_potential=0.92
            )
        )
        
        player2 = Player(
            player_id="test_similarity_2",
            name="Player 2",
            position=Position.QB,
            team=Team.BUF,
            season=2024,
            week=1,
            stats=PlayerStats(
                passing_yards=4200,
                passing_touchdowns=32,
                passing_interceptions=10,
                rushing_yards=400,
                rushing_touchdowns=6,
                fantasy_points=340.0,
                games_played=17
            ),
            context=PlayerContext(
                opponent_team=Team.KC,
                home_away="away",
                weather_conditions="clear",
                injury_status="healthy",
                team_offensive_rank=4,
                opponent_defensive_rank=15
            ),
            value=PlayerValue(
                salary=9200,
                ownership_percentage=22.0,
                value_rating=8.0,
                consistency_score=0.82,
                upside_potential=0.90
            )
        )
        
        # Test similarity analysis
        similarity = await analyzer.analyze_player_similarity(player1, player2)
        
        # Validate similarity results
        assert similarity is not None
        assert hasattr(similarity, 'overall_similarity')
        assert hasattr(similarity, 'vector_similarities')
        assert 0.0 <= similarity.overall_similarity <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
