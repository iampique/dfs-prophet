#!/usr/bin/env python3
"""
DFS Prophet Multi-Vector Demo

Complete working example demonstrating all multi-vector features:
- Player data creation and ingestion
- Multi-vector embedding generation
- Vector storage and retrieval
- Advanced search capabilities
- Performance comparison
- Health monitoring

Usage:
    python examples/multi_vector_demo.py
"""

import asyncio
import sys
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import random
import time

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dfs_prophet.data.models.player import (
    Player, PlayerBase, PlayerStats, PlayerDFS, Position, Team,
    PlayerStatVector, PlayerContextVector, PlayerValueVector, PlayerMultiVector
)
from dfs_prophet.core.vector_engine import VectorEngine, CollectionType
from dfs_prophet.core.embedding_generator import EmbeddingGenerator, EmbeddingStrategy
from dfs_prophet.analytics.profile_analyzer import PlayerProfileAnalyzer, ArchetypeType
from dfs_prophet.search.advanced_search import AdvancedSearchEngine, SearchStrategy
from dfs_prophet.monitoring.vector_performance import VectorPerformanceMonitor
from dfs_prophet.config import get_settings


class MultiVectorDemo:
    """Complete multi-vector demonstration class."""
    
    def __init__(self):
        self.settings = get_settings()
        self.vector_engine = VectorEngine()
        self.embedding_generator = EmbeddingGenerator()
        self.profile_analyzer = PlayerProfileAnalyzer()
        self.search_engine = AdvancedSearchEngine()
        self.performance_monitor = VectorPerformanceMonitor()
        
        # Demo data
        self.demo_players = []
        self.search_results = {}
        
    async def run_complete_demo(self) -> Dict[str, Any]:
        """Run the complete multi-vector demonstration."""
        
        print("🚀 DFS Prophet Multi-Vector Demo")
        print("=" * 60)
        print("Demonstrating complete multi-vector functionality...")
        print()
        
        demo_results = {
            "start_time": datetime.now().isoformat(),
            "players_created": 0,
            "vectors_generated": 0,
            "searches_performed": 0,
            "performance_metrics": {},
            "errors": []
        }
        
        try:
            # 1. Initialize system
            print("1️⃣ Initializing system...")
            await self._initialize_system()
            
            # 2. Create demo players
            print("\n2️⃣ Creating demo players...")
            await self._create_demo_players()
            demo_results["players_created"] = len(self.demo_players)
            
            # 3. Generate multi-vector embeddings
            print("\n3️⃣ Generating multi-vector embeddings...")
            await self._generate_multi_vector_embeddings()
            demo_results["vectors_generated"] = len(self.demo_players) * 4  # 4 vector types per player
            
            # 4. Demonstrate search capabilities
            print("\n4️⃣ Demonstrating search capabilities...")
            await self._demonstrate_search_capabilities()
            demo_results["searches_performed"] = len(self.search_results)
            
            # 5. Performance comparison
            print("\n5️⃣ Performance comparison...")
            performance_metrics = await self._performance_comparison()
            demo_results["performance_metrics"] = performance_metrics
            
            # 6. Player analytics
            print("\n6️⃣ Player analytics...")
            await self._demonstrate_player_analytics()
            
            # 7. Health monitoring
            print("\n7️⃣ Health monitoring...")
            await self._demonstrate_health_monitoring()
            
            # 8. Advanced features
            print("\n8️⃣ Advanced features...")
            await self._demonstrate_advanced_features()
            
            print("\n✅ Demo completed successfully!")
            
        except Exception as e:
            error_msg = f"Demo failed: {str(e)}"
            print(f"\n❌ {error_msg}")
            demo_results["errors"].append(error_msg)
        
        finally:
            # Cleanup
            print("\n🧹 Cleaning up...")
            await self._cleanup()
            
        demo_results["end_time"] = datetime.now().isoformat()
        return demo_results
    
    async def _initialize_system(self):
        """Initialize the multi-vector system."""
        
        # Initialize vector collections
        await self.vector_engine.initialize_collections()
        
        # Verify system health
        health_status = await self._check_system_health()
        if health_status["overall_status"] != "healthy":
            raise Exception(f"System not healthy: {health_status}")
        
        print("   ✅ System initialized successfully")
    
    async def _create_demo_players(self):
        """Create comprehensive demo player data."""
        
        # Elite QBs
        self.demo_players.extend([
            Player(
                base=PlayerBase(
                    player_id="qb_mahomes_2024_w1",
                    name="Patrick Mahomes",
                    position=Position.QB,
                    team=Team.KC,
                    season=2024,
                    week=1
                ),
                stats=PlayerStats(
                    fantasy_points=28.5,
                    passing_yards=320,
                    rushing_yards=45,
                    passing_touchdowns=3,
                    rushing_touchdowns=1,
                    passing_interceptions=0,
                    games_played=1
                ),
                dfs=PlayerDFS(
                    salary=8500,
                    projected_points=26.2,
                    ownership_percentage=15.2,
                    value_rating=3.35,
                    weather_conditions="Clear, 72°F, 5mph wind",
                    injury_status="Healthy",
                    game_total=48.5,
                    team_spread=-3.5
                )
            ),
            Player(
                base=PlayerBase(
                    player_id="qb_allen_2024_w1",
                    name="Josh Allen",
                    position=Position.QB,
                    team=Team.BUF,
                    season=2024,
                    week=1
                ),
                stats=PlayerStats(
                    fantasy_points=26.8,
                    passing_yards=295,
                    rushing_yards=65,
                    passing_touchdowns=2,
                    rushing_touchdowns=1,
                    passing_interceptions=1,
                    games_played=1
                ),
                dfs=PlayerDFS(
                    salary=8200,
                    projected_points=25.1,
                    ownership_percentage=12.8,
                    value_rating=3.27,
                    weather_conditions="Clear, 68°F, 8mph wind",
                    injury_status="Healthy",
                    game_total=52.0,
                    team_spread=2.5
                )
            )
        ])
        
        # Elite RBs
        self.demo_players.extend([
            Player(
                base=PlayerBase(
                    player_id="rb_mccaffrey_2024_w1",
                    name="Christian McCaffrey",
                    position=Position.RB,
                    team=Team.SF,
                    season=2024,
                    week=1
                ),
                stats=PlayerStats(
                    fantasy_points=24.2,
                    rushing_yards=120,
                    rushing_touchdowns=1,
                    receiving_yards=45,
                    receiving_touchdowns=1,
                    targets=8,
                    receptions=6,
                    games_played=1
                ),
                dfs=PlayerDFS(
                    salary=9200,
                    projected_points=22.0,
                    ownership_percentage=18.5,
                    value_rating=2.39,
                    weather_conditions="Clear, 75°F, 3mph wind",
                    injury_status="Healthy",
                    game_total=45.5,
                    team_spread=-6.5
                )
            ),
            Player(
                base=PlayerBase(
                    player_id="rb_taylor_2024_w1",
                    name="Jonathan Taylor",
                    position=Position.RB,
                    team=Team.IND,
                    season=2024,
                    week=1
                ),
                stats=PlayerStats(
                    fantasy_points=22.1,
                    rushing_yards=95,
                    rushing_touchdowns=2,
                    receiving_yards=25,
                    receiving_touchdowns=0,
                    targets=4,
                    receptions=3,
                    games_played=1
                ),
                dfs=PlayerDFS(
                    salary=7800,
                    projected_points=19.8,
                    ownership_percentage=14.2,
                    value_rating=2.54,
                    weather_conditions="Clear, 70°F, 6mph wind",
                    injury_status="Healthy",
                    game_total=44.0,
                    team_spread=-2.0
                )
            )
        ])
        
        # Elite WRs
        self.demo_players.extend([
            Player(
                base=PlayerBase(
                    player_id="wr_adams_2024_w1",
                    name="Davante Adams",
                    position=Position.WR,
                    team=Team.LV,
                    season=2024,
                    week=1
                ),
                stats=PlayerStats(
                    fantasy_points=21.8,
                    receiving_yards=125,
                    receiving_touchdowns=1,
                    targets=12,
                    receptions=8,
                    games_played=1
                ),
                dfs=PlayerDFS(
                    salary=7800,
                    projected_points=18.5,
                    ownership_percentage=12.1,
                    value_rating=2.37,
                    weather_conditions="Clear, 78°F, 4mph wind",
                    injury_status="Healthy",
                    game_total=46.5,
                    team_spread=-1.5
                )
            ),
            Player(
                base=PlayerBase(
                    player_id="wr_hill_2024_w1",
                    name="Tyreek Hill",
                    position=Position.WR,
                    team=Team.MIA,
                    season=2024,
                    week=1
                ),
                stats=PlayerStats(
                    fantasy_points=23.5,
                    receiving_yards=145,
                    receiving_touchdowns=1,
                    targets=11,
                    receptions=7,
                    games_played=1
                ),
                dfs=PlayerDFS(
                    salary=8200,
                    projected_points=20.2,
                    ownership_percentage=16.8,
                    value_rating=2.46,
                    weather_conditions="Clear, 82°F, 7mph wind",
                    injury_status="Healthy",
                    game_total=49.0,
                    team_spread=-4.0
                )
            )
        ])
        
        print(f"   ✅ Created {len(self.demo_players)} demo players")
    
    async def _generate_multi_vector_embeddings(self):
        """Generate multi-vector embeddings for all demo players."""
        
        for i, player in enumerate(self.demo_players):
            print(f"   Generating embeddings for {player.name} ({i+1}/{len(self.demo_players)})...")
            
            try:
                # Generate statistical embedding
                stats_embedding = await self.embedding_generator.generate_player_embedding(
                    player, strategy=EmbeddingStrategy.STATISTICAL
                )
                
                # Generate contextual embedding
                context_embedding = await self.embedding_generator.generate_player_embedding(
                    player, strategy=EmbeddingStrategy.CONTEXTUAL
                )
                
                # Generate value embedding
                value_embedding = await self.embedding_generator.generate_player_embedding(
                    player, strategy=EmbeddingStrategy.VALUE
                )
                
                # Generate combined embedding
                combined_embedding = await self.embedding_generator.generate_player_embedding(
                    player, strategy=EmbeddingStrategy.COMBINED
                )
                
                # Store multi-vector player
                await self.vector_engine.upsert_multi_vector_player(
                    player, {
                        "stats": stats_embedding,
                        "context": context_embedding,
                        "value": value_embedding,
                        "combined": combined_embedding
                    }
                )
                
                print(f"      ✅ {player.name}: All vectors generated and stored")
                
            except Exception as e:
                print(f"      ❌ {player.name}: Error generating embeddings - {str(e)}")
    
    async def _demonstrate_search_capabilities(self):
        """Demonstrate various search capabilities."""
        
        demo_queries = [
            ("high passing yards quarterback", ["stats"]),
            ("favorable matchup home game", ["context"]),
            ("low ownership high value", ["value"]),
            ("elite quarterback with good matchup", ["stats", "context"]),
            ("comprehensive player analysis", ["stats", "context", "value"]),
            ("rushing touchdown leader", ["stats"]),
            ("weather impact analysis", ["context"]),
            ("salary efficiency play", ["value"])
        ]
        
        for query, vector_types in demo_queries:
            print(f"   Searching: '{query}' with {vector_types} vectors...")
            
            try:
                # Generate query embedding
                query_embedding = await self.embedding_generator.generate_query_embedding(query)
                
                # Perform search
                results = await self.vector_engine.search_vectors(
                    query_embedding,
                    CollectionType.REGULAR,
                    vector_types=vector_types,
                    limit=3
                )
                
                self.search_results[query] = {
                    "vector_types": vector_types,
                    "results": results,
                    "count": len(results)
                }
                
                print(f"      ✅ Found {len(results)} results")
                
                # Display top result
                if results:
                    top_result = results[0]
                    print(f"      Top: {top_result.player.name} (Score: {top_result.similarity_score:.3f})")
                
            except Exception as e:
                print(f"      ❌ Search failed: {str(e)}")
    
    async def _performance_comparison(self) -> Dict[str, Any]:
        """Compare performance between regular and binary quantized search."""
        
        print("   Comparing regular vs binary quantized search...")
        
        test_query = "elite quarterback"
        query_embedding = await self.embedding_generator.generate_query_embedding(test_query)
        
        performance_metrics = {}
        
        # Regular search
        start_time = time.time()
        regular_results = await self.vector_engine.search_vectors(
            query_embedding,
            CollectionType.REGULAR,
            vector_types=["combined"],
            limit=5
        )
        regular_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Binary quantized search
        start_time = time.time()
        quantized_results = await self.vector_engine.search_vectors(
            query_embedding,
            CollectionType.BINARY_QUANTIZED,
            vector_types=["combined"],
            limit=5
        )
        quantized_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Calculate metrics
        speedup = regular_time / quantized_time if quantized_time > 0 else 0
        
        # Calculate accuracy (simple overlap metric)
        regular_players = {r.player.player_id for r in regular_results}
        quantized_players = {r.player.player_id for r in quantized_results}
        overlap = len(regular_players.intersection(quantized_players)) / len(regular_players) if regular_players else 0
        
        performance_metrics = {
            "regular_search_time_ms": round(regular_time, 2),
            "quantized_search_time_ms": round(quantized_time, 2),
            "speedup_factor": round(speedup, 2),
            "accuracy_overlap": round(overlap, 3),
            "regular_results_count": len(regular_results),
            "quantized_results_count": len(quantized_results)
        }
        
        print(f"      ✅ Regular: {regular_time:.2f}ms, Quantized: {quantized_time:.2f}ms")
        print(f"      ✅ Speedup: {speedup:.2f}x, Accuracy: {overlap:.1%}")
        
        return performance_metrics
    
    async def _demonstrate_player_analytics(self):
        """Demonstrate player analytics capabilities."""
        
        if not self.demo_players:
            print("   ⚠️ No demo players available for analytics")
            return
        
        player = self.demo_players[0]  # Use first player for demo
        
        print(f"   Analyzing player: {player.name}...")
        
        try:
            # Player archetype classification
            archetype = await self.profile_analyzer.classify_player_archetype(player)
            print(f"      Archetype: {archetype.archetype_type.value}")
            print(f"      Confidence: {archetype.confidence:.2f}")
            
            # Vector contribution analysis
            vector_analysis = await self.profile_analyzer.analyze_vector_contributions(player)
            print(f"      Vector contributions: {vector_analysis.contribution_breakdown}")
            
            # Similarity analysis
            similar_players = await self.profile_analyzer.find_similar_players(player, limit=3)
            print(f"      Similar players: {[p.name for p in similar_players]}")
            
            print("      ✅ Player analytics completed")
            
        except Exception as e:
            print(f"      ❌ Analytics failed: {str(e)}")
    
    async def _demonstrate_health_monitoring(self):
        """Demonstrate health monitoring capabilities."""
        
        print("   Checking system health...")
        
        try:
            # Track some performance metrics
            await self.performance_monitor.track_search_metrics(
                vector_type="combined",
                collection_type="regular",
                latency_ms=25.5,
                result_count=5,
                accuracy_score=0.85
            )
            
            # Get performance summary
            summary = await self.performance_monitor.get_performance_summary()
            print(f"      Total searches: {summary.total_searches}")
            print(f"      Average latency: {summary.average_latency:.2f}ms")
            print(f"      Active alerts: {len(summary.active_alerts)}")
            
            # Get dashboard data
            dashboard = await self.performance_monitor.get_dashboard_data()
            print(f"      System health score: {dashboard.health_score:.1f}")
            
            print("      ✅ Health monitoring active")
            
        except Exception as e:
            print(f"      ❌ Health monitoring failed: {str(e)}")
    
    async def _demonstrate_advanced_features(self):
        """Demonstrate advanced search features."""
        
        print("   Testing advanced search features...")
        
        try:
            # Advanced search with custom weights
            query = "elite quarterback with favorable matchup"
            query_embedding = await self.embedding_generator.generate_query_embedding(query)
            
            # Weighted fusion search
            weighted_results = await self.search_engine.search(
                query_embedding,
                strategy=SearchStrategy.WEIGHTED_FUSION,
                weights={"stats": 0.4, "context": 0.3, "value": 0.3},
                limit=3
            )
            
            print(f"      Weighted fusion: {len(weighted_results)} results")
            
            # Conditional logic search
            conditional_results = await self.search_engine.search(
                query_embedding,
                strategy=SearchStrategy.CONDITIONAL_LOGIC,
                limit=3
            )
            
            print(f"      Conditional logic: {len(conditional_results)} results")
            
            # Ensemble search
            ensemble_results = await self.search_engine.search(
                query_embedding,
                strategy=SearchStrategy.ENSEMBLE,
                limit=3
            )
            
            print(f"      Ensemble: {len(ensemble_results)} results")
            
            print("      ✅ Advanced features tested")
            
        except Exception as e:
            print(f"      ❌ Advanced features failed: {str(e)}")
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """Check overall system health."""
        
        try:
            # Check vector collections
            regular_stats = await self.vector_engine.get_collection_stats(CollectionType.REGULAR)
            quantized_stats = await self.vector_engine.get_collection_stats(CollectionType.BINARY_QUANTIZED)
            
            health_status = {
                "overall_status": "healthy",
                "regular_collection": {
                    "status": "healthy" if regular_stats["points_count"] > 0 else "empty",
                    "points_count": regular_stats["points_count"]
                },
                "quantized_collection": {
                    "status": "healthy" if quantized_stats["points_count"] > 0 else "empty",
                    "points_count": quantized_stats["points_count"]
                }
            }
            
            return health_status
            
        except Exception as e:
            return {
                "overall_status": "unhealthy",
                "error": str(e)
            }
    
    async def _cleanup(self):
        """Clean up demo data."""
        
        try:
            # Clear collections
            await self.vector_engine.clear_collection(CollectionType.REGULAR)
            await self.vector_engine.clear_collection(CollectionType.BINARY_QUANTIZED)
            print("   ✅ Demo data cleaned up")
        except Exception as e:
            print(f"   ⚠️ Cleanup warning: {str(e)}")


async def main():
    """Main demo function."""
    
    demo = MultiVectorDemo()
    results = await demo.run_complete_demo()
    
    print("\n" + "=" * 60)
    print("📊 Demo Summary")
    print("=" * 60)
    print(f"Players Created: {results['players_created']}")
    print(f"Vectors Generated: {results['vectors_generated']}")
    print(f"Searches Performed: {results['searches_performed']}")
    
    if results['performance_metrics']:
        metrics = results['performance_metrics']
        print(f"\nPerformance Metrics:")
        print(f"  Regular Search: {metrics['regular_search_time_ms']}ms")
        print(f"  Quantized Search: {metrics['quantized_search_time_ms']}ms")
        print(f"  Speedup: {metrics['speedup_factor']}x")
        print(f"  Accuracy: {metrics['accuracy_overlap']:.1%}")
    
    if results['errors']:
        print(f"\nErrors Encountered: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  - {error}")
    
    print(f"\nDemo Duration: {results['start_time']} to {results['end_time']}")


if __name__ == "__main__":
    asyncio.run(main())
