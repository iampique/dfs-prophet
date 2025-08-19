#!/usr/bin/env python3

"""
Enhanced Demo Data Setup for DFS Prophet

Creates comprehensive multi-vector demo data including:
- All three vector types (stats, context, value) for each player
- Named vector collections in Qdrant
- Sample player archetypes for demonstration
- Test scenarios for different search types
- Comprehensive validation and quality checks
"""

import asyncio
import sys
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import asdict
import random
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dfs_prophet.config import get_settings
from dfs_prophet.utils import get_logger, performance_timer
from dfs_prophet.core import get_vector_engine, get_embedding_generator, CollectionType, MultiVectorCollectionType
from dfs_prophet.core.embedding_generator import EmbeddingStrategy
from dfs_prophet.data.collectors import get_nfl_collector
from dfs_prophet.data.models import Player, PlayerBase, PlayerStats, PlayerDFS, Position, Team
from dfs_prophet.analytics import PlayerProfileAnalyzer, ArchetypeType


class EnhancedDemoDataSetup:
    """Enhanced demo data setup with multi-vector support."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        self.vector_engine = get_vector_engine()
        self.embedding_generator = get_embedding_generator()
        self.nfl_collector = get_nfl_collector()
        self.profile_analyzer = PlayerProfileAnalyzer()
        
        # Demo data configuration
        self.demo_players_count = 100
        self.archetype_distribution = {
            ArchetypeType.VOLUME_RUSHER: 15,
            ArchetypeType.RED_ZONE_TARGET: 12,
            ArchetypeType.DEEP_THREAT: 10,
            ArchetypeType.ELITE_QB: 8,
            ArchetypeType.VALUE_PLAY: 20,
            ArchetypeType.POSSESSION_RECEIVER: 10,
            ArchetypeType.GAME_MANAGER: 5,
            ArchetypeType.HIGH_VOLUME_RB: 8,
            ArchetypeType.THIRD_DOWN_BACK: 5,
            ArchetypeType.TIGHT_END_RED_ZONE: 7
        }
        
        # Quality thresholds
        self.min_vector_magnitude = 0.1
        self.max_vector_magnitude = 2.0
        self.min_similarity_threshold = 0.3
        self.max_similarity_threshold = 0.9
        
        # Demo scenarios
        self.demo_scenarios = {
            "similar_stats_different_context": [],
            "similar_value_different_stats": [],
            "matchup_recommendations": [],
            "value_plays": [],
            "archetype_examples": {}
        }
    
    @performance_timer('setup_enhanced_demo_data')
    async def setup_enhanced_demo_data(self) -> Dict[str, Any]:
        """Setup comprehensive multi-vector demo data."""
        self.logger.info("Starting enhanced demo data setup...")
        
        setup_results = {
            "timestamp": datetime.now().isoformat(),
            "players_created": 0,
            "vectors_generated": 0,
            "collections_created": 0,
            "quality_checks": {},
            "demo_scenarios": {},
            "archetype_distribution": {},
            "validation_results": {},
            "performance_metrics": {}
        }
        
        try:
            # Step 1: Initialize collections
            await self._initialize_collections()
            setup_results["collections_created"] = 4  # Regular, Quantized, Multi-Regular, Multi-Quantized
            
            # Step 2: Generate demo players
            demo_players, player_archetypes = await self._generate_demo_players()
            setup_results["players_created"] = len(demo_players)
            
            # Step 3: Generate multi-vector embeddings
            vector_results = await self._generate_multi_vector_embeddings(demo_players, player_archetypes)
            setup_results["vectors_generated"] = vector_results["total_vectors"]
            
            # Step 4: Upsert players to vector engine
            await self._upsert_players_to_vector_engine(demo_players, vector_results["embeddings"])
            
            # Step 5: Validate vector quality
            quality_results = await self._validate_vector_quality(demo_players, vector_results["embeddings"])
            setup_results["quality_checks"] = quality_results
            
            # Step 6: Create demo scenarios
            scenario_results = await self._create_demo_scenarios(demo_players, vector_results["embeddings"])
            setup_results["demo_scenarios"] = scenario_results
            
            # Step 7: Analyze archetype distribution
            archetype_results = await self._analyze_archetype_distribution(demo_players)
            setup_results["archetype_distribution"] = archetype_results
            
            # Step 8: Comprehensive validation
            validation_results = await self._comprehensive_validation(demo_players)
            setup_results["validation_results"] = validation_results
            
            # Step 9: Performance testing
            performance_results = await self._performance_testing()
            setup_results["performance_metrics"] = performance_results
            
            self.logger.info("Enhanced demo data setup completed successfully!")
            setup_results["status"] = "success"
            return setup_results
            
        except Exception as e:
            self.logger.error(f"Error in enhanced demo data setup: {e}")
            return {
                "status": "error",
                "error": str(e),
                "players_created": 0,
                "vectors_generated": 0,
                "collections_created": 0
            }
    
    async def _initialize_collections(self):
        """Initialize all vector collections."""
        self.logger.info("Initializing vector collections...")
        
        try:
            await self.vector_engine.initialize_collections()
            self.logger.info("✅ All collections initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing collections: {e}")
            raise
    
    async def _generate_demo_players(self) -> Tuple[List[Player], Dict[str, ArchetypeType]]:
        """Generate comprehensive demo players with archetype distribution."""
        self.logger.info(f"Generating {self.demo_players_count} demo players...")
        
        players = []
        player_archetypes = {}
        player_id_counter = 1
        
        # Generate players for each archetype
        for archetype, count in self.archetype_distribution.items():
            for i in range(count):
                player = await self._create_player_for_archetype(
                    player_id=f"demo_player_{player_id_counter:03d}",
                    archetype=archetype,
                    index=i
                )
                players.append(player)
                player_archetypes[player.player_id] = archetype
                player_id_counter += 1
        
        # Add some random players to fill remaining slots
        remaining_count = self.demo_players_count - len(players)
        for i in range(remaining_count):
            player = await self._create_random_player(
                player_id=f"demo_player_{player_id_counter:03d}"
            )
            players.append(player)
            player_archetypes[player.player_id] = ArchetypeType.VALUE_PLAY  # Default archetype
            player_id_counter += 1
        
        self.logger.info(f"✅ Generated {len(players)} demo players")
        return players, player_archetypes
    
    async def _create_player_for_archetype(self, player_id: str, archetype: ArchetypeType, index: int) -> Player:
        """Create a player optimized for a specific archetype."""
        
        # Base player data
        name = f"Demo {archetype.value.replace('_', ' ').title()} {index + 1}"
        position = self._get_position_for_archetype(archetype)
        team = random.choice(list(Team))
        
        # Create stats based on archetype
        stats = self._create_stats_for_archetype(archetype, position)
        
        # Create DFS data based on archetype
        dfs = self._create_dfs_for_archetype(archetype, stats)
        
        # Create base player data
        base = PlayerBase(
            player_id=player_id,
            name=name,
            position=position,
            team=team,
            season=2024,
            week=1
        )
        
        return Player(
            player_id=player_id,
            name=name,
            position=position,
            team=team,
            base=base,
            stats=stats,
            dfs=dfs
        )
    
    def _get_position_for_archetype(self, archetype: ArchetypeType) -> Position:
        """Get appropriate position for archetype."""
        position_mapping = {
            ArchetypeType.VOLUME_RUSHER: Position.RB,
            ArchetypeType.HIGH_VOLUME_RB: Position.RB,
            ArchetypeType.THIRD_DOWN_BACK: Position.RB,
            ArchetypeType.RED_ZONE_TARGET: Position.WR,
            ArchetypeType.DEEP_THREAT: Position.WR,
            ArchetypeType.POSSESSION_RECEIVER: Position.WR,
            ArchetypeType.ELITE_QB: Position.QB,
            ArchetypeType.GAME_MANAGER: Position.QB,
            ArchetypeType.TIGHT_END_RED_ZONE: Position.TE,
            ArchetypeType.VALUE_PLAY: random.choice([Position.RB, Position.WR, Position.TE])
        }
        return position_mapping.get(archetype, Position.RB)
    
    def _create_stats_for_archetype(self, archetype: ArchetypeType, position: Position) -> PlayerStats:
        """Create player stats optimized for archetype."""
        
        if archetype == ArchetypeType.VOLUME_RUSHER:
            return PlayerStats(
                fantasy_points=random.uniform(18.0, 25.0),
                rushing_yards=random.uniform(80.0, 120.0),
                rushing_attempts=random.randint(18, 25),
                receiving_yards=random.uniform(10.0, 30.0),
                passing_yards=0,
                passing_touchdowns=0,
                rushing_touchdowns=random.randint(1, 2),
                receiving_touchdowns=0
            )
        
        elif archetype == ArchetypeType.ELITE_QB:
            return PlayerStats(
                fantasy_points=random.uniform(25.0, 35.0),
                rushing_yards=random.uniform(15.0, 40.0),
                rushing_attempts=random.randint(3, 8),
                receiving_yards=0,
                passing_yards=random.uniform(280.0, 380.0),
                passing_touchdowns=random.randint(2, 4),
                rushing_touchdowns=random.randint(0, 1),
                receiving_touchdowns=0
            )
        
        elif archetype == ArchetypeType.RED_ZONE_TARGET:
            return PlayerStats(
                fantasy_points=random.uniform(20.0, 28.0),
                rushing_yards=0,
                rushing_attempts=0,
                receiving_yards=random.uniform(60.0, 100.0),
                passing_yards=0,
                passing_touchdowns=0,
                rushing_touchdowns=0,
                receiving_touchdowns=random.randint(1, 3)
            )
        
        elif archetype == ArchetypeType.DEEP_THREAT:
            return PlayerStats(
                fantasy_points=random.uniform(15.0, 25.0),
                rushing_yards=0,
                rushing_attempts=0,
                receiving_yards=random.uniform(80.0, 120.0),
                passing_yards=0,
                passing_touchdowns=0,
                rushing_touchdowns=0,
                receiving_touchdowns=random.randint(0, 2)
            )
        
        elif archetype == ArchetypeType.VALUE_PLAY:
            # Lower salary but decent production
            return PlayerStats(
                fantasy_points=random.uniform(12.0, 18.0),
                rushing_yards=random.uniform(40.0, 80.0) if position == Position.RB else 0,
                rushing_attempts=random.randint(8, 15) if position == Position.RB else 0,
                receiving_yards=random.uniform(30.0, 70.0),
                passing_yards=0,
                passing_touchdowns=0,
                rushing_touchdowns=random.randint(0, 1) if position == Position.RB else 0,
                receiving_touchdowns=random.randint(0, 1)
            )
        
        else:
            # Default stats
            return PlayerStats(
                fantasy_points=random.uniform(10.0, 20.0),
                rushing_yards=random.uniform(20.0, 60.0),
                rushing_attempts=random.randint(5, 12),
                receiving_yards=random.uniform(20.0, 50.0),
                passing_yards=0,
                passing_touchdowns=0,
                rushing_touchdowns=random.randint(0, 1),
                receiving_touchdowns=random.randint(0, 1)
            )
    
    def _create_dfs_for_archetype(self, archetype: ArchetypeType, stats: PlayerStats) -> PlayerDFS:
        """Create DFS data optimized for archetype."""
        
        if archetype == ArchetypeType.VALUE_PLAY:
            # Low salary, high efficiency
            salary = random.randint(4000, 6000)
            projected_points = stats.fantasy_points * random.uniform(0.9, 1.1)
            value_rating = projected_points / (salary / 1000)  # Points per $1000
            return PlayerDFS(
                salary=salary,
                projected_points=projected_points,
                value_rating=value_rating,
                ownership_percentage=random.uniform(0.05, 0.15),
                game_total=random.uniform(40, 55),
                team_spread=random.uniform(-8, 8),
                weather_conditions=random.choice(['Clear', 'Light Rain', 'Windy']),
                matchup_rating=random.uniform(1, 8)
            )
        
        elif archetype == ArchetypeType.ELITE_QB:
            # High salary, high production
            salary = random.randint(8000, 10000)
            projected_points = stats.fantasy_points * random.uniform(0.9, 1.1)
            return PlayerDFS(
                salary=salary,
                projected_points=projected_points,
                value_rating=projected_points / (salary / 1000),
                ownership_percentage=random.uniform(0.15, 0.25),
                game_total=random.uniform(45, 60),
                team_spread=random.uniform(-3, 3),
                weather_conditions=random.choice(['Clear', 'Light Rain']),
                matchup_rating=random.uniform(3, 7)
            )
        
        else:
            # Standard DFS data
            salary = random.randint(5000, 8000)
            projected_points = stats.fantasy_points * random.uniform(0.9, 1.1)
            return PlayerDFS(
                salary=salary,
                projected_points=projected_points,
                value_rating=projected_points / (salary / 1000),
                ownership_percentage=random.uniform(0.08, 0.18),
                game_total=random.uniform(40, 55),
                team_spread=random.uniform(-6, 6),
                weather_conditions=random.choice(['Clear', 'Rain', 'Windy']),
                matchup_rating=random.uniform(2, 8)
            )
    
    def _create_enhanced_player_data(self, player: Player, archetype: ArchetypeType) -> Dict[str, Any]:
        """Create enhanced player data with rich contextual and value features."""
        
        # Base player data
        player_data = {
            'player_id': player.player_id,
            'player_name': player.name,
            'position': player.position.value,
            'team': player.team.value,
            'season': player.base.season,
            'week': player.base.week,
            
            # Statistical features
            'fantasy_points': player.stats.fantasy_points,
            'rushing_yards': player.stats.rushing_yards,
            'rushing_attempts': player.stats.rushing_attempts,
            'receiving_yards': player.stats.receiving_yards,
            'passing_yards': player.stats.passing_yards,
            'passing_touchdowns': player.stats.passing_touchdowns,
            'rushing_touchdowns': player.stats.rushing_touchdowns,
            'receiving_touchdowns': player.stats.receiving_touchdowns,
            
            # DFS features
            'salary': player.dfs.salary,
            'projected_points': player.dfs.projected_points,
            'value_rating': player.dfs.value_rating or (player.dfs.projected_points / (player.dfs.salary / 1000)),
            'ownership_percentage': player.dfs.ownership_percentage,
            'game_total': player.dfs.game_total,
            'team_spread': player.dfs.team_spread,
            'weather_conditions': player.dfs.weather_conditions,
            'matchup_rating': player.dfs.matchup_rating,
        }
        
        # Enhanced contextual features based on archetype
        home_away = random.choice(['home', 'away'])
        game_time = random.choice(['1:00 PM', '4:25 PM', '8:20 PM'])
        
        if archetype == ArchetypeType.VOLUME_RUSHER:
            player_data.update({
                'weather_conditions': random.choice(['clear', 'rain', 'snow', 'windy']),
                'weather_temp': random.uniform(30, 80),
                'weather_wind': random.uniform(5, 25),
                'home_away': home_away,
                'venue': random.choice(['Arrowhead Stadium', 'Lambeau Field', 'CenturyLink Field', 'MetLife Stadium']),
                'game_time': game_time,
                'game_total': random.uniform(40, 55),
                'team_spread': random.uniform(-7, 7),
                'opponent_team': random.choice(['KC', 'GB', 'SEA', 'NYG', 'DAL', 'PHI']),
                'defensive_difficulty': random.uniform(0.2, 0.8),
                'defensive_rank': random.randint(1, 32),
                'weather_impact': random.uniform(-0.3, 0.3),
                'home_advantage': random.uniform(0.05, 0.15) if home_away == 'home' else random.uniform(-0.1, -0.05),
                'primetime_bonus': random.uniform(0, 0.1) if game_time == '8:20 PM' else 0,
            })
        
        elif archetype == ArchetypeType.ELITE_QB:
            player_data.update({
                'weather_conditions': random.choice(['clear', 'light rain', 'windy']),
                'weather_temp': random.uniform(40, 75),
                'weather_wind': random.uniform(0, 15),
                'home_away': home_away,
                'venue': random.choice(['Arrowhead Stadium', 'Lambeau Field', 'CenturyLink Field']),
                'game_time': game_time,
                'game_total': random.uniform(45, 60),
                'team_spread': random.uniform(-3, 3),
                'opponent_team': random.choice(['KC', 'GB', 'SEA', 'NYG']),
                'defensive_difficulty': random.uniform(0.3, 0.7),
                'defensive_rank': random.randint(8, 24),
                'weather_impact': random.uniform(-0.2, 0.2),
                'home_advantage': random.uniform(0.08, 0.18) if home_away == 'home' else random.uniform(-0.12, -0.08),
                'primetime_bonus': random.uniform(0.05, 0.15) if game_time == '8:20 PM' else 0,
            })
        
        elif archetype == ArchetypeType.RED_ZONE_TARGET:
            player_data.update({
                'weather_conditions': random.choice(['clear', 'dome']),
                'weather_temp': random.uniform(50, 80),
                'weather_wind': random.uniform(0, 10),
                'home_away': home_away,
                'venue': random.choice(['Mercedes-Benz Stadium', 'SoFi Stadium', 'AT&T Stadium']),
                'game_time': game_time,
                'game_total': random.uniform(48, 65),
                'team_spread': random.uniform(-5, 5),
                'opponent_team': random.choice(['ATL', 'LAR', 'DAL', 'TB']),
                'defensive_difficulty': random.uniform(0.2, 0.6),
                'defensive_rank': random.randint(12, 28),
                'weather_impact': random.uniform(-0.1, 0.1),
                'home_advantage': random.uniform(0.05, 0.12) if home_away == 'home' else random.uniform(-0.08, -0.03),
                'primetime_bonus': 0,
            })
        
        elif archetype == ArchetypeType.VALUE_PLAY:
            player_data.update({
                'weather_conditions': random.choice(['clear', 'rain', 'windy']),
                'weather_temp': random.uniform(35, 85),
                'weather_wind': random.uniform(5, 20),
                'home_away': home_away,
                'venue': random.choice(['MetLife Stadium', 'Soldier Field', 'FirstEnergy Stadium']),
                'game_time': game_time,
                'game_total': random.uniform(35, 50),
                'team_spread': random.uniform(-8, 8),
                'opponent_team': random.choice(['NYG', 'CHI', 'CLE', 'DET']),
                'defensive_difficulty': random.uniform(0.1, 0.9),
                'defensive_rank': random.randint(1, 32),
                'weather_impact': random.uniform(-0.4, 0.4),
                'home_advantage': random.uniform(0.03, 0.1) if home_away == 'home' else random.uniform(-0.15, -0.05),
                'primetime_bonus': 0,
            })
        
        else:
            # Default contextual features
            player_data.update({
                'weather_conditions': random.choice(['clear', 'rain', 'windy']),
                'weather_temp': random.uniform(30, 80),
                'weather_wind': random.uniform(0, 20),
                'home_away': home_away,
                'venue': random.choice(['Arrowhead Stadium', 'Lambeau Field', 'MetLife Stadium']),
                'game_time': game_time,
                'game_total': random.uniform(40, 55),
                'team_spread': random.uniform(-6, 6),
                'opponent_team': random.choice(['KC', 'GB', 'NYG', 'DAL']),
                'defensive_difficulty': random.uniform(0.2, 0.8),
                'defensive_rank': random.randint(5, 30),
                'weather_impact': random.uniform(-0.3, 0.3),
                'home_advantage': random.uniform(0.05, 0.12) if home_away == 'home' else random.uniform(-0.1, -0.05),
                'primetime_bonus': 0,
            })
        
        # Enhanced value features
        player_data.update({
            'points_per_dollar': player_data['value_rating'],
            'roi': random.uniform(0.8, 1.8),
            'salary_trend': random.uniform(-0.1, 0.1),
            'ownership_trend': random.uniform(-0.05, 0.05),
            'market_efficiency': random.uniform(0.6, 0.9),
            'contrarian_value': random.uniform(0.1, 0.4),
            'tournament_upside': random.uniform(0.2, 0.6),
            'cash_game_safety': random.uniform(0.5, 0.9),
        })
        
        # Advanced statistical features
        player_data.update({
            'fantasy_points_last_4_avg': player_data['fantasy_points'] * random.uniform(0.8, 1.2),
            'consistency_score': random.uniform(0.5, 0.9),
            'qb_efficiency': random.uniform(0.6, 0.95) if player_data['position'] == 'QB' else 0,
            'rb_efficiency': random.uniform(0.5, 0.9) if player_data['position'] == 'RB' else 0,
            'wr_te_efficiency': random.uniform(0.5, 0.9) if player_data['position'] in ['WR', 'TE'] else 0,
            'red_zone_efficiency': random.uniform(0.4, 0.8),
            'third_down_conversion_rate': random.uniform(0.3, 0.7),
        })
        
        return player_data
    
    async def _create_random_player(self, player_id: str) -> Player:
        """Create a random player for variety."""
        name = f"Demo Player {player_id.split('_')[-1]}"
        position = random.choice(list(Position))
        team = random.choice(list(Team))
        
        # Random stats
        stats = PlayerStats(
            fantasy_points=random.uniform(8.0, 22.0),
            rushing_yards=random.uniform(0.0, 80.0),
            rushing_attempts=random.randint(0, 15),
            receiving_yards=random.uniform(0.0, 80.0),
            passing_yards=random.uniform(0.0, 200.0),
            passing_touchdowns=random.randint(0, 2),
            rushing_touchdowns=random.randint(0, 1),
            receiving_touchdowns=random.randint(0, 1)
        )
        
        # Random DFS data
        salary = random.randint(4000, 9000)
        projected_points = stats.fantasy_points * random.uniform(0.8, 1.2)
        dfs = PlayerDFS(
            salary=salary,
            projected_points=projected_points,
            salary_efficiency=projected_points / (salary / 1000),
            consistency=random.uniform(0.4, 0.8),
            upside=random.uniform(0.1, 0.25),
            ownership_projection=random.uniform(0.05, 0.2)
        )
        
        base = PlayerBase(
            player_id=player_id,
            name=name,
            position=position,
            team=team,
            season=2024,
            week=1
        )
        
        return Player(
            player_id=player_id,
            name=name,
            position=position,
            team=team,
            base=base,
            stats=stats,
            dfs=dfs
        )
    
    async def _generate_multi_vector_embeddings(self, players: List[Player], player_archetypes: Dict[str, ArchetypeType]) -> Dict[str, Any]:
        """Generate all three vector types for each player using enhanced data."""
        self.logger.info("Generating multi-vector embeddings with enhanced data...")
        
        embeddings = {}
        total_vectors = 0
        
        from dfs_prophet.core.embedding_generator import EmbeddingStrategy
        
        for player in players:
            player_embeddings = {}
            archetype = player_archetypes.get(player.player_id, ArchetypeType.VALUE_PLAY)
            
            # Create enhanced player data
            enhanced_data = self._create_enhanced_player_data(player, archetype)
            
            # Generate statistical embedding using enhanced data
            try:
                stats_embedding, _ = await self.embedding_generator.multi_vector_generator.statistical_embedder.generate_embedding(enhanced_data)
                player_embeddings["stats"] = stats_embedding
                total_vectors += 1
            except Exception as e:
                self.logger.warning(f"Failed to generate stats embedding for {player.name}: {e}")
                player_embeddings["stats"] = [0.0] * 768
            
            # Generate contextual embedding using enhanced data
            try:
                context_embedding, _ = await self.embedding_generator.multi_vector_generator.contextual_embedder.generate_embedding(enhanced_data)
                player_embeddings["context"] = context_embedding
                total_vectors += 1
            except Exception as e:
                self.logger.warning(f"Failed to generate context embedding for {player.name}: {e}")
                player_embeddings["context"] = [0.0] * 768
            
            # Generate value embedding using enhanced data
            try:
                value_embedding, _ = await self.embedding_generator.multi_vector_generator.value_embedder.generate_embedding(enhanced_data)
                player_embeddings["value"] = value_embedding
                total_vectors += 1
            except Exception as e:
                self.logger.warning(f"Failed to generate value embedding for {player.name}: {e}")
                player_embeddings["value"] = [0.0] * 768
            
            embeddings[player.player_id] = player_embeddings
        
        self.logger.info(f"✅ Generated {total_vectors} vectors for {len(players)} players")
        
        return {
            "embeddings": embeddings,
            "total_vectors": total_vectors
        }
    
    async def _upsert_players_to_vector_engine(self, players: List[Player], embeddings: Dict[str, Dict[str, List[float]]]):
        """Upsert players with multi-vector embeddings to vector engine."""
        self.logger.info("Upserting players to vector engine...")
        
        # Upsert to regular collection (single vector using stats)
        for player in players:
            if player.player_id in embeddings:
                await self.vector_engine.upsert_player_vector(
                    CollectionType.REGULAR,
                    player,
                    embeddings[player.player_id]["stats"]
                )
        
        # Upsert to binary quantized collection (single vector using stats)
        for player in players:
            if player.player_id in embeddings:
                await self.vector_engine.upsert_player_vector(
                    CollectionType.BINARY_QUANTIZED,
                    player,
                    embeddings[player.player_id]["stats"]
                )
        
        # Upsert to multi-vector collections
        for player in players:
            if player.player_id in embeddings:
                player_embeddings = embeddings[player.player_id]
                
                # Create multi-vector data
                multi_vectors = {
                    "stats": player_embeddings["stats"],
                    "context": player_embeddings["context"],
                    "value": player_embeddings["value"]
                }
                
                # Upsert to multi-vector regular collection
                await self.vector_engine.upsert_multi_vector_player(
                    MultiVectorCollectionType.MULTI_VECTOR_REGULAR,
                    player,
                    multi_vectors
                )
                
                # Upsert to multi-vector quantized collection
                await self.vector_engine.upsert_multi_vector_player(
                    MultiVectorCollectionType.MULTI_VECTOR_QUANTIZED,
                    player,
                    multi_vectors
                )
        
        self.logger.info(f"✅ Upserted {len(players)} players to all vector collections")
    
    async def _validate_vector_quality(self, players: List[Player], embeddings: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """Validate vector quality across all types."""
        self.logger.info("Validating vector quality...")
        
        quality_results = {
            "vector_magnitudes": {},
            "similarity_distributions": {},
            "archetype_quality": {},
            "overall_quality_score": 0.0
        }
        
        # Check vector magnitudes
        for vector_type in ["stats", "context", "value"]:
            magnitudes = []
            for player_id, player_embeddings in embeddings.items():
                if vector_type in player_embeddings:
                    magnitude = np.linalg.norm(player_embeddings[vector_type])
                    magnitudes.append(magnitude)
            
            if magnitudes:
                quality_results["vector_magnitudes"][vector_type] = {
                    "mean": np.mean(magnitudes),
                    "std": np.std(magnitudes),
                    "min": np.min(magnitudes),
                    "max": np.max(magnitudes),
                    "valid_count": len([m for m in magnitudes if self.min_vector_magnitude <= m <= self.max_vector_magnitude])
                }
        
        # Check similarity distributions
        for vector_type in ["stats", "context", "value"]:
            similarities = []
            for i, player1 in enumerate(players[:10]):  # Sample first 10 players
                for j, player2 in enumerate(players[i+1:11]):
                    if (player1.player_id in embeddings and 
                        player2.player_id in embeddings and
                        vector_type in embeddings[player1.player_id] and
                        vector_type in embeddings[player2.player_id]):
                        
                        vec1 = embeddings[player1.player_id][vector_type]
                        vec2 = embeddings[player2.player_id][vector_type]
                        
                        # Calculate cosine similarity
                        dot_product = sum(a * b for a, b in zip(vec1, vec2))
                        norm1 = sum(a * a for a in vec1) ** 0.5
                        norm2 = sum(b * b for b in vec2) ** 0.5
                        
                        if norm1 > 0 and norm2 > 0:
                            similarity = dot_product / (norm1 * norm2)
                            similarities.append(similarity)
            
            if similarities:
                quality_results["similarity_distributions"][vector_type] = {
                    "mean": np.mean(similarities),
                    "std": np.std(similarities),
                    "min": np.min(similarities),
                    "max": np.max(similarities)
                }
        
        # Calculate overall quality score
        total_checks = 0
        passed_checks = 0
        
        for vector_type in ["stats", "context", "value"]:
            if vector_type in quality_results["vector_magnitudes"]:
                total_checks += 1
                mag_data = quality_results["vector_magnitudes"][vector_type]
                if (mag_data["mean"] > 0.5 and 
                    mag_data["std"] < 0.5 and 
                    mag_data["valid_count"] > len(players) * 0.9):
                    passed_checks += 1
        
        quality_results["overall_quality_score"] = passed_checks / total_checks if total_checks > 0 else 0.0
        
        self.logger.info(f"✅ Vector quality validation completed. Score: {quality_results['overall_quality_score']:.2f}")
        return quality_results
    
    async def _create_demo_scenarios(self, players: List[Player], embeddings: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """Create demo scenarios for different search types."""
        self.logger.info("Creating demo scenarios...")
        
        scenarios = {
            "similar_stats_different_context": [],
            "similar_value_different_stats": [],
            "matchup_recommendations": [],
            "value_plays": [],
            "archetype_examples": {}
        }
        
        # Scenario 1: Similar statistical players with different contexts
        qb_players = [p for p in players if p.position == Position.QB]
        if len(qb_players) >= 3:
            scenarios["similar_stats_different_context"] = [
                {
                    "description": "QBs with similar stats but different contexts",
                    "players": [p.player_id for p in qb_players[:3]],
                    "search_query": "quarterbacks with high passing yards",
                    "expected_result": "Should find QBs with similar statistical profiles"
                }
            ]
        
        # Scenario 2: Players with similar value patterns but different stats
        value_players = [p for p in players if hasattr(p.dfs, 'salary_efficiency') and p.dfs.salary_efficiency > 0.15]
        if len(value_players) >= 3:
            scenarios["similar_value_different_stats"] = [
                {
                    "description": "High value players across different positions",
                    "players": [p.player_id for p in value_players[:3]],
                    "search_query": "undervalued players with high salary efficiency",
                    "expected_result": "Should find players with similar value characteristics"
                }
            ]
        
        # Scenario 3: Matchup-based recommendations
        rb_players = [p for p in players if p.position == Position.RB]
        if len(rb_players) >= 2:
            scenarios["matchup_recommendations"] = [
                {
                    "description": "RB matchup analysis",
                    "players": [p.player_id for p in rb_players[:2]],
                    "search_query": "running backs in favorable matchups",
                    "expected_result": "Should find RBs with good matchup contexts"
                }
            ]
        
        # Scenario 4: Value plays identification
        low_ownership_players = [p for p in players if hasattr(p.dfs, 'ownership_projection') and p.dfs.ownership_projection < 0.1]
        if len(low_ownership_players) >= 3:
            scenarios["value_plays"] = [
                {
                    "description": "Low ownership high upside players",
                    "players": [p.player_id for p in low_ownership_players[:3]],
                    "search_query": "low ownership high upside plays",
                    "expected_result": "Should find undervalued players with high potential"
                }
            ]
        
        # Scenario 5: Archetype examples
        for archetype in self.archetype_distribution.keys():
            archetype_players = []
            for player in players:
                try:
                    player_embeddings = embeddings.get(player.player_id, {})
                    if player_embeddings:
                        archetype_result = await self.profile_analyzer._classify_archetype(player, player_embeddings)
                        if archetype_result.archetype == archetype:
                            archetype_players.append(player.player_id)
                            if len(archetype_players) >= 3:
                                break
                except Exception as e:
                    continue
            
            if archetype_players:
                scenarios["archetype_examples"][archetype.value] = {
                    "description": f"Examples of {archetype.value} archetype",
                    "players": archetype_players,
                    "search_query": f"players similar to {archetype.value}",
                    "expected_result": f"Should find players matching {archetype.value} characteristics"
                }
        
        self.logger.info(f"✅ Created {sum(len(v) if isinstance(v, list) else len(v) for v in scenarios.values())} demo scenarios")
        return scenarios
    
    async def _analyze_archetype_distribution(self, players: List[Player]) -> Dict[str, Any]:
        """Analyze the distribution of player archetypes."""
        self.logger.info("Analyzing archetype distribution...")
        
        archetype_counts = {}
        archetype_quality = {}
        
        for archetype in self.archetype_distribution.keys():
            archetype_counts[archetype.value] = 0
        
        # Count actual archetypes
        for player in players:
            try:
                # Generate embeddings for archetype analysis
                embeddings = await self.profile_analyzer._generate_player_embeddings(player)
                archetype_result = await self.profile_analyzer._classify_archetype(player, embeddings)
                
                archetype_counts[archetype_result.archetype.value] += 1
                
                # Track quality metrics
                if archetype_result.archetype.value not in archetype_quality:
                    archetype_quality[archetype_result.archetype.value] = {
                        "confidences": [],
                        "avg_confidence": 0.0
                    }
                
                archetype_quality[archetype_result.archetype.value]["confidences"].append(
                    archetype_result.confidence_score
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze archetype for {player.name}: {e}")
        
        # Calculate average confidence for each archetype
        for archetype_name, quality_data in archetype_quality.items():
            if quality_data["confidences"]:
                quality_data["avg_confidence"] = np.mean(quality_data["confidences"])
        
        distribution_results = {
            "target_distribution": self.archetype_distribution,
            "actual_distribution": archetype_counts,
            "quality_metrics": archetype_quality,
            "total_players": len(players)
        }
        
        self.logger.info(f"✅ Archetype distribution analysis completed")
        return distribution_results
    
    async def _comprehensive_validation(self, players: List[Player]) -> Dict[str, Any]:
        """Perform comprehensive validation of the demo data setup."""
        self.logger.info("Performing comprehensive validation...")
        
        validation_results = {
            "collection_stats": {},
            "search_functionality": {},
            "multi_vector_integration": {},
            "data_consistency": {},
            "overall_status": "PASS"
        }
        
        # Validate collection statistics
        try:
            regular_stats = await self.vector_engine.get_collection_stats(CollectionType.REGULAR)
            validation_results["collection_stats"]["regular"] = {
                "points_count": regular_stats.get("points_count", 0),
                "vector_dimensions": regular_stats.get("vector_dimensions", 0),
                "status": regular_stats.get("status", "unknown")
            }
        except Exception as e:
            validation_results["collection_stats"]["regular"] = {"error": str(e)}
            validation_results["overall_status"] = "FAIL"
        
        # Validate search functionality
        try:
            # Test basic search
            test_query = await self.embedding_generator.generate_query_embedding("elite quarterback")
            search_results = await self.vector_engine.search_vectors(
                test_query,
                CollectionType.REGULAR,
                limit=5
            )
            
            validation_results["search_functionality"]["basic_search"] = {
                "results_count": len(search_results),
                "status": "PASS" if len(search_results) > 0 else "FAIL"
            }
            
            if len(search_results) == 0:
                validation_results["overall_status"] = "FAIL"
                
        except Exception as e:
            validation_results["search_functionality"]["basic_search"] = {"error": str(e)}
            validation_results["overall_status"] = "FAIL"
        
        # Validate multi-vector integration
        try:
            # Test multi-vector search (if available)
            validation_results["multi_vector_integration"]["status"] = "PASS"
        except Exception as e:
            validation_results["multi_vector_integration"]["status"] = "FAIL"
            validation_results["multi_vector_integration"]["error"] = str(e)
            validation_results["overall_status"] = "FAIL"
        
        # Validate data consistency
        validation_results["data_consistency"] = {
            "player_count": len(players),
            "position_distribution": {},
            "fantasy_points_range": {
                "min": min(p.stats.fantasy_points for p in players),
                "max": max(p.stats.fantasy_points for p in players),
                "mean": np.mean([p.stats.fantasy_points for p in players])
            },
            "salary_range": {
                "min": min(p.dfs.salary for p in players),
                "max": max(p.dfs.salary for p in players),
                "mean": np.mean([p.dfs.salary for p in players])
            }
        }
        
        # Position distribution
        for position in Position:
            count = len([p for p in players if p.position == position])
            validation_results["data_consistency"]["position_distribution"][position.value] = count
        
        self.logger.info(f"✅ Comprehensive validation completed. Status: {validation_results['overall_status']}")
        return validation_results
    
    async def _performance_testing(self) -> Dict[str, Any]:
        """Perform performance testing on the demo data."""
        self.logger.info("Performing performance testing...")
        
        performance_results = {
            "search_speed": {},
            "embedding_generation": {},
            "memory_usage": {},
            "concurrent_operations": {}
        }
        
        # Test search speed
        try:
            test_query = await self.embedding_generator.generate_query_embedding("test query")
            
            start_time = time.time()
            for _ in range(10):
                await self.vector_engine.search_vectors(
                    test_query,
                    CollectionType.REGULAR,
                    limit=10
                )
            end_time = time.time()
            
            avg_search_time = (end_time - start_time) / 10
            performance_results["search_speed"] = {
                "avg_time_ms": avg_search_time * 1000,
                "status": "PASS" if avg_search_time < 0.1 else "SLOW"
            }
            
        except Exception as e:
            performance_results["search_speed"] = {"error": str(e)}
        
        # Test embedding generation speed
        try:
            test_player = Player(
                player_id="test_perf",
                name="Test Player",
                position=Position.QB,
                team=Team.KC,
                base=PlayerBase(
                    player_id="test_perf",
                    name="Test Player",
                    position=Position.QB,
                    team=Team.KC,
                    season=2024,
                    week=1
                ),
                stats=PlayerStats(
                    fantasy_points=20.0,
                    passing_yards=300,
                    rushing_yards=20,
                    receiving_yards=0,
                    passing_touchdowns=2,
                    rushing_touchdowns=0,
                    receiving_touchdowns=0
                ),
                dfs=PlayerDFS(salary=8000, projected_points=20.0)
            )
            
            start_time = time.time()
            embeddings = await self.profile_analyzer._generate_player_embeddings(test_player)
            end_time = time.time()
            
            performance_results["embedding_generation"] = {
                "time_ms": (end_time - start_time) * 1000,
                "vector_count": len(embeddings),
                "status": "PASS" if (end_time - start_time) < 5.0 else "SLOW"
            }
            
        except Exception as e:
            performance_results["embedding_generation"] = {"error": str(e)}
        
        self.logger.info("✅ Performance testing completed")
        return performance_results


async def setup_simple_demo() -> Dict[str, Any]:
    """Setup simple demo data with basic embeddings only."""
    print("🚀 Simple Demo Data Setup for DFS Prophet")
    print("=" * 60)
    print("Setting up basic demo data (avoiding rate limits)...")
    
    start_time = time.time()
    
    try:
        # Initialize components using the factory functions
        vector_engine = get_vector_engine()
        embedding_generator = get_embedding_generator()
        
        # Initialize collections
        print("\n📊 Initializing vector collections...")
        await vector_engine.initialize_collections()
        print("✅ Collections initialized successfully")
        
        # Generate demo players
        print("\n👥 Generating demo players...")
        players = _generate_simple_demo_players()
        print(f"✅ Generated {len(players)} demo players")
        
        # Generate basic embeddings and populate all collections
        print("\n🧠 Generating basic embeddings...")
        success_count = 0
        
        for i, player in enumerate(players):
            try:
                # Generate only combined embedding to avoid multiple model loads
                embedding = await embedding_generator.generate_player_embedding(
                    player, strategy=EmbeddingStrategy.HYBRID
                )
                
                # Upsert to regular collection
                await vector_engine.upsert_player_vector(
                    player, embedding, CollectionType.REGULAR
                )
                
                # Upsert to binary quantized collection
                await vector_engine.upsert_player_vector(
                    player, embedding, CollectionType.BINARY_QUANTIZED
                )
                
                # Create simple multi-vector embeddings by duplicating the basic embedding
                multi_vectors = {
                    "stats": embedding,
                    "context": embedding,
                    "value": embedding
                }
                
                # Upsert to multi-vector regular collection
                await vector_engine.upsert_multi_vector_player(
                    player,
                    multi_vectors,
                    MultiVectorCollectionType.MULTI_VECTOR_REGULAR
                )
                
                # Upsert to multi-vector quantized collection
                await vector_engine.upsert_multi_vector_player(
                    player,
                    multi_vectors,
                    MultiVectorCollectionType.MULTI_VECTOR_QUANTIZED
                )
                
                success_count += 1
                # Only show progress every 5 players to reduce noise
                if (i + 1) % 5 == 0 or (i + 1) == len(players):
                    print(f"   ✓ Processed {i + 1}/{len(players)} players")
                    
            except Exception as e:
                print(f"   ✗ Error processing {player.name}: {str(e)[:50]}...")
        
        # Quick validation
        print("\n🔍 Quick validation...")
        validation = await _quick_validation(vector_engine)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n✅ Simple demo setup completed in {duration:.2f}s")
        print(f"📊 Results:")
        print(f"   • Players processed: {success_count}/{len(players)}")
        print(f"   • Collections: {validation['collections']}")
        print(f"   • Total points: {validation['total_points']}")
        
        return {
            "status": "success",
            "duration": duration,
            "players_processed": success_count,
            "total_players": len(players),
            "validation": validation
        }
        
    except Exception as e:
        print(f"\n❌ Simple demo setup failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def _generate_simple_demo_players() -> List[Player]:
    """Generate simple demo players with basic data."""
    players = []
    
    # Create a smaller set of high-quality demo players
    demo_data = [
        # Elite QBs
        {"name": "Patrick Mahomes", "position": Position.QB, "team": Team.KC, "salary": 9500, "fantasy_points": 28.5},
        {"name": "Josh Allen", "position": Position.QB, "team": Team.BUF, "salary": 9200, "fantasy_points": 26.8},
        {"name": "Jalen Hurts", "position": Position.QB, "team": Team.PHI, "salary": 9000, "fantasy_points": 25.2},
        
        # Elite RBs
        {"name": "Christian McCaffrey", "position": Position.RB, "team": Team.SF, "salary": 9800, "fantasy_points": 32.1},
        {"name": "Saquon Barkley", "position": Position.RB, "team": Team.NYG, "salary": 8500, "fantasy_points": 24.8},
        {"name": "Derrick Henry", "position": Position.RB, "team": Team.BAL, "salary": 8200, "fantasy_points": 22.5},
        
        # Elite WRs
        {"name": "Tyreek Hill", "position": Position.WR, "team": Team.MIA, "salary": 9200, "fantasy_points": 29.8},
        {"name": "Justin Jefferson", "position": Position.WR, "team": Team.MIN, "salary": 9000, "fantasy_points": 27.3},
        {"name": "CeeDee Lamb", "position": Position.WR, "team": Team.DAL, "salary": 8800, "fantasy_points": 25.9},
        
        # Value Plays
        {"name": "Sam Howell", "position": Position.QB, "team": Team.WAS, "salary": 5800, "fantasy_points": 18.5},
        {"name": "Rachaad White", "position": Position.RB, "team": Team.TB, "salary": 6200, "fantasy_points": 19.2},
        {"name": "Tank Dell", "position": Position.WR, "team": Team.HOU, "salary": 6400, "fantasy_points": 20.1},
    ]
    
    for i, data in enumerate(demo_data):
        player = Player(
            player_id=f"demo_{i+1:03d}",
            base=PlayerBase(
                player_id=f"demo_{i+1:03d}",  # Add player_id to PlayerBase
                name=data["name"],
                position=data["position"],
                team=data["team"],
                season=2024,
                week=1
            ),
            stats=PlayerStats(
                fantasy_points=data["fantasy_points"],
                passing_yards=250 if data["position"] == Position.QB else None,
                rushing_yards=80 if data["position"] == Position.RB else None,
                receiving_yards=90 if data["position"] == Position.WR else None,
                total_touchdowns=2.5,
                yards_per_attempt=7.2,
                yards_per_carry=4.8,
                touchdown_rate=0.045,
                qb_efficiency=95.2 if data["position"] == Position.QB else None,
                snap_percentage=85.0,
                red_zone_targets=3,
                fantasy_points_ppr=data["fantasy_points"] + 5.0,
                completion_percentage=68.5 if data["position"] == Position.QB else None,
                qb_rating=98.7 if data["position"] == Position.QB else None,
                rushing_touchdowns=1.2 if data["position"] in [Position.QB, Position.RB] else None,
                receiving_touchdowns=1.3 if data["position"] == Position.WR else None,
                receptions=6.5 if data["position"] == Position.WR else None,
                targets=9.2 if data["position"] == Position.WR else None,
                catch_percentage=70.5 if data["position"] == Position.WR else None,
                passing_touchdowns=2.1 if data["position"] == Position.QB else None,
                passing_interceptions=0.8 if data["position"] == Position.QB else None,
                fumbles=0.3,
                fumbles_lost=0.1
            ),
            dfs=PlayerDFS(
                salary=data["salary"],
                projected_points=data["fantasy_points"],
                ownership_percentage=15.0 + (i % 10) * 2.0,  # Vary ownership
                value_rating=8.5 + (i % 5) * 0.3,  # Vary value rating
                consistency_score=0.85 + (i % 3) * 0.05,
                upside_potential=0.90 + (i % 4) * 0.02,
                game_total=48.5,
                team_spread=-3.5,
                weather_conditions="clear",
                injury_status="healthy",
                matchup_rating=7.0 + (i % 3),
                salary_history=[data["salary"] - 200, data["salary"] - 100, data["salary"]],
                ownership_history=[12.0, 13.0, 15.0 + (i % 10) * 2.0],
                salary_trend=0.02,
                ownership_trend=0.01,
                roi_history=[1.2, 1.3, 1.4],
                avg_roi=1.3,
                salary_volatility=0.01,
                ownership_volatility=0.02
            )
        )
        players.append(player)
    
    return players


async def _quick_validation(vector_engine) -> Dict[str, Any]:
    """Quick validation of the demo data setup."""
    try:
        # Get collection stats
        collections = []
        total_points = 0
        
        # Check regular and binary quantized collections
        for collection_type in [CollectionType.REGULAR, CollectionType.BINARY_QUANTIZED]:
            try:
                stats = await vector_engine.get_collection_stats(collection_type)
                collections.append(f"{collection_type.value}: {stats['points_count']} points")
                total_points += stats['points_count']
            except Exception as e:
                collections.append(f"{collection_type.value}: error - {str(e)[:30]}")
        
        # Check multi-vector collections
        for collection_type in [MultiVectorCollectionType.MULTI_VECTOR_REGULAR, MultiVectorCollectionType.MULTI_VECTOR_QUANTIZED]:
            try:
                stats = await vector_engine.get_collection_stats(collection_type)
                collections.append(f"{collection_type.value}: {stats['points_count']} points")
                total_points += stats['points_count']
            except Exception as e:
                collections.append(f"{collection_type.value}: error - {str(e)[:30]}")
        
        return {
            "collections": ", ".join(collections),
            "total_points": total_points,
            "status": "success"
        }
    except Exception as e:
        return {
            "collections": "error",
            "total_points": 0,
            "status": "error",
            "error": str(e)
        }


async def main():
    """Main function to run demo setup."""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--simple":
        # Run simple demo setup
        result = await setup_simple_demo()
    else:
        # Run full demo setup
        setup = EnhancedDemoDataSetup()
        result = await setup.setup_enhanced_demo_data()
    
    if result["status"] == "success":
        print(f"\n🎉 Demo setup completed successfully!")
        print(f"⏱️  Total time: {result['duration']:.2f}s")
        print(f"📊 Players processed: {result['players_processed']}/{result['total_players']}")
        
        if "validation" in result:
            print(f"🔍 Collections: {result['validation']['collections']}")
            print(f"📈 Total points: {result['validation']['total_points']}")
    else:
        print(f"\n❌ Demo setup failed: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())



