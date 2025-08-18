#!/usr/bin/env python3

import asyncio
import sys
import os
import random
from typing import List, Dict, Any
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dfs_prophet.data.models import Player, PlayerBase, PlayerStats, PlayerDFS, Position, Team
from dfs_prophet.core import get_vector_engine, get_embedding_generator, CollectionType, EmbeddingStrategy

# Elite NFL Players with realistic data
ELITE_PLAYERS = [
    # Elite QBs
    {"name": "Josh Allen", "position": Position.QB, "team": Team.BUF, "fantasy_points": 28.5, "passing_yards": 320, "rushing_yards": 45, "salary": 8500, "projected": 26.2},
    {"name": "Patrick Mahomes", "position": Position.QB, "team": Team.KC, "fantasy_points": 26.8, "passing_yards": 295, "rushing_yards": 25, "salary": 8200, "projected": 25.1},
    {"name": "Jalen Hurts", "position": Position.QB, "team": Team.PHI, "fantasy_points": 25.2, "passing_yards": 280, "rushing_yards": 65, "salary": 8000, "projected": 24.8},
    {"name": "Justin Herbert", "position": Position.QB, "team": Team.LAC, "fantasy_points": 24.1, "passing_yards": 310, "rushing_yards": 15, "salary": 7800, "projected": 23.5},
    {"name": "Lamar Jackson", "position": Position.QB, "team": Team.BAL, "fantasy_points": 23.8, "passing_yards": 265, "rushing_yards": 85, "salary": 7900, "projected": 23.2},
    
    # Elite RBs
    {"name": "Christian McCaffrey", "position": Position.RB, "team": Team.SF, "fantasy_points": 32.5, "rushing_yards": 125, "receiving_yards": 45, "salary": 9500, "projected": 30.1},
    {"name": "Saquon Barkley", "position": Position.RB, "team": Team.NYG, "fantasy_points": 28.2, "rushing_yards": 110, "receiving_yards": 35, "salary": 8800, "projected": 26.8},
    {"name": "Derrick Henry", "position": Position.RB, "team": Team.TEN, "fantasy_points": 26.8, "rushing_yards": 135, "receiving_yards": 15, "salary": 8700, "projected": 25.5},
    {"name": "Austin Ekeler", "position": Position.RB, "team": Team.LAC, "fantasy_points": 25.5, "rushing_yards": 85, "receiving_yards": 55, "salary": 8500, "projected": 24.2},
    {"name": "Alvin Kamara", "position": Position.RB, "team": Team.NO, "fantasy_points": 24.8, "rushing_yards": 95, "receiving_yards": 40, "salary": 8300, "projected": 23.8},
    
    # Elite WRs
    {"name": "Tyreek Hill", "position": Position.WR, "team": Team.MIA, "fantasy_points": 31.2, "receiving_yards": 145, "salary": 9200, "projected": 29.5},
    {"name": "Stefon Diggs", "position": Position.WR, "team": Team.BUF, "fantasy_points": 28.8, "receiving_yards": 125, "salary": 8900, "projected": 27.2},
    {"name": "Davante Adams", "position": Position.WR, "team": Team.LV, "fantasy_points": 27.5, "receiving_yards": 115, "salary": 8700, "projected": 26.1},
    {"name": "Justin Jefferson", "position": Position.WR, "team": Team.MIN, "fantasy_points": 26.8, "receiving_yards": 135, "salary": 9000, "projected": 25.8},
    {"name": "A.J. Brown", "position": Position.WR, "team": Team.PHI, "fantasy_points": 25.2, "receiving_yards": 105, "salary": 8400, "projected": 24.5},
    
    # Elite TEs
    {"name": "Travis Kelce", "position": Position.TE, "team": Team.KC, "fantasy_points": 29.5, "receiving_yards": 95, "salary": 8800, "projected": 28.2},
    {"name": "Mark Andrews", "position": Position.TE, "team": Team.BAL, "fantasy_points": 22.8, "receiving_yards": 85, "salary": 7200, "projected": 21.5},
    {"name": "George Kittle", "position": Position.TE, "team": Team.SF, "fantasy_points": 21.5, "receiving_yards": 75, "salary": 6800, "projected": 20.8},
    {"name": "Darren Waller", "position": Position.TE, "team": Team.NYG, "fantasy_points": 19.8, "receiving_yards": 65, "salary": 6200, "projected": 18.5},
    {"name": "T.J. Hockenson", "position": Position.TE, "team": Team.MIN, "fantasy_points": 18.5, "receiving_yards": 55, "salary": 5800, "projected": 17.2},
]

# Solid Mid-tier Players
MID_TIER_PLAYERS = [
    # Mid-tier QBs
    {"name": "Dak Prescott", "position": Position.QB, "team": Team.DAL, "fantasy_points": 22.5, "passing_yards": 285, "rushing_yards": 20, "salary": 7500, "projected": 21.8},
    {"name": "Kirk Cousins", "position": Position.QB, "team": Team.MIN, "fantasy_points": 21.8, "passing_yards": 275, "rushing_yards": 10, "salary": 7200, "projected": 20.5},
    {"name": "Russell Wilson", "position": Position.QB, "team": Team.DEN, "fantasy_points": 20.2, "passing_yards": 260, "rushing_yards": 25, "salary": 6800, "projected": 19.8},
    {"name": "Tua Tagovailoa", "position": Position.QB, "team": Team.MIA, "fantasy_points": 19.5, "passing_yards": 250, "rushing_yards": 15, "salary": 6500, "projected": 18.2},
    {"name": "Geno Smith", "position": Position.QB, "team": Team.SEA, "fantasy_points": 18.8, "passing_yards": 240, "rushing_yards": 20, "salary": 6200, "projected": 17.5},
    
    # Mid-tier RBs
    {"name": "Joe Mixon", "position": Position.RB, "team": Team.CIN, "fantasy_points": 22.5, "rushing_yards": 95, "receiving_yards": 25, "salary": 7800, "projected": 21.2},
    {"name": "Aaron Jones", "position": Position.RB, "team": Team.GB, "fantasy_points": 21.8, "rushing_yards": 85, "receiving_yards": 30, "salary": 7500, "projected": 20.8},
    {"name": "Rhamondre Stevenson", "position": Position.RB, "team": Team.NE, "fantasy_points": 20.5, "rushing_yards": 90, "receiving_yards": 20, "salary": 7200, "projected": 19.5},
    {"name": "Dameon Pierce", "position": Position.RB, "team": Team.HOU, "fantasy_points": 19.8, "rushing_yards": 85, "receiving_yards": 15, "salary": 6800, "projected": 18.2},
    {"name": "Breece Hall", "position": Position.RB, "team": Team.NYJ, "fantasy_points": 18.5, "rushing_yards": 75, "receiving_yards": 25, "salary": 6500, "projected": 17.8},
    
    # Mid-tier WRs
    {"name": "CeeDee Lamb", "position": Position.WR, "team": Team.DAL, "fantasy_points": 23.5, "receiving_yards": 105, "salary": 8200, "projected": 22.8},
    {"name": "Jaylen Waddle", "position": Position.WR, "team": Team.MIA, "fantasy_points": 22.8, "receiving_yards": 95, "salary": 7800, "projected": 21.5},
    {"name": "Deebo Samuel", "position": Position.WR, "team": Team.SF, "fantasy_points": 21.5, "receiving_yards": 85, "salary": 7500, "projected": 20.2},
    {"name": "Terry McLaurin", "position": Position.WR, "team": Team.WAS, "fantasy_points": 20.8, "receiving_yards": 90, "salary": 7200, "projected": 19.8},
    {"name": "Brandon Aiyuk", "position": Position.WR, "team": Team.SF, "fantasy_points": 19.5, "receiving_yards": 75, "salary": 6800, "projected": 18.5},
]

# Value Players
VALUE_PLAYERS = [
    # Value QBs
    {"name": "Sam Howell", "position": Position.QB, "team": Team.WAS, "fantasy_points": 17.5, "passing_yards": 220, "rushing_yards": 25, "salary": 5800, "projected": 16.8},
    {"name": "Baker Mayfield", "position": Position.QB, "team": Team.TB, "fantasy_points": 16.8, "passing_yards": 210, "rushing_yards": 15, "salary": 5500, "projected": 15.5},
    {"name": "Gardner Minshew", "position": Position.QB, "team": Team.IND, "fantasy_points": 15.5, "passing_yards": 195, "rushing_yards": 20, "salary": 5200, "projected": 14.8},
    
    # Value RBs
    {"name": "Zach Charbonnet", "position": Position.RB, "team": Team.SEA, "fantasy_points": 16.5, "rushing_yards": 70, "receiving_yards": 20, "salary": 5800, "projected": 15.8},
    {"name": "Tyjae Spears", "position": Position.RB, "team": Team.TEN, "fantasy_points": 15.8, "rushing_yards": 65, "receiving_yards": 25, "salary": 5500, "projected": 15.2},
    {"name": "Roschon Johnson", "position": Position.RB, "team": Team.CHI, "fantasy_points": 14.5, "rushing_yards": 60, "receiving_yards": 15, "salary": 5200, "projected": 13.8},
    
    # Value WRs
    {"name": "Rashid Shaheed", "position": Position.WR, "team": Team.NO, "fantasy_points": 17.5, "receiving_yards": 85, "salary": 5800, "projected": 16.8},
    {"name": "Jahan Dotson", "position": Position.WR, "team": Team.WAS, "fantasy_points": 16.8, "receiving_yards": 75, "salary": 5500, "projected": 15.5},
    {"name": "Khalil Shakir", "position": Position.WR, "team": Team.BUF, "fantasy_points": 15.5, "receiving_yards": 65, "salary": 5200, "projected": 14.8},
    
    # Value TEs
    {"name": "Luke Musgrave", "position": Position.TE, "team": Team.GB, "fantasy_points": 14.5, "receiving_yards": 55, "salary": 4800, "projected": 13.8},
    {"name": "Sam LaPorta", "position": Position.TE, "team": Team.DET, "fantasy_points": 15.8, "receiving_yards": 65, "salary": 5200, "projected": 15.2},
    {"name": "Michael Mayer", "position": Position.TE, "team": Team.LV, "fantasy_points": 13.5, "receiving_yards": 45, "salary": 4500, "projected": 12.8},
]

def generate_synthetic_players() -> List[Player]:
    """Generate high-quality synthetic NFL player data."""
    players = []
    player_id_counter = 1
    
    # Combine all player tiers
    all_player_data = ELITE_PLAYERS + MID_TIER_PLAYERS + VALUE_PLAYERS
    
    for player_data in all_player_data:
        # Create player base
        base = PlayerBase(
            player_id=f"synthetic_{player_id_counter}",
            name=player_data["name"],
            position=player_data["position"],
            team=player_data["team"],
            season=2024,
            week=1
        )
        
        # Create player stats
        stats = PlayerStats(
            fantasy_points=player_data["fantasy_points"],
            passing_yards=player_data.get("passing_yards", 0.0),
            rushing_yards=player_data.get("rushing_yards", 0.0),
            receiving_yards=player_data.get("receiving_yards", 0.0)
        )
        
        # Create DFS data
        dfs = PlayerDFS(
            salary=player_data["salary"],
            projected_points=player_data["projected"],
            ownership_percentage=random.uniform(5.0, 25.0),
            value_rating=player_data["fantasy_points"] / (player_data["salary"] / 1000)
        )
        
        # Create player object
        player = Player(
            base=base,
            stats=stats,
            dfs=dfs
        )
        
        players.append(player)
        player_id_counter += 1
    
    return players

async def main():
    """Generate and load synthetic player data."""
    print("🎯 Generating High-Quality Synthetic NFL Player Data...")
    
    # Generate players
    players = generate_synthetic_players()
    print(f"✅ Generated {len(players)} high-quality players")
    
    # Get components
    engine = get_vector_engine()
    generator = get_embedding_generator()
    
    # Clear existing collections
    print("🗑️  Clearing existing collections...")
    await engine.clear_collection(CollectionType.REGULAR)
    await engine.clear_collection(CollectionType.BINARY_QUANTIZED)
    
    # Initialize collections
    print("🏗️  Initializing collections...")
    await engine.initialize_collections()
    
    # Generate embeddings
    print("🧠 Generating embeddings...")
    embeddings = await generator.generate_batch_embeddings(
        players,
        strategy=EmbeddingStrategy.TEXT_ONLY
    )
    print(f"✅ Generated {len(embeddings)} embeddings")
    
    # Load into collections
    print("📥 Loading embeddings into collections...")
    await engine.batch_upsert_vectors(players, embeddings, CollectionType.REGULAR)
    await engine.batch_upsert_vectors(players, embeddings, CollectionType.BINARY_QUANTIZED)
    
    # Get collection stats
    print("📊 Collection statistics:")
    regular_stats = await engine.get_collection_stats(CollectionType.REGULAR)
    binary_stats = await engine.get_collection_stats(CollectionType.BINARY_QUANTIZED)
    
    print(f"  Regular collection: {regular_stats.get('total_points', 'N/A')} vectors")
    print(f"  Binary collection: {binary_stats.get('total_points', 'N/A')} vectors")
    
    print("✅ Synthetic data generation complete!")
    print(f"📈 Now you have {len(players)} high-quality players with realistic fantasy points!")

if __name__ == "__main__":
    asyncio.run(main())
