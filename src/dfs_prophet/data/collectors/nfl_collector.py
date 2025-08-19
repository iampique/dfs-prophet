"""
NFL Data Collector for DFS Prophet using nfl-data-py.

Features:
- Fetch recent NFL player statistics (2022-2024 seasons)
- Data cleaning and normalization
- Feature engineering for embedding generation
- Export functions for different formats
- Progress tracking and error handling
- Sample dataset creation (1000+ players)

Focus on:
- Rushing, receiving, passing statistics
- Game context data
- Performance consistency metrics
- Fantasy-relevant features
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import hashlib

import pandas as pd
import numpy as np
import nfl_data_py as nfl
import httpx

from ...config import get_settings
from ...utils import get_logger, performance_timer
from ..models import Player, PlayerBase, PlayerStats, PlayerDFS, Position, Team


@dataclass
class CollectionProgress:
    """Track collection progress and statistics."""
    total_players: int = 0
    processed_players: int = 0
    successful_fetches: int = 0
    failed_fetches: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_season: Optional[int] = None
    current_week: Optional[int] = None
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_players == 0:
            return 0.0
        return (self.processed_players / self.total_players) * 100
    
    @property
    def elapsed_time(self) -> Optional[timedelta]:
        """Calculate elapsed time."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return datetime.now() - self.start_time
        return None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.successful_fetches + self.failed_fetches
        if total == 0:
            return 0.0
        return (self.successful_fetches / total) * 100


class NFLDataCollector:
    """Comprehensive NFL data collector with caching and async operations."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Data storage
        self.players_data: Dict[str, Dict] = {}
        self.progress = CollectionProgress()
        
        # Cache management
        self.cache_dir = Path("data/raw/nfl_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Team mapping
        self.team_mapping = self._create_team_mapping()
        
        # Position mapping
        self.position_mapping = {
            'QB': Position.QB,
            'RB': Position.RB,
            'WR': Position.WR,
            'TE': Position.TE,
            'K': Position.K,
            'DEF': Position.DEF,
            'DST': Position.DEF,
        }
    
    def _create_team_mapping(self) -> Dict[str, Team]:
        """Create mapping from NFL team abbreviations to Team enum."""
        return {
            'ARI': Team.ARI, 'ATL': Team.ATL, 'BAL': Team.BAL, 'BUF': Team.BUF,
            'CAR': Team.CAR, 'CHI': Team.CHI, 'CIN': Team.CIN, 'CLE': Team.CLE,
            'DAL': Team.DAL, 'DEN': Team.DEN, 'DET': Team.DET, 'GB': Team.GB,
            'HOU': Team.HOU, 'IND': Team.IND, 'JAX': Team.JAX, 'KC': Team.KC,
            'LAC': Team.LAC, 'LAR': Team.LAR, 'LV': Team.LV, 'MIA': Team.MIA,
            'MIN': Team.MIN, 'NE': Team.NE, 'NO': Team.NO, 'NYG': Team.NYG,
            'NYJ': Team.NYJ, 'PHI': Team.PHI, 'PIT': Team.PIT, 'SEA': Team.SEA,
            'SF': Team.SF, 'TB': Team.TB, 'TEN': Team.TEN, 'WAS': Team.WAS,
        }
    
    def _get_cache_key(self, data_type: str, season: int, week: Optional[int] = None) -> str:
        """Generate cache key for data."""
        key_parts = [data_type, str(season)]
        if week:
            key_parts.append(str(week))
        return "_".join(key_parts)
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.json"
    
    def _load_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Load data from cache if available and not expired."""
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None
        
        # Check if cache is older than 24 hours
        cache_age = time.time() - cache_path.stat().st_mtime
        if cache_age > 86400:  # 24 hours
            self.logger.info(f"Cache expired for {cache_key}, will refetch")
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            self.logger.info(f"Loaded {cache_key} from cache")
            return data
        except Exception as e:
            self.logger.warning(f"Failed to load cache for {cache_key}: {e}")
            return None
    
    def _save_to_cache(self, cache_key: str, data: Dict) -> None:
        """Save data to cache."""
        try:
            cache_path = self._get_cache_path(cache_key)
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.info(f"Saved {cache_key} to cache")
        except Exception as e:
            self.logger.error(f"Failed to save cache for {cache_key}: {e}")
    
    @performance_timer('fetch_game_context_data')
    async def fetch_game_context_data(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Fetch game context data including weather, venue, and time information."""
        cache_key = self._get_cache_key("game_context", season, week)
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data:
            return pd.DataFrame(cached_data)
        
        try:
            self.logger.info(f"Fetching game context data for season {season}" + (f" week {week}" if week else ""))
            
            # Fetch schedule data
            schedule_df = nfl.import_schedules([season])
            
            # Filter by week if specified
            if week:
                schedule_df = schedule_df[schedule_df['week'] == week]
            
            # Add weather and venue information (simulated for demo)
            context_data = []
            for _, game in schedule_df.iterrows():
                context_data.append({
                    'game_id': game.get('game_id', ''),
                    'season': season,
                    'week': game.get('week', 0),
                    'home_team': game.get('home_team', ''),
                    'away_team': game.get('away_team', ''),
                    'venue': game.get('venue', 'Unknown Stadium'),
                    'weather_temp': np.random.uniform(45, 85),  # Simulated temperature
                    'weather_wind': np.random.uniform(0, 25),   # Simulated wind speed
                    'weather_conditions': np.random.choice(['clear', 'cloudy', 'rain', 'snow']),
                    'game_time': game.get('game_time', ''),
                    'home_away': 'home',  # Will be set per player
                    'game_total': np.random.uniform(35, 55),    # Simulated game total
                    'team_spread': np.random.uniform(-10, 10),  # Simulated spread
                })
            
            context_df = pd.DataFrame(context_data)
            self._save_to_cache(cache_key, context_df.to_dict('records'))
            return context_df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch game context data: {e}")
            return pd.DataFrame()
    
    @performance_timer('fetch_opponent_defensive_data')
    async def fetch_opponent_defensive_data(self, season: int) -> pd.DataFrame:
        """Fetch opponent defensive rankings and matchup difficulty."""
        cache_key = self._get_cache_key("opponent_defense", season)
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data:
            return pd.DataFrame(cached_data)
        
        try:
            self.logger.info(f"Fetching opponent defensive data for season {season}")
            
            # Fetch team defensive stats
            team_stats_df = nfl.import_team_desc()
            
            # Create defensive rankings (simulated for demo)
            defensive_data = []
            for team in self.team_mapping.keys():
                defensive_data.append({
                    'team': team,
                    'season': season,
                    'defensive_rank': np.random.randint(1, 33),
                    'points_allowed_per_game': np.random.uniform(15, 35),
                    'yards_allowed_per_game': np.random.uniform(300, 450),
                    'passing_yards_allowed': np.random.uniform(200, 350),
                    'rushing_yards_allowed': np.random.uniform(80, 150),
                    'sacks_per_game': np.random.uniform(1.5, 4.0),
                    'interceptions_per_game': np.random.uniform(0.5, 2.0),
                    'fumbles_forced_per_game': np.random.uniform(0.5, 1.5),
                    'red_zone_defense_rank': np.random.randint(1, 33),
                    'third_down_defense_rank': np.random.randint(1, 33),
                    'matchup_difficulty': np.random.uniform(1, 10),  # 1=easy, 10=hard
                })
            
            defense_df = pd.DataFrame(defensive_data)
            self._save_to_cache(cache_key, defense_df.to_dict('records'))
            return defense_df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch opponent defensive data: {e}")
            return pd.DataFrame()
    
    @performance_timer('fetch_historical_dfs_data')
    async def fetch_historical_dfs_data(self, season: int, weeks: int = 5) -> pd.DataFrame:
        """Fetch historical DFS pricing and ownership percentages."""
        cache_key = self._get_cache_key("historical_dfs", season)
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data:
            return pd.DataFrame(cached_data)
        
        try:
            self.logger.info(f"Fetching historical DFS data for season {season}, last {weeks} weeks")
            
            # Generate historical DFS data (simulated for demo)
            dfs_data = []
            for week in range(max(1, season - 1), season + 1):
                for team in self.team_mapping.keys():
                    # Generate sample players for each team
                    for pos in ['QB', 'RB', 'WR', 'TE']:
                        for player_num in range(1, 4):  # 3 players per position
                            player_name = f"Player{player_num}_{pos}_{team}"
                            
                            # Historical salary and ownership trends
                            base_salary = np.random.randint(3000, 9000)
                            base_ownership = np.random.uniform(5, 25)
                            
                            for week_num in range(1, weeks + 1):
                                # Add some variation to simulate trends
                                salary_trend = 1 + np.random.uniform(-0.1, 0.1)
                                ownership_trend = 1 + np.random.uniform(-0.2, 0.2)
                                
                                dfs_data.append({
                                    'player_name': player_name,
                                    'team': team,
                                    'position': pos,
                                    'season': week,
                                    'week': week_num,
                                    'salary': int(base_salary * salary_trend),
                                    'ownership_percentage': max(0, base_ownership * ownership_trend),
                                    'projected_points': np.random.uniform(10, 30),
                                    'actual_points': np.random.uniform(5, 35),
                                    'roi': np.random.uniform(0.5, 2.0),
                                    'value_rating': np.random.uniform(1.5, 4.0),
                                    'game_total': np.random.uniform(35, 55),
                                    'team_spread': np.random.uniform(-10, 10),
                                })
            
            dfs_df = pd.DataFrame(dfs_data)
            self._save_to_cache(cache_key, dfs_df.to_dict('records'))
            return dfs_df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch historical DFS data: {e}")
            return pd.DataFrame()
    
    @performance_timer('fetch_injury_reports')
    async def fetch_injury_reports(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Fetch player injury reports and snap count trends."""
        cache_key = self._get_cache_key("injury_reports", season, week)
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data:
            return pd.DataFrame(cached_data)
        
        try:
            self.logger.info(f"Fetching injury reports for season {season}" + (f" week {week}" if week else ""))
            
            # Generate injury report data (simulated for demo)
            injury_data = []
            
            # Get player stats to use as base
            player_stats = await self.fetch_player_stats(season, week)
            
            for _, player in player_stats.iterrows():
                player_name = player.get('player_name', 'Unknown')
                team = player.get('recent_team', 'Unknown')
                position = player.get('position', 'Unknown')
                
                # Simulate injury status
                injury_status = np.random.choice(
                    ['healthy', 'questionable', 'doubtful', 'out', 'injured_reserve'],
                    p=[0.7, 0.15, 0.05, 0.05, 0.05]
                )
                
                # Simulate snap count trends
                snap_percentage = np.random.uniform(0, 100) if injury_status == 'healthy' else np.random.uniform(0, 50)
                
                injury_data.append({
                    'player_name': player_name,
                    'team': team,
                    'position': position,
                    'season': season,
                    'week': week or 1,
                    'injury_status': injury_status,
                    'injury_type': np.random.choice(['ankle', 'knee', 'hamstring', 'concussion', 'shoulder', 'none']),
                    'practice_status': np.random.choice(['full', 'limited', 'did_not_participate', 'not_listed']),
                    'snap_percentage': snap_percentage,
                    'snap_count': int(snap_percentage * 0.7),  # Approximate snap count
                    'red_zone_snaps': int(snap_percentage * 0.1),
                    'target_share': np.random.uniform(0, 25) if position in ['WR', 'TE', 'RB'] else 0,
                    'touch_share': np.random.uniform(0, 30) if position in ['RB'] else 0,
                })
            
            injury_df = pd.DataFrame(injury_data)
            self._save_to_cache(cache_key, injury_df.to_dict('records'))
            return injury_df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch injury reports: {e}")
            return pd.DataFrame()
    
    @performance_timer('fetch_team_scheme_data')
    async def fetch_team_scheme_data(self, season: int) -> pd.DataFrame:
        """Fetch team offensive scheme and usage patterns."""
        cache_key = self._get_cache_key("team_schemes", season)
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data:
            return pd.DataFrame(cached_data)
        
        try:
            self.logger.info(f"Fetching team scheme data for season {season}")
            
            # Generate team scheme data (simulated for demo)
            scheme_data = []
            
            for team in self.team_mapping.keys():
                # Simulate offensive scheme characteristics
                scheme_data.append({
                    'team': team,
                    'season': season,
                    'offensive_scheme': np.random.choice(['west_coast', 'air_raid', 'power_run', 'spread', 'pro_style']),
                    'play_calling_tendency': np.random.choice(['pass_heavy', 'balanced', 'run_heavy']),
                    'tempo': np.random.choice(['fast', 'medium', 'slow']),
                    'red_zone_efficiency': np.random.uniform(0.4, 0.8),
                    'third_down_conversion_rate': np.random.uniform(0.3, 0.6),
                    'average_plays_per_game': np.random.uniform(55, 75),
                    'time_of_possession_avg': np.random.uniform(25, 35),
                    'qb_mobility_usage': np.random.uniform(0, 0.3),
                    'screen_pass_frequency': np.random.uniform(0.05, 0.2),
                    'deep_pass_frequency': np.random.uniform(0.1, 0.3),
                    'run_pass_ratio': np.random.uniform(0.3, 0.7),
                    'personnel_groupings': {
                        '11_personnel': np.random.uniform(0.2, 0.6),
                        '12_personnel': np.random.uniform(0.1, 0.4),
                        '21_personnel': np.random.uniform(0.05, 0.2),
                        '22_personnel': np.random.uniform(0.05, 0.15),
                    },
                    'formation_tendencies': {
                        'shotgun': np.random.uniform(0.3, 0.8),
                        'under_center': np.random.uniform(0.1, 0.4),
                        'pistol': np.random.uniform(0.05, 0.2),
                    }
                })
            
            scheme_df = pd.DataFrame(scheme_data)
            self._save_to_cache(cache_key, scheme_df.to_dict('records'))
            return scheme_df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch team scheme data: {e}")
            return pd.DataFrame()
    
    @performance_timer('fetch_contract_salary_data')
    async def fetch_contract_salary_data(self, season: int) -> pd.DataFrame:
        """Fetch contract and salary cap information."""
        cache_key = self._get_cache_key("contract_salary", season)
        cached_data = self._load_from_cache(cache_key)
        
        if cached_data:
            return pd.DataFrame(cached_data)
        
        try:
            self.logger.info(f"Fetching contract and salary data for season {season}")
            
            # Generate contract data (simulated for demo)
            contract_data = []
            
            # Get player stats to use as base
            player_stats = await self.fetch_player_stats(season)
            
            for _, player in player_stats.iterrows():
                player_name = player.get('player_name', 'Unknown')
                team = player.get('recent_team', 'Unknown')
                position = player.get('position', 'Unknown')
                
                # Simulate contract information
                contract_data.append({
                    'player_name': player_name,
                    'team': team,
                    'position': position,
                    'season': season,
                    'contract_year': np.random.randint(1, 6),
                    'contract_length': np.random.randint(1, 5),
                    'base_salary': np.random.randint(500000, 20000000),
                    'cap_hit': np.random.randint(500000, 25000000),
                    'dead_cap': np.random.randint(0, 10000000),
                    'guaranteed_money': np.random.randint(0, 50000000),
                    'performance_bonuses': np.random.randint(0, 5000000),
                    'roster_bonus': np.random.randint(0, 2000000),
                    'signing_bonus': np.random.randint(0, 10000000),
                    'contract_value': np.random.randint(1000000, 50000000),
                    'avg_annual_value': np.random.randint(500000, 15000000),
                    'contract_status': np.random.choice(['active', 'expired', 'franchise_tag', 'rookie_deal']),
                    'cap_percentage': np.random.uniform(0.1, 15.0),
                })
            
            contract_df = pd.DataFrame(contract_data)
            self._save_to_cache(cache_key, contract_df.to_dict('records'))
            return contract_df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch contract salary data: {e}")
            return pd.DataFrame()
    
    @performance_timer('fetch_player_stats')
    async def fetch_player_stats(self, season: int, week: Optional[int] = None, force_refresh: bool = False) -> pd.DataFrame:
        """Fetch player statistics for a specific season and week.

        Set force_refresh=True to bypass cached data.
        """
        cache_key = self._get_cache_key("player_stats", season, week)
        cached_data: Optional[Dict] = None
        if not force_refresh:
            cached_data = self._load_from_cache(cache_key)
        
        if cached_data:
            df = pd.DataFrame(cached_data)
            # Apply feature engineering to cached data
            df = self._engineer_features(df)
            return df
        
        try:
            self.logger.info(f"Fetching player stats for season {season}" + (f" week {week}" if week else ""))
            
            # Fetch player stats using weekly data
            if week:
                stats_df = nfl.import_weekly_data([season])
                # Filter by week if needed
                stats_df = stats_df[stats_df['week'] == week]
            else:
                stats_df = nfl.import_weekly_data([season])
            
            # For now, use the same data for all stat types
            # In a real implementation, we'd filter by stat type
            rushing_df = stats_df.copy()
            receiving_df = stats_df.copy()
            
            # Merge all stats
            merged_df = self._merge_player_stats(stats_df, rushing_df, receiving_df)
            
            # Cache the result (before feature engineering)
            self._save_to_cache(cache_key, merged_df.to_dict('records'))
            
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Failed to fetch player stats for season {season}: {e}")
            raise
    
    def _merge_player_stats(self, passing_df: pd.DataFrame, rushing_df: pd.DataFrame, receiving_df: pd.DataFrame) -> pd.DataFrame:
        """Merge passing, rushing, and receiving statistics."""
        # Create a base dataframe with all players
        all_players = set()
        
        # Add players from all dataframes
        if not passing_df.empty:
            all_players.update(passing_df['player_name'].unique())
        if not rushing_df.empty:
            all_players.update(rushing_df['player_name'].unique())
        if not receiving_df.empty:
            all_players.update(receiving_df['player_name'].unique())
        
        # If no players found, return empty dataframe
        if not all_players:
            return pd.DataFrame()
        
        # Create base dataframe
        base_df = pd.DataFrame({'player_name': list(all_players)})
        
        # Merge passing stats
        if not passing_df.empty:
            passing_agg = passing_df.groupby('player_name').agg({
                'passing_yards': 'sum',
                'passing_tds': 'sum',
                'interceptions': 'sum',
                'attempts': 'sum',
                'completions': 'sum',
                'fantasy_points': 'sum',
                'fantasy_points_ppr': 'sum',
                'recent_team': 'first',
                'position': 'first',
                'season': 'first',
                'week': 'max'
            }).reset_index()
            base_df = base_df.merge(passing_agg, on='player_name', how='left')
        else:
            # Add empty columns
            base_df['passing_yards'] = 0
            base_df['passing_tds'] = 0
            base_df['interceptions'] = 0
            base_df['attempts'] = 0
            base_df['completions'] = 0
            base_df['fantasy_points'] = 0
            base_df['fantasy_points_ppr'] = 0
            base_df['recent_team'] = None
            base_df['position'] = None
            base_df['season'] = None
            base_df['week'] = None
        
        # Merge rushing stats
        if not rushing_df.empty:
            rushing_agg = rushing_df.groupby('player_name').agg({
                'rushing_yards': 'sum',
                'rushing_tds': 'sum',
                'carries': 'sum',
                'rushing_first_downs': 'sum'
            }).reset_index()
            base_df = base_df.merge(rushing_agg, on='player_name', how='left')
        else:
            base_df['rushing_yards'] = 0
            base_df['rushing_tds'] = 0
            base_df['carries'] = 0
            base_df['rushing_first_downs'] = 0
        
        # Merge receiving stats
        if not receiving_df.empty:
            receiving_agg = receiving_df.groupby('player_name').agg({
                'receiving_yards': 'sum',
                'receiving_tds': 'sum',
                'receptions': 'sum',
                'targets': 'sum',
                'receiving_first_downs': 'sum'
            }).reset_index()
            base_df = base_df.merge(receiving_agg, on='player_name', how='left')
        else:
            base_df['receiving_yards'] = 0
            base_df['receiving_tds'] = 0
            base_df['receptions'] = 0
            base_df['targets'] = 0
            base_df['receiving_first_downs'] = 0
        
        # Fill NaN values
        base_df = base_df.fillna(0)
        
        return base_df
    
    def _calculate_fantasy_points(self, row: pd.Series) -> float:
        """Calculate fantasy points based on standard scoring."""
        points = 0.0
        
        # Passing points
        points += (row['passing_yards'] / 25) * 1  # 1 point per 25 passing yards
        points += row['passing_tds'] * 4     # 4 points per passing TD
        points -= row['interceptions'] * 2  # -2 points per interception
        
        # Rushing points
        points += (row['rushing_yards'] / 10) * 1   # 1 point per 10 rushing yards
        points += row['rushing_tds'] * 6      # 6 points per rushing TD
        
        # Receiving points
        points += (row['receiving_yards'] / 10) * 1  # 1 point per 10 receiving yards
        points += row['receiving_tds'] * 6     # 6 points per receiving TD
        points += row['receptions'] * 1     # 1 point per reception (PPR)
        
        return round(points, 2)
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for embedding generation."""
        # Use existing fantasy_points if available, otherwise calculate
        if 'fantasy_points' not in df.columns:
            df['fantasy_points'] = df.apply(self._calculate_fantasy_points, axis=1)
        
        # Calculate efficiency metrics using correct column names
        df['passing_efficiency'] = np.where(
            df['attempts'] > 0,
            df['completions'] / df['attempts'],
            0
        )
        
        df['rushing_efficiency'] = np.where(
            df['carries'] > 0,
            df['rushing_yards'] / df['carries'],
            0
        )
        
        df['receiving_efficiency'] = np.where(
            df['targets'] > 0,
            df['receptions'] / df['targets'],
            0
        )
        
        # Calculate total yards
        df['total_yards'] = df['passing_yards'] + df['rushing_yards'] + df['receiving_yards']
        
        # Calculate total touchdowns
        df['total_touchdowns'] = df['passing_tds'] + df['rushing_tds'] + df['receiving_tds']
        
        # Add opponent team (simulated for demo)
        df['opponent_team'] = df['recent_team'].apply(lambda x: 'BUF' if x == 'KC' else 'KC' if x == 'BUF' else 'SF' if x == 'GB' else 'GB')
        
        # Calculate consistency metrics (placeholder for now)
        df['consistency_score'] = 0.5  # Will be calculated based on historical data
        
        # Add position-specific features
        df['is_qb'] = (df['position'] == 'QB').astype(int)
        df['is_rb'] = (df['position'] == 'RB').astype(int)
        df['is_wr'] = (df['position'] == 'WR').astype(int)
        df['is_te'] = (df['position'] == 'TE').astype(int)
        
        return df
    
    def _engineer_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features specifically for statistical vector generation."""
        # Enhanced statistical features
        df['fantasy_points_per_game'] = df['fantasy_points']
        df['yards_per_attempt'] = np.where(
            df['attempts'] > 0,
            df['passing_yards'] / df['attempts'],
            0
        )
        df['yards_per_carry'] = np.where(
            df['carries'] > 0,
            df['rushing_yards'] / df['carries'],
            0
        )
        df['yards_per_target'] = np.where(
            df['targets'] > 0,
            df['receiving_yards'] / df['targets'],
            0
        )
        df['touchdown_rate'] = np.where(
            (df['attempts'] + df['carries'] + df['targets']) > 0,
            df['total_touchdowns'] / (df['attempts'] + df['carries'] + df['targets']),
            0
        )
        
        # Position-specific efficiency metrics
        df['qb_efficiency'] = np.where(
            df['position'] == 'QB',
            (df['passing_yards'] * 0.04 + df['passing_tds'] * 4 - df['interceptions'] * 2) / df['attempts'],
            0
        )
        df['rb_efficiency'] = np.where(
            df['position'] == 'RB',
            (df['rushing_yards'] * 0.1 + df['rushing_tds'] * 6 + df['receiving_yards'] * 0.1 + df['receiving_tds'] * 6) / (df['carries'] + df['targets']),
            0
        )
        df['wr_te_efficiency'] = np.where(
            df['position'].isin(['WR', 'TE']),
            (df['receiving_yards'] * 0.1 + df['receiving_tds'] * 6) / df['targets'],
            0
        )
        
        # Red zone efficiency (simulated)
        df['red_zone_efficiency'] = np.random.uniform(0.3, 0.8)
        df['third_down_conversion_rate'] = np.random.uniform(0.2, 0.6)
        
        return df
    
    def _engineer_contextual_features(self, df: pd.DataFrame, context_df: pd.DataFrame, defense_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features specifically for contextual vector generation."""
        # Check if we have the required columns before merging
        if 'season' not in df.columns:
            self.logger.warning("Missing 'season' column in contextual dataframe. Available columns: " + str(list(df.columns)))
            return df
        
        # Merge context data
        df = df.merge(context_df, left_on=['recent_team', 'season', 'week'], 
                     right_on=['home_team', 'season', 'week'], how='left', suffixes=('', '_context'))
        
        # Merge defensive data
        df = df.merge(defense_df, left_on='opponent_team', right_on='team', how='left', suffixes=('', '_opponent'))
        
        # Weather impact features (with safe access)
        weather_conditions = df.get('weather_conditions', pd.Series(['clear'] * len(df)))
        df['weather_impact'] = np.where(
            weather_conditions.str.contains('rain|snow', na=False),
            -0.1,  # Negative impact for bad weather
            0.05   # Slight positive for good weather
        )
        
        weather_wind = df.get('weather_wind', pd.Series([0] * len(df)))
        df['wind_impact'] = np.where(
            weather_wind > 15,
            -0.15,  # High wind negative impact
            np.where(weather_wind > 10, -0.05, 0)  # Moderate wind slight impact
        )
        
        # Temperature impact
        weather_temp = df.get('weather_temp', pd.Series([70] * len(df)))
        df['temperature_impact'] = np.where(
            (weather_temp < 32) | (weather_temp > 85),
            -0.1,  # Extreme temperatures
            0.02   # Normal temperatures
        )
        
        # Venue and time features (with safe access)
        home_away = df.get('home_away', pd.Series(['home'] * len(df)))
        df['home_advantage'] = np.where(home_away == 'home', 0.05, -0.05)
        
        game_time = df.get('game_time', pd.Series([''] * len(df)))
        df['primetime_bonus'] = np.where(
            game_time.str.contains('20:00|21:00|22:00', na=False),
            0.03,  # Primetime games
            0
        )
        
        # Matchup difficulty features
        df['defensive_difficulty'] = (33 - df['defensive_rank']) / 32  # Normalize to 0-1
        df['points_allowed_factor'] = df['points_allowed_per_game'] / 30  # Normalize around league average
        df['yards_allowed_factor'] = df['yards_allowed_per_game'] / 350  # Normalize around league average
        
        # Game flow features
        df['game_total_impact'] = (df['game_total'] - 45) / 20  # Normalize around league average
        df['spread_impact'] = df['team_spread'] / 10  # Normalize spread
        
        return df
    
    def _engineer_value_features(self, df: pd.DataFrame, dfs_df: pd.DataFrame, contract_df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features specifically for value vector generation."""
        # Check if we have the required columns before merging
        if 'season' not in df.columns:
            self.logger.warning("Missing 'season' column in dataframe. Available columns: " + str(list(df.columns)))
            return df
        
        # Merge historical DFS data
        df = df.merge(dfs_df, left_on=['player_name', 'recent_team', 'position', 'season', 'week'], 
                     right_on=['player_name', 'team', 'position', 'season', 'week'], how='left', suffixes=('', '_dfs'))
        
        # Merge contract data
        df = df.merge(contract_df, left_on=['player_name', 'recent_team', 'position', 'season'], 
                     right_on=['player_name', 'team', 'position', 'season'], how='left', suffixes=('', '_contract'))
        
        # Value-based metrics
        df['points_per_dollar'] = df['fantasy_points'] / (df['salary'] / 1000)
        df['roi'] = df['actual_points'] / (df['salary'] / 1000)
        df['value_rating'] = df['points_per_dollar']
        
        # Salary trends (simulated)
        df['salary_trend'] = np.random.uniform(0.9, 1.1)
        df['salary_volatility'] = np.random.uniform(0.05, 0.25)
        
        # Ownership trends (simulated)
        df['ownership_trend'] = np.random.uniform(0.8, 1.2)
        df['ownership_volatility'] = np.random.uniform(0.1, 0.4)
        
        # Contract-based features
        df['contract_year_impact'] = np.where(
            df['contract_year'] == df['contract_length'],
            0.1,  # Contract year boost
            0
        )
        df['cap_percentage_impact'] = df['cap_percentage'] / 10  # Normalize cap percentage
        
        # Performance consistency
        df['consistency_rating'] = np.random.uniform(5, 10)  # 1-10 scale
        df['volatility_score'] = np.random.uniform(0.1, 0.5)  # Lower is more consistent
        
        # Market efficiency features
        df['ownership_vs_projection'] = df['ownership_percentage'] / (df['projected_points'] * 2)  # Normalize
        df['salary_vs_projection'] = df['salary'] / (df['projected_points'] * 300)  # Normalize
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame, historical_weeks: int = 4) -> pd.DataFrame:
        """Create temporal features based on historical performance."""
        # Sort by player, season, week
        df = df.sort_values(['player_name', 'season', 'week'])
        
        # Rolling averages for key metrics
        for metric in ['fantasy_points', 'passing_yards', 'rushing_yards', 'receiving_yards', 'total_touchdowns']:
            if metric in df.columns:
                df[f'{metric}_last_{historical_weeks}_avg'] = df.groupby('player_name')[metric].rolling(
                    window=historical_weeks, min_periods=1
                ).mean().reset_index(0, drop=True)
                
                df[f'{metric}_trend'] = df.groupby('player_name')[metric].rolling(
                    window=historical_weeks, min_periods=2
                ).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0).reset_index(0, drop=True)
        
        # Consistency metrics
        df['fantasy_points_std'] = df.groupby('player_name')['fantasy_points'].rolling(
            window=historical_weeks, min_periods=2
        ).std().reset_index(0, drop=True)
        
        df['consistency_score'] = np.where(
            df['fantasy_points_std'] > 0,
            1 / (1 + df['fantasy_points_std']),  # Higher consistency = higher score
            0.5
        )
        
        return df
    
    def _validate_data_quality(self, df: pd.DataFrame, data_type: str) -> pd.DataFrame:
        """Validate data quality for each vector type."""
        initial_count = len(df)
        
        # Remove rows with missing critical data
        if data_type == 'statistical':
            critical_cols = ['player_name', 'position', 'fantasy_points']
        elif data_type == 'contextual':
            critical_cols = ['player_name', 'weather_conditions', 'game_total']
        elif data_type == 'value':
            critical_cols = ['player_name']  # More lenient for value features
        else:
            critical_cols = ['player_name']
        
        df = df.dropna(subset=critical_cols)
        
        # Remove outliers for numerical columns (only for critical metrics)
        critical_numerical_cols = ['fantasy_points', 'passing_yards', 'rushing_yards', 'receiving_yards']
        for col in critical_numerical_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 2.0 * IQR  # More lenient outlier detection
                upper_bound = Q3 + 2.0 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Log quality metrics
        final_count = len(df)
        quality_score = final_count / initial_count if initial_count > 0 else 0
        
        self.logger.info(f"Data quality validation for {data_type}: {initial_count} -> {final_count} records (quality: {quality_score:.2%})")
        
        return df
    
    @performance_timer('collect_multi_dimensional_data')
    async def collect_multi_dimensional_data(self, season: int, week: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Collect comprehensive multi-dimensional player data for all vector types."""
        
        self.logger.info(f"Starting multi-dimensional data collection for season {season}" + (f" week {week}" if week else ""))
        
        try:
            # Collect all data types concurrently
            tasks = [
                self.fetch_player_stats(season, week),
                self.fetch_game_context_data(season, week),
                self.fetch_opponent_defensive_data(season),
                self.fetch_historical_dfs_data(season),
                self.fetch_injury_reports(season, week),
                self.fetch_team_scheme_data(season),
                self.fetch_contract_salary_data(season)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Unpack results
            player_stats_df = results[0] if not isinstance(results[0], Exception) else pd.DataFrame()
            context_df = results[1] if not isinstance(results[1], Exception) else pd.DataFrame()
            defense_df = results[2] if not isinstance(results[2], Exception) else pd.DataFrame()
            dfs_df = results[3] if not isinstance(results[3], Exception) else pd.DataFrame()
            injury_df = results[4] if not isinstance(results[4], Exception) else pd.DataFrame()
            scheme_df = results[5] if not isinstance(results[5], Exception) else pd.DataFrame()
            contract_df = results[6] if not isinstance(results[6], Exception) else pd.DataFrame()
            
            # Log collection results
            self.logger.info(f"Data collection completed:")
            self.logger.info(f"  Player stats: {len(player_stats_df)} records")
            self.logger.info(f"  Game context: {len(context_df)} records")
            self.logger.info(f"  Defensive data: {len(defense_df)} records")
            self.logger.info(f"  Historical DFS: {len(dfs_df)} records")
            self.logger.info(f"  Injury reports: {len(injury_df)} records")
            self.logger.info(f"  Team schemes: {len(scheme_df)} records")
            self.logger.info(f"  Contract data: {len(contract_df)} records")
            
            # Create vector-specific datasets
            vector_datasets = {}
            
            # 1. Statistical vector data
            if not player_stats_df.empty:
                stat_df = player_stats_df.copy()
                stat_df = self._engineer_statistical_features(stat_df)
                stat_df = self._create_temporal_features(stat_df)
                stat_df = self._validate_data_quality(stat_df, 'statistical')
                vector_datasets['statistical'] = stat_df
                self.logger.info(f"Statistical features: {len(stat_df)} records")
            
            # 2. Contextual vector data
            if not player_stats_df.empty and not context_df.empty and not defense_df.empty:
                context_df_enhanced = player_stats_df.copy()
                context_df_enhanced = self._engineer_contextual_features(context_df_enhanced, context_df, defense_df)
                context_df_enhanced = self._validate_data_quality(context_df_enhanced, 'contextual')
                vector_datasets['contextual'] = context_df_enhanced
                self.logger.info(f"Contextual features: {len(context_df_enhanced)} records")
            
            # 3. Value vector data
            if not player_stats_df.empty and not dfs_df.empty and not contract_df.empty:
                value_df = player_stats_df.copy()
                value_df = self._engineer_value_features(value_df, dfs_df, contract_df)
                value_df = self._validate_data_quality(value_df, 'value')
                vector_datasets['value'] = value_df
                self.logger.info(f"Value features: {len(value_df)} records")
            
            # 4. Combined vector data (all features)
            if vector_datasets:
                combined_df = player_stats_df.copy()
                
                # Merge all additional data with proper suffixes to avoid column conflicts
                if not context_df.empty:
                    combined_df = combined_df.merge(context_df, left_on=['recent_team', 'season', 'week'], 
                                                  right_on=['home_team', 'season', 'week'], how='left', suffixes=('', '_context'))
                if not defense_df.empty:
                    combined_df = combined_df.merge(defense_df, left_on='opponent_team', right_on='team', how='left', suffixes=('', '_defense'))
                if not dfs_df.empty:
                    combined_df = combined_df.merge(dfs_df, left_on=['player_name', 'recent_team', 'position', 'season', 'week'], 
                                                  right_on=['player_name', 'team', 'position', 'season', 'week'], how='left', suffixes=('', '_dfs'))
                if not contract_df.empty:
                    combined_df = combined_df.merge(contract_df, left_on=['player_name', 'recent_team', 'position', 'season'], 
                                                  right_on=['player_name', 'team', 'position', 'season'], how='left', suffixes=('', '_contract'))
                
                # Apply all feature engineering
                combined_df = self._engineer_statistical_features(combined_df)
                combined_df = self._engineer_contextual_features(combined_df, context_df, defense_df)
                combined_df = self._engineer_value_features(combined_df, dfs_df, contract_df)
                combined_df = self._create_temporal_features(combined_df)
                combined_df = self._validate_data_quality(combined_df, 'combined')
                
                vector_datasets['combined'] = combined_df
                self.logger.info(f"Combined features: {len(combined_df)} records")
            
            # Add raw datasets for reference
            vector_datasets['raw_player_stats'] = player_stats_df
            vector_datasets['raw_context'] = context_df
            vector_datasets['raw_defense'] = defense_df
            vector_datasets['raw_dfs'] = dfs_df
            vector_datasets['raw_injury'] = injury_df
            vector_datasets['raw_scheme'] = scheme_df
            vector_datasets['raw_contract'] = contract_df
            
            self.logger.info(f"Multi-dimensional data collection completed successfully")
            return vector_datasets
            
        except Exception as e:
            self.logger.error(f"Failed to collect multi-dimensional data: {e}")
            return {}
    
    def get_data_summary(self, vector_datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate summary statistics for collected data."""
        summary = {
            'collection_timestamp': datetime.now().isoformat(),
            'vector_types': {},
            'data_quality': {},
            'feature_counts': {}
        }
        
        for vector_type, df in vector_datasets.items():
            if not df.empty:
                summary['vector_types'][vector_type] = {
                    'record_count': len(df),
                    'columns': list(df.columns),
                    'missing_data': df.isnull().sum().to_dict(),
                    'data_types': df.dtypes.to_dict()
                }
                
                # Data quality metrics
                # Check for duplicates only on basic columns to avoid unhashable types
                basic_cols = [col for col in df.columns if col in ['player_name', 'position', 'team', 'season', 'week']]
                duplicates_count = 0
                if basic_cols:
                    try:
                        duplicates_count = df[basic_cols].duplicated().sum()
                    except Exception:
                        duplicates_count = 0
                
                summary['data_quality'][vector_type] = {
                    'completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                    'duplicates': duplicates_count,
                    'unique_players': df['player_name'].nunique() if 'player_name' in df.columns else 0
                }
                
                # Feature counts by type
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(include=['object']).columns
                summary['feature_counts'][vector_type] = {
                    'numerical': len(numerical_cols),
                    'categorical': len(categorical_cols),
                    'total': len(df.columns)
                }
        
        return summary
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and normalize the data."""
        # Remove rows with missing essential data
        df = df.dropna(subset=['player_name', 'position'])
        
        # Clean player names
        df['player_name'] = df['player_name'].str.strip()
        
        # Normalize team names
        df['recent_team'] = df['recent_team'].str.upper()
        
        # Remove players with no meaningful stats
        df = df[
            (df['passing_yards'] > 0) |
            (df['rushing_yards'] > 0) |
            (df['receiving_yards'] > 0) |
            (df['fantasy_points'] > 0)
        ]
        
        # Ensure numeric columns are numeric
        numeric_columns = [
            'passing_yards', 'passing_touchdowns', 'passing_interceptions',
            'rushing_yards', 'rushing_touchdowns', 'rushing_attempts',
            'receiving_yards', 'receiving_touchdowns', 'receiving_receptions',
            'fantasy_points', 'total_yards', 'total_touchdowns'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def _convert_to_player_objects(self, df: pd.DataFrame) -> List[Player]:
        """Convert DataFrame rows to Player objects."""
        players = []
        
        for _, row in df.iterrows():
            try:
                # Create player ID
                player_id = f"{row['player_name'].replace(' ', '_').lower()}_{row['season']}_{row['week']}"
                
                # Get position enum
                position = self.position_mapping.get(row['position'], Position.QB)
                
                # Get team enum
                team = self.team_mapping.get(row['recent_team'], Team.ARI)  # Default to ARI for unknown teams
                
                # Create PlayerBase
                base = PlayerBase(
                    player_id=player_id,
                    name=row['player_name'],
                    position=position,
                    team=team,
                    season=int(row['season']),
                    week=int(row['week']) if pd.notna(row['week']) else None
                )
                
                # Create PlayerStats
                stats = PlayerStats(
                    fantasy_points=float(row['fantasy_points']),
                    passing_yards=float(row['passing_yards']),
                    rushing_yards=float(row['rushing_yards']),
                    receiving_yards=float(row['receiving_yards']),
                    passing_touchdowns=int(row['passing_tds']),
                    rushing_touchdowns=int(row['rushing_tds']),
                    receiving_touchdowns=int(row['receiving_tds']),
                    passing_interceptions=int(row['interceptions']),
                    passing_attempts=int(row['attempts']),
                    passing_completions=int(row['completions']),
                    rushing_attempts=0,  # Not available in this dataset
                    receiving_receptions=int(row['receptions']),
                    receiving_targets=int(row['targets']),
                    total_yards=float(row['total_yards']),
                    total_touchdowns=int(row['total_touchdowns'])
                )
                
                # Create PlayerDFS (placeholder data)
                dfs = PlayerDFS(
                    salary=5000,  # Default salary
                    projected_points=float(row['fantasy_points']),
                    ownership_percentage=5.0  # Default ownership
                )
                
                # Create Player object
                player = Player(base=base, stats=stats, dfs=dfs)
                players.append(player)
                
            except Exception as e:
                self.logger.warning(f"Failed to convert player {row.get('player_name', 'Unknown')}: {e}")
                continue
        
        return players
    
    @performance_timer('collect_nfl_data')
    async def collect_nfl_data(
        self,
        seasons: List[int] = None,
        max_players: int = 1000,
        include_weeks: bool = True
    ) -> List[Player]:
        """Collect NFL data for specified seasons."""
        if seasons is None:
            seasons = [2022, 2023, 2024]
        
        self.progress.start_time = datetime.now()
        self.progress.total_players = max_players
        self.logger.info(f"Starting NFL data collection for seasons {seasons}")
        
        all_players = []
        
        try:
            for season in seasons:
                self.progress.current_season = season
                self.logger.info(f"Processing season {season}")
                
                if include_weeks:
                    # Collect weekly data
                    for week in range(1, 19):  # NFL regular season weeks
                        self.progress.current_week = week
                        
                        try:
                            df = await self.fetch_player_stats(season, week)
                            df = self._clean_data(df)
                            df = self._engineer_features(df)
                            
                            players = self._convert_to_player_objects(df)
                            all_players.extend(players)
                            
                            self.progress.processed_players += len(players)
                            self.progress.successful_fetches += 1
                            
                            self.logger.info(f"Season {season} Week {week}: {len(players)} players")
                            
                            # Check if we have enough players
                            if len(all_players) >= max_players:
                                break
                                
                        except Exception as e:
                            self.progress.failed_fetches += 1
                            self.logger.error(f"Failed to fetch week {week} for season {season}: {e}")
                            continue
                else:
                    # Collect season totals
                    try:
                        df = await self.fetch_player_stats(season)
                        df = self._clean_data(df)
                        df = self._engineer_features(df)
                        
                        players = self._convert_to_player_objects(df)
                        all_players.extend(players)
                        
                        self.progress.processed_players += len(players)
                        self.progress.successful_fetches += 1
                        
                        self.logger.info(f"Season {season}: {len(players)} players")
                        
                    except Exception as e:
                        self.progress.failed_fetches += 1
                        self.logger.error(f"Failed to fetch season {season}: {e}")
                        continue
                
                # Check if we have enough players
                if len(all_players) >= max_players:
                    break
        
        except Exception as e:
            self.logger.error(f"Error during data collection: {e}")
            raise
        finally:
            self.progress.end_time = datetime.now()
        
        # Limit to max_players
        all_players = all_players[:max_players]
        
        self.logger.info(f"Data collection completed: {len(all_players)} players")
        self.logger.info(f"Success rate: {self.progress.success_rate:.1f}%")
        self.logger.info(f"Elapsed time: {self.progress.elapsed_time}")
        
        return all_players
    
    def export_to_json(self, players: List[Player], filepath: str) -> None:
        """Export players to JSON format."""
        try:
            data = []
            for player in players:
                player_dict = {
                    'player_id': player.player_id,
                    'name': player.name,
                    'position': player.position.value,
                    'team': player.team.value,
                    'season': player.base.season,
                    'week': player.base.week,
                    'fantasy_points': player.stats.fantasy_points,
                    'passing_yards': player.stats.passing_yards,
                    'rushing_yards': player.stats.rushing_yards,
                    'receiving_yards': player.stats.receiving_yards,
                    'total_yards': getattr(player.stats, 'total_yards', 0),
                    'total_touchdowns': getattr(player.stats, 'total_touchdowns', 0),
                    'salary': player.dfs.salary,
                    'projected_points': player.dfs.projected_points,
                    'ownership_percentage': player.dfs.ownership_percentage
                }
                data.append(player_dict)
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.logger.info(f"Exported {len(players)} players to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export to JSON: {e}")
            raise
    
    def export_to_csv(self, players: List[Player], filepath: str) -> None:
        """Export players to CSV format."""
        try:
            # Convert to DataFrame
            data = []
            for player in players:
                row = {
                    'player_id': player.player_id,
                    'name': player.name,
                    'position': player.position.value,
                    'team': player.team.value,
                    'season': player.base.season,
                    'week': player.base.week,
                    'fantasy_points': player.stats.fantasy_points,
                    'passing_yards': player.stats.passing_yards,
                    'rushing_yards': player.stats.rushing_yards,
                    'receiving_yards': player.stats.receiving_yards,
                    'total_yards': getattr(player.stats, 'total_yards', 0),
                    'total_touchdowns': getattr(player.stats, 'total_touchdowns', 0),
                    'salary': player.dfs.salary,
                    'projected_points': player.dfs.projected_points,
                    'ownership_percentage': player.dfs.ownership_percentage
                }
                data.append(row)
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
            
            self.logger.info(f"Exported {len(players)} players to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to export to CSV: {e}")
            raise
    
    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of the data collection."""
        return {
            'total_players': self.progress.total_players,
            'processed_players': self.progress.processed_players,
            'successful_fetches': self.progress.successful_fetches,
            'failed_fetches': self.progress.failed_fetches,
            'success_rate': self.progress.success_rate,
            'progress_percentage': self.progress.progress_percentage,
            'elapsed_time': str(self.progress.elapsed_time) if self.progress.elapsed_time else None,
            'current_season': self.progress.current_season,
            'current_week': self.progress.current_week
        }

    def clear_cache(self) -> int:
        """Clear all cached files.

        Returns the number of files removed.
        """
        removed = 0
        try:
            for path in self.cache_dir.glob("*.json"):
                try:
                    path.unlink()
                    removed += 1
                except Exception as e:
                    self.logger.warning(f"Failed to delete cache file {path}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to iterate cache directory {self.cache_dir}: {e}")
        return removed


# Global collector instance
_nfl_collector: Optional[NFLDataCollector] = None


def get_nfl_collector() -> NFLDataCollector:
    """Get the global NFL data collector instance."""
    global _nfl_collector
    if _nfl_collector is None:
        _nfl_collector = NFLDataCollector()
    return _nfl_collector
