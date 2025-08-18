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
        
        # Calculate consistency metrics (placeholder for now)
        df['consistency_score'] = 0.5  # Will be calculated based on historical data
        
        # Add position-specific features
        df['is_qb'] = (df['position'] == 'QB').astype(int)
        df['is_rb'] = (df['position'] == 'RB').astype(int)
        df['is_wr'] = (df['position'] == 'WR').astype(int)
        df['is_te'] = (df['position'] == 'TE').astype(int)
        
        return df
    
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
