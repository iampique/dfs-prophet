"""
Vector Performance Monitoring for DFS Prophet

Comprehensive performance tracking and analytics for multi-vector operations:
- Search latency per vector type
- Memory usage breakdown by vector type
- Search accuracy metrics for fusion vs individual vectors
- Vector quality scores and degradation detection
- User interaction patterns with different vector types
"""

import asyncio
import time
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import psutil
import numpy as np
from collections import defaultdict, deque

from ..config import get_settings
from ..utils import get_logger, performance_timer
from ..core import CollectionType, MultiVectorCollectionType, VectorType


class MetricType(str, Enum):
    """Types of performance metrics."""
    SEARCH_LATENCY = "search_latency"
    MEMORY_USAGE = "memory_usage"
    SEARCH_ACCURACY = "search_accuracy"
    VECTOR_QUALITY = "vector_quality"
    USER_INTERACTION = "user_interaction"
    RESOURCE_USAGE = "resource_usage"


class AlertLevel(str, Enum):
    """Alert levels for performance issues."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class VectorSearchMetrics:
    """Metrics for vector search operations."""
    vector_type: str
    collection_type: str
    latency_ms: float
    result_count: int
    accuracy_score: float
    memory_usage_mb: float
    timestamp: datetime
    query_type: str
    user_id: Optional[str] = None


@dataclass
class VectorQualityMetrics:
    """Metrics for vector quality assessment."""
    vector_type: str
    quality_score: float
    degradation_rate: float
    consistency_score: float
    outlier_count: int
    timestamp: datetime
    sample_size: int


@dataclass
class MultiVectorComparisonMetrics:
    """Metrics comparing multi-vector vs single-vector performance."""
    fusion_accuracy: float
    individual_accuracies: Dict[str, float]
    fusion_latency: float
    individual_latencies: Dict[str, float]
    fusion_memory: float
    individual_memory: Dict[str, float]
    timestamp: datetime
    query_count: int


@dataclass
class UserInteractionMetrics:
    """Metrics for user interaction patterns."""
    user_id: str
    vector_type_preferences: Dict[str, int]
    search_patterns: Dict[str, int]
    session_duration: float
    success_rate: float
    timestamp: datetime


@dataclass
class ResourceUsageMetrics:
    """Metrics for system resource usage."""
    cpu_usage_percent: float
    memory_usage_mb: float
    disk_usage_percent: float
    network_io_mb: float
    vector_cache_hit_rate: float
    timestamp: datetime


@dataclass
class PerformanceAlert:
    """Performance alert for monitoring."""
    alert_id: str
    alert_level: AlertLevel
    metric_type: MetricType
    message: str
    threshold: float
    current_value: float
    timestamp: datetime
    vector_type: Optional[str] = None
    recommendations: List[str] = None


class VectorPerformanceMonitor:
    """Comprehensive vector performance monitoring system."""
    
    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(__name__)
        
        # Metrics storage
        self.search_metrics: deque = deque(maxlen=10000)
        self.quality_metrics: deque = deque(maxlen=1000)
        self.comparison_metrics: deque = deque(maxlen=1000)
        self.interaction_metrics: deque = deque(maxlen=5000)
        self.resource_metrics: deque = deque(maxlen=1000)
        self.alerts: deque = deque(maxlen=1000)
        
        # Performance thresholds
        self.thresholds = {
            "search_latency_ms": 100.0,
            "memory_usage_mb": 1000.0,
            "quality_score": 0.7,
            "accuracy_threshold": 0.8,
            "cpu_usage_percent": 80.0,
            "cache_hit_rate": 0.9
        }
        
        # Statistics tracking
        self.vector_type_stats = defaultdict(lambda: {
            "total_searches": 0,
            "avg_latency": 0.0,
            "avg_accuracy": 0.0,
            "total_memory": 0.0,
            "last_updated": datetime.now()
        })
        
        # Alert tracking
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        
        # Performance optimization recommendations
        self.recommendations: List[str] = []
    
    @performance_timer('track_search_metrics')
    async def track_search_metrics(
        self,
        vector_type: str,
        collection_type: str,
        latency_ms: float,
        result_count: int,
        accuracy_score: float,
        memory_usage_mb: float,
        query_type: str = "general",
        user_id: Optional[str] = None
    ) -> VectorSearchMetrics:
        """Track search performance metrics."""
        metrics = VectorSearchMetrics(
            vector_type=vector_type,
            collection_type=collection_type,
            latency_ms=latency_ms,
            result_count=result_count,
            accuracy_score=accuracy_score,
            memory_usage_mb=memory_usage_mb,
            timestamp=datetime.now(),
            query_type=query_type,
            user_id=user_id
        )
        
        self.search_metrics.append(metrics)
        await self._update_vector_stats(metrics)
        await self._check_performance_alerts(metrics)
        
        return metrics
    
    @performance_timer('track_quality_metrics')
    async def track_quality_metrics(
        self,
        vector_type: str,
        quality_score: float,
        degradation_rate: float = 0.0,
        consistency_score: float = 1.0,
        outlier_count: int = 0,
        sample_size: int = 1
    ) -> VectorQualityMetrics:
        """Track vector quality metrics."""
        metrics = VectorQualityMetrics(
            vector_type=vector_type,
            quality_score=quality_score,
            degradation_rate=degradation_rate,
            consistency_score=consistency_score,
            outlier_count=outlier_count,
            timestamp=datetime.now(),
            sample_size=sample_size
        )
        
        self.quality_metrics.append(metrics)
        await self._check_quality_alerts(metrics)
        
        return metrics
    
    @performance_timer('track_multi_vector_comparison')
    async def track_multi_vector_comparison(
        self,
        fusion_accuracy: float,
        individual_accuracies: Dict[str, float],
        fusion_latency: float,
        individual_latencies: Dict[str, float],
        fusion_memory: float,
        individual_memory: Dict[str, float],
        query_count: int = 1
    ) -> MultiVectorComparisonMetrics:
        """Track multi-vector vs single-vector comparison metrics."""
        metrics = MultiVectorComparisonMetrics(
            fusion_accuracy=fusion_accuracy,
            individual_accuracies=individual_accuracies,
            fusion_latency=fusion_latency,
            individual_latencies=individual_latencies,
            fusion_memory=fusion_memory,
            individual_memory=individual_memory,
            timestamp=datetime.now(),
            query_count=query_count
        )
        
        self.comparison_metrics.append(metrics)
        await self._analyze_fusion_performance(metrics)
        
        return metrics
    
    @performance_timer('track_user_interaction')
    async def track_user_interaction(
        self,
        user_id: str,
        vector_type: str,
        search_pattern: str,
        session_duration: float,
        success: bool
    ) -> UserInteractionMetrics:
        """Track user interaction patterns."""
        # Get or create user metrics
        user_metrics = await self._get_user_metrics(user_id)
        
        # Update preferences
        user_metrics.vector_type_preferences[vector_type] = (
            user_metrics.vector_type_preferences.get(vector_type, 0) + 1
        )
        
        # Update search patterns
        user_metrics.search_patterns[search_pattern] = (
            user_metrics.search_patterns.get(search_pattern, 0) + 1
        )
        
        # Update success rate
        total_interactions = sum(user_metrics.search_patterns.values())
        successful_interactions = sum(
            1 for pattern, count in user_metrics.search_patterns.items()
            if "success" in pattern.lower()
        )
        user_metrics.success_rate = successful_interactions / total_interactions if total_interactions > 0 else 0.0
        
        user_metrics.timestamp = datetime.now()
        
        self.interaction_metrics.append(user_metrics)
        await self._analyze_user_patterns(user_metrics)
        
        return user_metrics
    
    @performance_timer('track_resource_usage')
    async def track_resource_usage(self) -> ResourceUsageMetrics:
        """Track system resource usage."""
        # Get system metrics
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        network = psutil.net_io_counters()
        
        # Calculate cache hit rate (mock for now)
        cache_hit_rate = await self._calculate_cache_hit_rate()
        
        metrics = ResourceUsageMetrics(
            cpu_usage_percent=cpu_usage,
            memory_usage_mb=memory.used / (1024 * 1024),
            disk_usage_percent=(disk.used / disk.total) * 100,
            network_io_mb=(network.bytes_sent + network.bytes_recv) / (1024 * 1024),
            vector_cache_hit_rate=cache_hit_rate,
            timestamp=datetime.now()
        )
        
        self.resource_metrics.append(metrics)
        await self._check_resource_alerts(metrics)
        
        return metrics
    
    async def get_performance_summary(self, time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        cutoff_time = datetime.now() - time_window
        
        # Filter metrics by time window
        recent_searches = [
            m for m in self.search_metrics 
            if m.timestamp >= cutoff_time
        ]
        recent_quality = [
            m for m in self.quality_metrics 
            if m.timestamp >= cutoff_time
        ]
        recent_resources = [
            m for m in self.resource_metrics 
            if m.timestamp >= cutoff_time
        ]
        
        summary = {
            "time_window": str(time_window),
            "total_searches": len(recent_searches),
            "vector_type_performance": {},
            "quality_metrics": {},
            "resource_usage": {},
            "alerts": [],
            "recommendations": self.recommendations[-10:]  # Last 10 recommendations
        }
        
        # Vector type performance
        for vector_type in set(m.vector_type for m in recent_searches):
            type_searches = [m for m in recent_searches if m.vector_type == vector_type]
            if type_searches:
                summary["vector_type_performance"][vector_type] = {
                    "total_searches": len(type_searches),
                    "avg_latency_ms": statistics.mean(m.latency_ms for m in type_searches),
                    "avg_accuracy": statistics.mean(m.accuracy_score for m in type_searches),
                    "avg_memory_mb": statistics.mean(m.memory_usage_mb for m in type_searches),
                    "total_memory_mb": sum(m.memory_usage_mb for m in type_searches)
                }
        
        # Quality metrics
        for vector_type in set(m.vector_type for m in recent_quality):
            type_quality = [m for m in recent_quality if m.vector_type == vector_type]
            if type_quality:
                summary["quality_metrics"][vector_type] = {
                    "avg_quality_score": statistics.mean(m.quality_score for m in type_quality),
                    "avg_degradation_rate": statistics.mean(m.degradation_rate for m in type_quality),
                    "avg_consistency": statistics.mean(m.consistency_score for m in type_quality),
                    "total_outliers": sum(m.outlier_count for m in type_quality)
                }
        
        # Resource usage
        if recent_resources:
            summary["resource_usage"] = {
                "avg_cpu_percent": statistics.mean(m.cpu_usage_percent for m in recent_resources),
                "avg_memory_mb": statistics.mean(m.memory_usage_mb for m in recent_resources),
                "avg_disk_percent": statistics.mean(m.disk_usage_percent for m in recent_resources),
                "avg_cache_hit_rate": statistics.mean(m.vector_cache_hit_rate for m in recent_resources),
                "peak_memory_mb": max(m.memory_usage_mb for m in recent_resources),
                "peak_cpu_percent": max(m.cpu_usage_percent for m in recent_resources)
            }
        
        # Active alerts
        summary["alerts"] = [
            asdict(alert) for alert in self.active_alerts.values()
            if alert.timestamp >= cutoff_time
        ]
        
        return summary
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data formatted for dashboard visualization."""
        # Get recent metrics
        recent_searches = list(self.search_metrics)[-100:]  # Last 100 searches
        recent_quality = list(self.quality_metrics)[-50:]   # Last 50 quality checks
        recent_resources = list(self.resource_metrics)[-100:]  # Last 100 resource checks
        
        dashboard_data = {
            "performance_trends": {
                "latency_trend": [
                    {"timestamp": m.timestamp.isoformat(), "latency": m.latency_ms, "vector_type": m.vector_type}
                    for m in recent_searches
                ],
                "accuracy_trend": [
                    {"timestamp": m.timestamp.isoformat(), "accuracy": m.accuracy_score, "vector_type": m.vector_type}
                    for m in recent_searches
                ],
                "memory_trend": [
                    {"timestamp": m.timestamp.isoformat(), "memory": m.memory_usage_mb, "vector_type": m.vector_type}
                    for m in recent_searches
                ]
            },
            "vector_type_comparison": {
                "latency_comparison": {},
                "accuracy_comparison": {},
                "memory_comparison": {},
                "usage_distribution": {}
            },
            "quality_metrics": {
                "quality_scores": [
                    {"timestamp": m.timestamp.isoformat(), "score": m.quality_score, "vector_type": m.vector_type}
                    for m in recent_quality
                ],
                "degradation_trends": [
                    {"timestamp": m.timestamp.isoformat(), "degradation": m.degradation_rate, "vector_type": m.vector_type}
                    for m in recent_quality
                ]
            },
            "resource_usage": {
                "cpu_trend": [
                    {"timestamp": m.timestamp.isoformat(), "cpu": m.cpu_usage_percent}
                    for m in recent_resources
                ],
                "memory_trend": [
                    {"timestamp": m.timestamp.isoformat(), "memory": m.memory_usage_mb}
                    for m in recent_resources
                ],
                "cache_hit_rate": [
                    {"timestamp": m.timestamp.isoformat(), "hit_rate": m.vector_cache_hit_rate}
                    for m in recent_resources
                ]
            },
            "alerts": [
                asdict(alert) for alert in list(self.active_alerts.values())[-10:]
            ],
            "recommendations": self.recommendations[-5:]
        }
        
        # Calculate vector type comparisons
        vector_types = set(m.vector_type for m in recent_searches)
        for vector_type in vector_types:
            type_searches = [m for m in recent_searches if m.vector_type == vector_type]
            if type_searches:
                dashboard_data["vector_type_comparison"]["latency_comparison"][vector_type] = {
                    "avg": statistics.mean(m.latency_ms for m in type_searches),
                    "min": min(m.latency_ms for m in type_searches),
                    "max": max(m.latency_ms for m in type_searches)
                }
                dashboard_data["vector_type_comparison"]["accuracy_comparison"][vector_type] = {
                    "avg": statistics.mean(m.accuracy_score for m in type_searches),
                    "min": min(m.accuracy_score for m in type_searches),
                    "max": max(m.accuracy_score for m in type_searches)
                }
                dashboard_data["vector_type_comparison"]["memory_comparison"][vector_type] = {
                    "avg": statistics.mean(m.memory_usage_mb for m in type_searches),
                    "total": sum(m.memory_usage_mb for m in type_searches)
                }
                dashboard_data["vector_type_comparison"]["usage_distribution"][vector_type] = len(type_searches)
        
        return dashboard_data
    
    async def generate_optimization_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on performance data."""
        recommendations = []
        
        # Analyze recent performance
        recent_searches = list(self.search_metrics)[-100:]
        if not recent_searches:
            return ["No performance data available for analysis"]
        
        # Check for high latency
        avg_latency = statistics.mean(m.latency_ms for m in recent_searches)
        if avg_latency > self.thresholds["search_latency_ms"]:
            recommendations.append(
                f"High average search latency ({avg_latency:.1f}ms). "
                "Consider optimizing vector indexing or reducing vector dimensions."
            )
        
        # Check for memory usage
        avg_memory = statistics.mean(m.memory_usage_mb for m in recent_searches)
        if avg_memory > self.thresholds["memory_usage_mb"]:
            recommendations.append(
                f"High memory usage ({avg_memory:.1f}MB). "
                "Consider implementing vector quantization or memory pooling."
            )
        
        # Check for accuracy issues
        avg_accuracy = statistics.mean(m.accuracy_score for m in recent_searches)
        if avg_accuracy < self.thresholds["accuracy_threshold"]:
            recommendations.append(
                f"Low search accuracy ({avg_accuracy:.2f}). "
                "Consider retraining embedding models or adjusting search parameters."
            )
        
        # Check resource usage
        recent_resources = list(self.resource_metrics)[-50:]
        if recent_resources:
            avg_cpu = statistics.mean(m.cpu_usage_percent for m in recent_resources)
            if avg_cpu > self.thresholds["cpu_usage_percent"]:
                recommendations.append(
                    f"High CPU usage ({avg_cpu:.1f}%). "
                    "Consider implementing caching or load balancing."
                )
            
            avg_cache_hit = statistics.mean(m.vector_cache_hit_rate for m in recent_resources)
            if avg_cache_hit < self.thresholds["cache_hit_rate"]:
                recommendations.append(
                    f"Low cache hit rate ({avg_cache_hit:.2f}). "
                    "Consider expanding cache size or optimizing cache strategy."
                )
        
        # Vector type specific recommendations
        vector_type_performance = defaultdict(list)
        for search in recent_searches:
            vector_type_performance[search.vector_type].append(search)
        
        for vector_type, searches in vector_type_performance.items():
            type_avg_latency = statistics.mean(m.latency_ms for m in searches)
            type_avg_accuracy = statistics.mean(m.accuracy_score for m in searches)
            
            if type_avg_latency > avg_latency * 1.5:
                recommendations.append(
                    f"High latency for {vector_type} vectors ({type_avg_latency:.1f}ms). "
                    "Consider optimizing this vector type specifically."
                )
            
            if type_avg_accuracy < avg_accuracy * 0.8:
                recommendations.append(
                    f"Low accuracy for {vector_type} vectors ({type_avg_accuracy:.2f}). "
                    "Consider retraining or adjusting parameters for this vector type."
                )
        
        # Store recommendations
        self.recommendations.extend(recommendations)
        if len(self.recommendations) > 100:
            self.recommendations = self.recommendations[-100:]
        
        return recommendations
    
    async def _update_vector_stats(self, metrics: VectorSearchMetrics):
        """Update vector type statistics."""
        stats = self.vector_type_stats[metrics.vector_type]
        stats["total_searches"] += 1
        
        # Update running averages
        n = stats["total_searches"]
        stats["avg_latency"] = (stats["avg_latency"] * (n - 1) + metrics.latency_ms) / n
        stats["avg_accuracy"] = (stats["avg_accuracy"] * (n - 1) + metrics.accuracy_score) / n
        stats["total_memory"] += metrics.memory_usage_mb
        stats["last_updated"] = datetime.now()
    
    async def _check_performance_alerts(self, metrics: VectorSearchMetrics):
        """Check for performance alerts."""
        # Check latency
        if metrics.latency_ms > self.thresholds["search_latency_ms"]:
            await self._create_alert(
                AlertLevel.WARNING,
                MetricType.SEARCH_LATENCY,
                f"High search latency for {metrics.vector_type}: {metrics.latency_ms:.1f}ms",
                self.thresholds["search_latency_ms"],
                metrics.latency_ms,
                metrics.vector_type
            )
        
        # Check memory usage
        if metrics.memory_usage_mb > self.thresholds["memory_usage_mb"]:
            await self._create_alert(
                AlertLevel.WARNING,
                MetricType.MEMORY_USAGE,
                f"High memory usage for {metrics.vector_type}: {metrics.memory_usage_mb:.1f}MB",
                self.thresholds["memory_usage_mb"],
                metrics.memory_usage_mb,
                metrics.vector_type
            )
        
        # Check accuracy
        if metrics.accuracy_score < self.thresholds["accuracy_threshold"]:
            await self._create_alert(
                AlertLevel.WARNING,
                MetricType.SEARCH_ACCURACY,
                f"Low search accuracy for {metrics.vector_type}: {metrics.accuracy_score:.2f}",
                self.thresholds["accuracy_threshold"],
                metrics.accuracy_score,
                metrics.vector_type
            )
    
    async def _check_quality_alerts(self, metrics: VectorQualityMetrics):
        """Check for quality alerts."""
        if metrics.quality_score < self.thresholds["quality_score"]:
            await self._create_alert(
                AlertLevel.CRITICAL,
                MetricType.VECTOR_QUALITY,
                f"Low vector quality for {metrics.vector_type}: {metrics.quality_score:.2f}",
                self.thresholds["quality_score"],
                metrics.quality_score,
                metrics.vector_type
            )
    
    async def _check_resource_alerts(self, metrics: ResourceUsageMetrics):
        """Check for resource alerts."""
        if metrics.cpu_usage_percent > self.thresholds["cpu_usage_percent"]:
            await self._create_alert(
                AlertLevel.WARNING,
                MetricType.RESOURCE_USAGE,
                f"High CPU usage: {metrics.cpu_usage_percent:.1f}%",
                self.thresholds["cpu_usage_percent"],
                metrics.cpu_usage_percent
            )
        
        if metrics.vector_cache_hit_rate < self.thresholds["cache_hit_rate"]:
            await self._create_alert(
                AlertLevel.INFO,
                MetricType.RESOURCE_USAGE,
                f"Low cache hit rate: {metrics.vector_cache_hit_rate:.2f}",
                self.thresholds["cache_hit_rate"],
                metrics.vector_cache_hit_rate
            )
    
    async def _create_alert(
        self,
        level: AlertLevel,
        metric_type: MetricType,
        message: str,
        threshold: float,
        current_value: float,
        vector_type: Optional[str] = None
    ):
        """Create a performance alert."""
        alert_id = f"{metric_type}_{vector_type or 'general'}_{int(time.time())}"
        
        alert = PerformanceAlert(
            alert_id=alert_id,
            alert_level=level,
            metric_type=metric_type,
            message=message,
            threshold=threshold,
            current_value=current_value,
            timestamp=datetime.now(),
            vector_type=vector_type,
            recommendations=await self._generate_alert_recommendations(metric_type, vector_type)
        )
        
        self.active_alerts[alert_id] = alert
        self.alerts.append(alert)
        
        # Log alert
        log_level = "warning" if level == AlertLevel.WARNING else "error" if level == AlertLevel.CRITICAL else "info"
        getattr(self.logger, log_level)(f"Performance Alert: {message}")
    
    async def _generate_alert_recommendations(self, metric_type: MetricType, vector_type: Optional[str] = None) -> List[str]:
        """Generate recommendations for an alert."""
        recommendations = []
        
        if metric_type == MetricType.SEARCH_LATENCY:
            recommendations.extend([
                "Consider implementing vector quantization",
                "Optimize vector indexing strategy",
                "Implement caching for frequent queries"
            ])
        elif metric_type == MetricType.MEMORY_USAGE:
            recommendations.extend([
                "Implement memory pooling",
                "Use vector compression techniques",
                "Consider distributed vector storage"
            ])
        elif metric_type == MetricType.SEARCH_ACCURACY:
            recommendations.extend([
                "Retrain embedding models with more data",
                "Adjust search parameters and thresholds",
                "Implement ensemble search strategies"
            ])
        elif metric_type == MetricType.VECTOR_QUALITY:
            recommendations.extend([
                "Monitor vector degradation over time",
                "Implement quality-based filtering",
                "Consider vector regeneration strategies"
            ])
        
        return recommendations
    
    async def _get_user_metrics(self, user_id: str) -> UserInteractionMetrics:
        """Get or create user interaction metrics."""
        # Find existing user metrics
        for metrics in reversed(self.interaction_metrics):
            if metrics.user_id == user_id:
                return metrics
        
        # Create new user metrics
        return UserInteractionMetrics(
            user_id=user_id,
            vector_type_preferences={},
            search_patterns={},
            session_duration=0.0,
            success_rate=0.0,
            timestamp=datetime.now()
        )
    
    async def _analyze_fusion_performance(self, metrics: MultiVectorComparisonMetrics):
        """Analyze multi-vector fusion performance."""
        # Check if fusion is performing better than individual vectors
        fusion_better_accuracy = metrics.fusion_accuracy > max(metrics.individual_accuracies.values())
        fusion_better_latency = metrics.fusion_latency < min(metrics.individual_latencies.values())
        
        if not fusion_better_accuracy and not fusion_better_latency:
            await self._create_alert(
                AlertLevel.WARNING,
                MetricType.SEARCH_ACCURACY,
                "Multi-vector fusion not performing optimally",
                0.8,
                metrics.fusion_accuracy
            )
    
    async def _analyze_user_patterns(self, metrics: UserInteractionMetrics):
        """Analyze user interaction patterns."""
        # Check for unusual patterns
        total_interactions = sum(metrics.vector_type_preferences.values())
        if total_interactions > 0:
            for vector_type, count in metrics.vector_type_preferences.items():
                usage_percentage = count / total_interactions
                if usage_percentage > 0.8:
                    self.logger.info(f"User {metrics.user_id} heavily prefers {vector_type} vectors ({usage_percentage:.1%})")
    
    async def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate (mock implementation)."""
        # In a real implementation, this would track actual cache hits/misses
        return 0.85  # Mock 85% hit rate

    async def get_search_analytics(self) -> Dict[str, Any]:
        """Get comprehensive search analytics for health checks."""
        try:
            # Calculate average latency across all searches
            if self.search_metrics:
                latencies = [m.latency_ms for m in self.search_metrics]
                average_latency = statistics.mean(latencies)
                total_searches = len(self.search_metrics)
            else:
                average_latency = 0.0
                total_searches = 0

            # Calculate strategy performance by vector type
            strategy_performance = {}
            vector_types = ["stats", "context", "value", "combined"]
            
            for vector_type in vector_types:
                vector_metrics = [m for m in self.search_metrics if m.vector_type == vector_type]
                if vector_metrics:
                    avg_latency = statistics.mean([m.latency_ms for m in vector_metrics])
                    strategy_performance[vector_type] = {
                        "latency": avg_latency,
                        "count": len(vector_metrics),
                        "accuracy": statistics.mean([m.accuracy_score for m in vector_metrics])
                    }
                else:
                    strategy_performance[vector_type] = {
                        "latency": 0.0,
                        "count": 0,
                        "accuracy": 0.0
                    }

            # Get recent performance trends
            recent_metrics = [m for m in self.search_metrics 
                            if m.timestamp > datetime.now() - timedelta(hours=1)]
            
            if recent_metrics:
                recent_latency = statistics.mean([m.latency_ms for m in recent_metrics])
                latency_trend = "stable"
                if recent_latency > average_latency * 1.2:
                    latency_trend = "increasing"
                elif recent_latency < average_latency * 0.8:
                    latency_trend = "decreasing"
            else:
                latency_trend = "stable"

            return {
                "total_searches": total_searches,
                "average_latency": average_latency,
                "strategy_performance": strategy_performance,
                "latency_trend": latency_trend,
                "cache_hit_rate": await self._calculate_cache_hit_rate(),
                "active_alerts": len(self.active_alerts),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting search analytics: {e}")
            return {
                "total_searches": 0,
                "average_latency": 0.0,
                "strategy_performance": {},
                "latency_trend": "unknown",
                "cache_hit_rate": 0.0,
                "active_alerts": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
