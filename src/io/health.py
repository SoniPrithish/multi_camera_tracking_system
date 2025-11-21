"""
Stream health monitoring and diagnostics.
Provides real-time health metrics and alerts for video streams.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from .streams import StreamHealth, StreamStatus

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthAlert:
    """Health alert event."""
    timestamp: float
    camera_id: str
    level: AlertLevel
    message: str
    metric_name: str
    metric_value: float


@dataclass
class HealthThresholds:
    """Configurable health thresholds."""
    min_fps: float = 10.0
    max_latency_ms: float = 200.0
    max_drop_rate: float = 0.1  # 10%
    max_reconnects: int = 5
    stale_frame_timeout_s: float = 5.0


class HealthMonitor:
    """
    Monitors health of multiple video streams.
    Generates alerts when thresholds are exceeded.
    """
    
    def __init__(
        self,
        thresholds: Optional[HealthThresholds] = None,
        check_interval: float = 1.0,
    ):
        """
        Initialize health monitor.
        
        Args:
            thresholds: Health thresholds for alerts
            check_interval: How often to check health (seconds)
        """
        self.thresholds = thresholds or HealthThresholds()
        self.check_interval = check_interval
        
        # Stream health cache
        self._health_cache: Dict[str, StreamHealth] = {}
        self._alerts: List[HealthAlert] = []
        self._alert_callbacks: List[Callable[[HealthAlert], None]] = []
        
        # Metrics history
        self._fps_history: Dict[str, List[float]] = defaultdict(list)
        self._latency_history: Dict[str, List[float]] = defaultdict(list)
        
        # Background monitoring
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stream_getter: Optional[Callable[[], Dict[str, StreamHealth]]] = None
    
    def start(self, stream_health_getter: Callable[[], Dict[str, StreamHealth]]):
        """
        Start background health monitoring.
        
        Args:
            stream_health_getter: Function that returns current stream health
        """
        self._stream_getter = stream_health_getter
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Health monitor started")
    
    def stop(self):
        """Stop background monitoring."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("Health monitor stopped")
    
    def add_alert_callback(self, callback: Callable[[HealthAlert], None]):
        """Register a callback for health alerts."""
        self._alert_callbacks.append(callback)
    
    def update(self, health_data: Dict[str, StreamHealth]):
        """
        Update health data and check for issues.
        
        Args:
            health_data: Current health for all streams
        """
        self._health_cache = health_data
        
        for cam_id, health in health_data.items():
            self._check_stream_health(cam_id, health)
    
    def _check_stream_health(self, cam_id: str, health: StreamHealth):
        """Check health metrics for a single stream."""
        now = time.time()
        
        # Check connection status
        if health.status == StreamStatus.FAILED:
            self._emit_alert(cam_id, AlertLevel.CRITICAL, 
                           "Stream connection failed", 
                           "status", 0)
        elif health.status == StreamStatus.RECONNECTING:
            self._emit_alert(cam_id, AlertLevel.WARNING,
                           "Stream reconnecting",
                           "reconnect_count", health.reconnect_count)
        
        # Check FPS
        if health.fps < self.thresholds.min_fps and health.status == StreamStatus.CONNECTED:
            self._emit_alert(cam_id, AlertLevel.WARNING,
                           f"Low FPS: {health.fps:.1f}",
                           "fps", health.fps)
        
        # Check drop rate
        if health.drop_rate > self.thresholds.max_drop_rate:
            self._emit_alert(cam_id, AlertLevel.WARNING,
                           f"High frame drop rate: {health.drop_rate*100:.1f}%",
                           "drop_rate", health.drop_rate)
        
        # Check reconnects
        if health.reconnect_count > self.thresholds.max_reconnects:
            self._emit_alert(cam_id, AlertLevel.ERROR,
                           f"Excessive reconnects: {health.reconnect_count}",
                           "reconnect_count", health.reconnect_count)
        
        # Check stale frames
        if health.last_frame_time > 0:
            stale_time = now - health.last_frame_time
            if stale_time > self.thresholds.stale_frame_timeout_s:
                self._emit_alert(cam_id, AlertLevel.ERROR,
                               f"No frames for {stale_time:.1f}s",
                               "stale_time", stale_time)
        
        # Track history
        self._fps_history[cam_id].append(health.fps)
        if len(self._fps_history[cam_id]) > 60:
            self._fps_history[cam_id].pop(0)
    
    def _emit_alert(
        self,
        cam_id: str,
        level: AlertLevel,
        message: str,
        metric_name: str,
        metric_value: float
    ):
        """Create and emit a health alert."""
        alert = HealthAlert(
            timestamp=time.time(),
            camera_id=cam_id,
            level=level,
            message=message,
            metric_name=metric_name,
            metric_value=metric_value
        )
        
        self._alerts.append(alert)
        if len(self._alerts) > 1000:
            self._alerts.pop(0)
        
        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
        
        # Log alert
        log_func = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical,
        }.get(level, logger.info)
        
        log_func(f"[{cam_id}] {message}")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            if self._stream_getter is not None:
                try:
                    health_data = self._stream_getter()
                    self.update(health_data)
                except Exception as e:
                    logger.error(f"Health check error: {e}")
            
            time.sleep(self.check_interval)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get health summary for all streams."""
        summary = {
            'streams': {},
            'alerts_count': len(self._alerts),
            'recent_alerts': self._alerts[-10:] if self._alerts else []
        }
        
        for cam_id, health in self._health_cache.items():
            fps_history = self._fps_history.get(cam_id, [])
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
            
            summary['streams'][cam_id] = {
                'status': health.status.value,
                'fps': health.fps,
                'avg_fps': avg_fps,
                'frame_count': health.frame_count,
                'drop_count': health.drop_count,
                'drop_rate': health.drop_rate,
                'reconnect_count': health.reconnect_count,
                'error': health.error_message
            }
        
        return summary
    
    def get_alerts(
        self,
        cam_id: Optional[str] = None,
        level: Optional[AlertLevel] = None,
        since: Optional[float] = None
    ) -> List[HealthAlert]:
        """
        Get filtered alerts.
        
        Args:
            cam_id: Filter by camera ID
            level: Filter by alert level
            since: Filter alerts after this timestamp
            
        Returns:
            List of matching alerts
        """
        alerts = self._alerts
        
        if cam_id is not None:
            alerts = [a for a in alerts if a.camera_id == cam_id]
        
        if level is not None:
            alerts = [a for a in alerts if a.level == level]
        
        if since is not None:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        return alerts

