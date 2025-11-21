"""
I/O module for multi-camera tracking system.
Handles video stream input, output, and health monitoring.
"""

from .streams import (
    VideoStream,
    MultiVideoReader,
    StreamConfig,
    StreamHealth,
    StreamStatus,
    AdaptiveResolutionManager
)
from .sink import VideoWriterMux, TracksWriter
from .health import (
    HealthMonitor,
    HealthThresholds,
    HealthAlert,
    AlertLevel
)


__all__ = [
    'VideoStream',
    'MultiVideoReader',
    'StreamConfig',
    'StreamHealth',
    'StreamStatus',
    'AdaptiveResolutionManager',
    'VideoWriterMux',
    'TracksWriter',
    'HealthMonitor',
    'HealthThresholds',
    'HealthAlert',
    'AlertLevel',
]

