"""
API module for multi-camera tracking system.
Provides REST and WebSocket endpoints.
"""

from .main import app, manager, set_pipeline_state, run_api
from .main import broadcast_track_update, broadcast_event, broadcast_stats_update
from .schemas import *


__all__ = [
    'app',
    'manager',
    'set_pipeline_state',
    'run_api',
    'broadcast_track_update',
    'broadcast_event',
    'broadcast_stats_update',
]

