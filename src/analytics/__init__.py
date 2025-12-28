"""
Analytics module for zone-based events and statistics.
Provides entry/exit detection, dwell time, line crossing, and aggregation.
"""

from .events import (
    Event, EventType, CrossDirection,
    ZoneEntryEvent, ZoneExitEvent, ZoneDwellEvent,
    LineCrossEvent, CameraEntryEvent, CameraExitEvent, HandoffEvent,
    generate_event_id
)
from .zones import AnalyticsZone, CountingLine, AnalyticsZoneManager
from .tracker_analytics import TrackAnalytics, TrackState, TrackZoneState
from .aggregator import EventAggregator, ZoneStats, LineStats, CameraStats


__all__ = [
    # Events
    'Event',
    'EventType',
    'CrossDirection',
    'ZoneEntryEvent',
    'ZoneExitEvent',
    'ZoneDwellEvent',
    'LineCrossEvent',
    'CameraEntryEvent',
    'CameraExitEvent',
    'HandoffEvent',
    'generate_event_id',
    # Zones
    'AnalyticsZone',
    'CountingLine',
    'AnalyticsZoneManager',
    # Tracking
    'TrackAnalytics',
    'TrackState',
    'TrackZoneState',
    # Aggregation
    'EventAggregator',
    'ZoneStats',
    'LineStats',
    'CameraStats',
]

