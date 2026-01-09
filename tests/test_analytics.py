"""
Tests for analytics module.
"""

import pytest
import time

from src.analytics import (
    Event, EventType, CrossDirection,
    ZoneEntryEvent, ZoneExitEvent, LineCrossEvent,
    AnalyticsZone, CountingLine, AnalyticsZoneManager,
    TrackAnalytics, TrackState,
    EventAggregator, ZoneStats, LineStats
)


class TestEvents:
    """Tests for event types."""
    
    def test_event_to_dict(self):
        event = Event(
            event_id='test123',
            event_type=EventType.ZONE_ENTRY,
            timestamp=time.time(),
            camera_id='cam1',
            global_id=1,
            track_id=5
        )
        
        d = event.to_dict()
        
        assert d['event_id'] == 'test123'
        assert d['event_type'] == 'zone_entry'
        assert d['camera_id'] == 'cam1'
    
    def test_zone_entry_event(self):
        event = ZoneEntryEvent(
            event_id='e1',
            event_type=EventType.ZONE_ENTRY,
            timestamp=time.time(),
            camera_id='cam1',
            zone_id='z1',
            zone_name='Entrance'
        )
        
        assert event.metadata['zone_id'] == 'z1'
        assert event.metadata['zone_name'] == 'Entrance'
    
    def test_line_cross_event(self):
        event = LineCrossEvent(
            event_id='e2',
            event_type=EventType.LINE_CROSS,
            timestamp=time.time(),
            camera_id='cam1',
            line_id='l1',
            line_name='Counter',
            direction=CrossDirection.POSITIVE
        )
        
        assert event.metadata['direction'] == 'positive'


class TestAnalyticsZone:
    """Tests for AnalyticsZone."""
    
    def test_init(self):
        zone = AnalyticsZone(
            zone_id='z1',
            camera_id='cam1',
            name='Test Zone',
            polygon=[[0, 0], [100, 0], [100, 100], [0, 100]]
        )
        assert zone is not None
    
    def test_contains_bbox(self):
        zone = AnalyticsZone(
            zone_id='z1',
            camera_id='cam1',
            name='Test Zone',
            polygon=[[0, 0], [100, 0], [100, 100], [0, 100]]
        )
        
        assert zone.contains_bbox((25, 25, 50, 50)) == True
        assert zone.contains_bbox((150, 150, 50, 50)) == False


class TestCountingLine:
    """Tests for CountingLine."""
    
    def test_init(self):
        line = CountingLine(
            line_id='l1',
            camera_id='cam1',
            name='Counter',
            start_point=(0, 50),
            end_point=(100, 50)
        )
        assert line is not None
    
    def test_get_side(self):
        line = CountingLine(
            line_id='l1',
            camera_id='cam1',
            name='Counter',
            start_point=(0, 50),
            end_point=(100, 50)
        )
        
        # Point above line
        side_above = line.get_side((50, 20))
        # Point below line
        side_below = line.get_side((50, 80))
        
        assert side_above != side_below
    
    def test_check_crossing(self):
        line = CountingLine(
            line_id='l1',
            camera_id='cam1',
            name='Counter',
            start_point=(0, 50),
            end_point=(100, 50)
        )
        
        # Cross from top to bottom
        crossing = line.check_crossing((50, 20), (50, 80))
        assert crossing is not None
        
        # No crossing - both on same side
        no_crossing = line.check_crossing((50, 20), (50, 30))
        assert no_crossing is None


class TestEventAggregator:
    """Tests for EventAggregator."""
    
    def test_init(self):
        aggregator = EventAggregator()
        assert aggregator is not None
    
    def test_process_zone_entry(self):
        aggregator = EventAggregator()
        
        event = ZoneEntryEvent(
            event_id='e1',
            event_type=EventType.ZONE_ENTRY,
            timestamp=time.time(),
            camera_id='cam1',
            zone_id='z1',
            zone_name='Test'
        )
        
        aggregator.process_event(event)
        
        stats = aggregator.get_zone_stats('z1')
        assert stats is not None
        assert stats['entry_count'] == 1
    
    def test_process_line_cross(self):
        aggregator = EventAggregator()
        
        event = LineCrossEvent(
            event_id='e1',
            event_type=EventType.LINE_CROSS,
            timestamp=time.time(),
            camera_id='cam1',
            line_id='l1',
            line_name='Counter',
            direction=CrossDirection.POSITIVE
        )
        
        aggregator.process_event(event)
        
        stats = aggregator.get_line_stats('l1')
        assert stats is not None
        assert stats['in_count'] == 1
    
    def test_get_summary(self):
        aggregator = EventAggregator()
        
        summary = aggregator.get_summary()
        
        assert 'uptime_seconds' in summary
        assert 'total_events' in summary
        assert 'zones' in summary
        assert 'lines' in summary
    
    def test_reset(self):
        aggregator = EventAggregator()
        
        event = ZoneEntryEvent(
            event_id='e1',
            event_type=EventType.ZONE_ENTRY,
            timestamp=time.time(),
            camera_id='cam1',
            zone_id='z1',
            zone_name='Test'
        )
        aggregator.process_event(event)
        
        aggregator.reset()
        
        assert aggregator.total_events == 0
        assert len(aggregator.zone_stats) == 0


class TestTrackState:
    """Tests for TrackState."""
    
    def test_init(self):
        state = TrackState(track_id=1, camera_id='cam1')
        assert state is not None
    
    def test_update_position(self):
        state = TrackState(track_id=1, camera_id='cam1')
        
        state.update_position((100, 100, 50, 100))
        
        assert state.last_bbox == (100, 100, 50, 100)
        assert state.last_position == (125, 150)  # Center
    
    def test_position_history(self):
        state = TrackState(track_id=1, camera_id='cam1')
        
        state.update_position((100, 100, 50, 100))
        state.update_position((110, 105, 50, 100))
        
        assert state.prev_bbox == (100, 100, 50, 100)
        assert state.last_bbox == (110, 105, 50, 100)

