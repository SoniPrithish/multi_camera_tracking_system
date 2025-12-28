"""
Per-track analytics state machine.
Tracks zone interactions and generates events.
"""

import time
from typing import Dict, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from .events import (
    Event, EventType, CrossDirection,
    ZoneEntryEvent, ZoneExitEvent, ZoneDwellEvent,
    LineCrossEvent, CameraEntryEvent, CameraExitEvent,
    generate_event_id
)
from .zones import AnalyticsZone, CountingLine, AnalyticsZoneManager

logger = logging.getLogger(__name__)


@dataclass
class TrackZoneState:
    """State of a track within a zone."""
    zone_id: str
    entry_time: float
    last_update_time: float
    last_dwell_event_time: float = 0.0
    
    @property
    def dwell_time(self) -> float:
        """Current dwell time in zone."""
        return time.time() - self.entry_time


@dataclass
class TrackState:
    """Complete state for a tracked person."""
    track_id: int
    camera_id: str
    global_id: Optional[int] = None
    
    # Position history
    last_bbox: Optional[Tuple[float, float, float, float]] = None
    prev_bbox: Optional[Tuple[float, float, float, float]] = None
    last_position: Optional[Tuple[float, float]] = None
    prev_position: Optional[Tuple[float, float]] = None
    
    # Timing
    first_seen_time: float = field(default_factory=time.time)
    last_seen_time: float = field(default_factory=time.time)
    
    # Zone states
    current_zones: Dict[str, TrackZoneState] = field(default_factory=dict)
    zone_history: List[Tuple[str, float, float]] = field(default_factory=list)  # (zone_id, entry, exit)
    
    # Line crossing
    lines_crossed: Set[str] = field(default_factory=set)
    
    def update_position(self, bbox: Tuple[float, float, float, float]):
        """Update position with new bbox."""
        self.prev_bbox = self.last_bbox
        self.last_bbox = bbox
        
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        
        self.prev_position = self.last_position
        self.last_position = (cx, cy)
        self.last_seen_time = time.time()


class TrackAnalytics:
    """
    Manages per-track analytics and generates events.
    """
    
    def __init__(
        self,
        zone_manager: AnalyticsZoneManager,
        dwell_update_interval: float = 10.0,  # Seconds between dwell updates
        track_timeout: float = 30.0,  # Seconds before considering track lost
    ):
        """
        Initialize track analytics.
        
        Args:
            zone_manager: Zone and line manager
            dwell_update_interval: How often to emit dwell events
            track_timeout: Time after which a track is considered lost
        """
        self.zone_manager = zone_manager
        self.dwell_update_interval = dwell_update_interval
        self.track_timeout = track_timeout
        
        # Track states per camera
        self.track_states: Dict[str, Dict[int, TrackState]] = defaultdict(dict)
        
        # Event callbacks
        self.event_callbacks: List[Callable[[Event], None]] = []
    
    def add_event_callback(self, callback: Callable[[Event], None]):
        """Add callback for event notifications."""
        self.event_callbacks.append(callback)
    
    def _emit_event(self, event: Event):
        """Emit an event to all callbacks."""
        for callback in self.event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    def update(
        self,
        camera_id: str,
        tracks: List[Dict],
        global_ids: Optional[Dict[int, int]] = None
    ) -> List[Event]:
        """
        Update analytics with new track data.
        
        Args:
            camera_id: Camera identifier
            tracks: List of tracks with 'tid' and 'bbox'
            global_ids: Optional mapping of track_id to global_id
            
        Returns:
            List of generated events
        """
        events = []
        now = time.time()
        global_ids = global_ids or {}
        
        current_track_ids = set()
        
        for track in tracks:
            tid = track['tid']
            bbox = track['bbox']
            gid = global_ids.get(tid)
            
            current_track_ids.add(tid)
            
            # Get or create track state
            if tid not in self.track_states[camera_id]:
                state = TrackState(
                    track_id=tid,
                    camera_id=camera_id,
                    global_id=gid,
                    first_seen_time=now,
                    last_seen_time=now
                )
                self.track_states[camera_id][tid] = state
                
                # Camera entry event
                event = CameraEntryEvent(
                    event_id=generate_event_id(),
                    event_type=EventType.CAMERA_ENTRY,
                    timestamp=now,
                    camera_id=camera_id,
                    global_id=gid,
                    track_id=tid
                )
                events.append(event)
                self._emit_event(event)
            
            state = self.track_states[camera_id][tid]
            state.global_id = gid
            state.update_position(bbox)
            
            # Check zone interactions
            zone_events = self._check_zones(state, camera_id, now)
            events.extend(zone_events)
            
            # Check line crossings
            line_events = self._check_lines(state, camera_id, now)
            events.extend(line_events)
        
        # Check for lost tracks
        lost_events = self._check_lost_tracks(camera_id, current_track_ids, now)
        events.extend(lost_events)
        
        return events
    
    def _check_zones(
        self,
        state: TrackState,
        camera_id: str,
        now: float
    ) -> List[Event]:
        """Check zone entry/exit/dwell."""
        events = []
        
        if state.last_bbox is None:
            return events
        
        zones = self.zone_manager.get_zones_for_camera(camera_id)
        current_zone_ids = set()
        
        for zone in zones:
            if zone.contains_bbox(state.last_bbox):
                current_zone_ids.add(zone.zone_id)
                
                if zone.zone_id not in state.current_zones:
                    # Zone entry
                    state.current_zones[zone.zone_id] = TrackZoneState(
                        zone_id=zone.zone_id,
                        entry_time=now,
                        last_update_time=now
                    )
                    
                    event = ZoneEntryEvent(
                        event_id=generate_event_id(),
                        event_type=EventType.ZONE_ENTRY,
                        timestamp=now,
                        camera_id=camera_id,
                        global_id=state.global_id,
                        track_id=state.track_id,
                        zone_id=zone.zone_id,
                        zone_name=zone.name
                    )
                    events.append(event)
                    self._emit_event(event)
                
                elif zone.track_dwell:
                    # Check for dwell update
                    zone_state = state.current_zones[zone.zone_id]
                    dwell = zone_state.dwell_time
                    
                    if (dwell >= zone.dwell_threshold and 
                        now - zone_state.last_dwell_event_time >= self.dwell_update_interval):
                        
                        event = ZoneDwellEvent(
                            event_id=generate_event_id(),
                            event_type=EventType.ZONE_DWELL,
                            timestamp=now,
                            camera_id=camera_id,
                            global_id=state.global_id,
                            track_id=state.track_id,
                            zone_id=zone.zone_id,
                            zone_name=zone.name,
                            dwell_time=dwell
                        )
                        events.append(event)
                        self._emit_event(event)
                        zone_state.last_dwell_event_time = now
        
        # Check for zone exits
        for zone_id in list(state.current_zones.keys()):
            if zone_id not in current_zone_ids:
                zone_state = state.current_zones.pop(zone_id)
                zone = self.zone_manager.zones.get(zone_id)
                
                dwell = now - zone_state.entry_time
                state.zone_history.append((zone_id, zone_state.entry_time, now))
                
                event = ZoneExitEvent(
                    event_id=generate_event_id(),
                    event_type=EventType.ZONE_EXIT,
                    timestamp=now,
                    camera_id=camera_id,
                    global_id=state.global_id,
                    track_id=state.track_id,
                    zone_id=zone_id,
                    zone_name=zone.name if zone else zone_id,
                    dwell_time=dwell
                )
                events.append(event)
                self._emit_event(event)
        
        return events
    
    def _check_lines(
        self,
        state: TrackState,
        camera_id: str,
        now: float
    ) -> List[Event]:
        """Check line crossings."""
        events = []
        
        if state.prev_position is None or state.last_position is None:
            return events
        
        lines = self.zone_manager.get_lines_for_camera(camera_id)
        
        for line in lines:
            crossing = line.check_crossing(state.prev_position, state.last_position)
            
            if crossing is not None:
                direction = CrossDirection.POSITIVE if crossing == "positive" else CrossDirection.NEGATIVE
                
                event = LineCrossEvent(
                    event_id=generate_event_id(),
                    event_type=EventType.LINE_CROSS,
                    timestamp=now,
                    camera_id=camera_id,
                    global_id=state.global_id,
                    track_id=state.track_id,
                    line_id=line.line_id,
                    line_name=line.name,
                    direction=direction
                )
                events.append(event)
                self._emit_event(event)
                state.lines_crossed.add(line.line_id)
        
        return events
    
    def _check_lost_tracks(
        self,
        camera_id: str,
        current_track_ids: Set[int],
        now: float
    ) -> List[Event]:
        """Check for tracks that have been lost."""
        events = []
        
        for tid, state in list(self.track_states[camera_id].items()):
            if tid not in current_track_ids:
                time_since = now - state.last_seen_time
                
                if time_since >= self.track_timeout:
                    # Track is lost - emit camera exit
                    event = CameraExitEvent(
                        event_id=generate_event_id(),
                        event_type=EventType.CAMERA_EXIT,
                        timestamp=now,
                        camera_id=camera_id,
                        global_id=state.global_id,
                        track_id=state.track_id,
                        time_in_view=now - state.first_seen_time
                    )
                    events.append(event)
                    self._emit_event(event)
                    
                    # Cleanup
                    del self.track_states[camera_id][tid]
        
        return events
    
    def get_track_state(self, camera_id: str, track_id: int) -> Optional[TrackState]:
        """Get state for a specific track."""
        return self.track_states.get(camera_id, {}).get(track_id)
    
    def get_active_tracks(self, camera_id: str) -> List[TrackState]:
        """Get all active tracks for a camera."""
        return list(self.track_states.get(camera_id, {}).values())
    
    def reset(self, camera_id: Optional[str] = None):
        """Reset track states."""
        if camera_id:
            self.track_states[camera_id].clear()
        else:
            self.track_states.clear()

