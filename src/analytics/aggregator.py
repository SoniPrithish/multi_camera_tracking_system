"""
Event aggregation for real-time statistics.
Computes counts, rates, and distributions from events.
"""

import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import json
import os
import threading

from .events import Event, EventType, CrossDirection

logger = logging.getLogger(__name__)


@dataclass
class ZoneStats:
    """Statistics for a single zone."""
    zone_id: str
    zone_name: str
    camera_id: str
    
    entry_count: int = 0
    exit_count: int = 0
    current_occupancy: int = 0
    
    total_dwell_time: float = 0.0
    dwell_count: int = 0
    min_dwell: float = float('inf')
    max_dwell: float = 0.0
    
    # Time-windowed counts
    entries_last_minute: int = 0
    entries_last_hour: int = 0
    
    def record_entry(self):
        self.entry_count += 1
        self.current_occupancy += 1
        self.entries_last_minute += 1
        self.entries_last_hour += 1
    
    def record_exit(self, dwell_time: float):
        self.exit_count += 1
        self.current_occupancy = max(0, self.current_occupancy - 1)
        
        if dwell_time > 0:
            self.total_dwell_time += dwell_time
            self.dwell_count += 1
            self.min_dwell = min(self.min_dwell, dwell_time)
            self.max_dwell = max(self.max_dwell, dwell_time)
    
    @property
    def avg_dwell(self) -> float:
        if self.dwell_count == 0:
            return 0.0
        return self.total_dwell_time / self.dwell_count
    
    def to_dict(self) -> Dict:
        return {
            'zone_id': self.zone_id,
            'zone_name': self.zone_name,
            'camera_id': self.camera_id,
            'entry_count': self.entry_count,
            'exit_count': self.exit_count,
            'current_occupancy': self.current_occupancy,
            'avg_dwell_time': self.avg_dwell,
            'min_dwell_time': self.min_dwell if self.min_dwell != float('inf') else 0,
            'max_dwell_time': self.max_dwell,
            'entries_last_minute': self.entries_last_minute,
            'entries_last_hour': self.entries_last_hour
        }


@dataclass
class LineStats:
    """Statistics for a counting line."""
    line_id: str
    line_name: str
    camera_id: str
    
    positive_count: int = 0  # "in" direction
    negative_count: int = 0  # "out" direction
    
    positive_last_minute: int = 0
    negative_last_minute: int = 0
    
    def record_crossing(self, direction: str):
        if direction in ("positive", CrossDirection.POSITIVE.value):
            self.positive_count += 1
            self.positive_last_minute += 1
        else:
            self.negative_count += 1
            self.negative_last_minute += 1
    
    @property
    def net_count(self) -> int:
        return self.positive_count - self.negative_count
    
    @property
    def total_count(self) -> int:
        return self.positive_count + self.negative_count
    
    def to_dict(self) -> Dict:
        return {
            'line_id': self.line_id,
            'line_name': self.line_name,
            'camera_id': self.camera_id,
            'in_count': self.positive_count,
            'out_count': self.negative_count,
            'net_count': self.net_count,
            'total_count': self.total_count,
            'in_last_minute': self.positive_last_minute,
            'out_last_minute': self.negative_last_minute
        }


@dataclass
class CameraStats:
    """Statistics for a camera."""
    camera_id: str
    
    total_entries: int = 0
    total_exits: int = 0
    current_count: int = 0
    unique_identities: set = field(default_factory=set)
    
    entries_last_minute: int = 0
    entries_last_hour: int = 0
    
    handoffs_in: int = 0
    handoffs_out: int = 0
    
    def record_entry(self, global_id: Optional[int] = None):
        self.total_entries += 1
        self.current_count += 1
        self.entries_last_minute += 1
        self.entries_last_hour += 1
        if global_id:
            self.unique_identities.add(global_id)
    
    def record_exit(self):
        self.total_exits += 1
        self.current_count = max(0, self.current_count - 1)
    
    def to_dict(self) -> Dict:
        return {
            'camera_id': self.camera_id,
            'total_entries': self.total_entries,
            'total_exits': self.total_exits,
            'current_count': self.current_count,
            'unique_identities': len(self.unique_identities),
            'entries_last_minute': self.entries_last_minute,
            'entries_last_hour': self.entries_last_hour,
            'handoffs_in': self.handoffs_in,
            'handoffs_out': self.handoffs_out
        }


class EventAggregator:
    """
    Aggregates events into real-time statistics.
    """
    
    def __init__(
        self,
        window_size_minutes: int = 60,
        persist_path: Optional[str] = None
    ):
        """
        Initialize aggregator.
        
        Args:
            window_size_minutes: Size of sliding window for rate calculations
            persist_path: Path to persist statistics
        """
        self.window_size_minutes = window_size_minutes
        self.persist_path = persist_path
        
        # Statistics
        self.zone_stats: Dict[str, ZoneStats] = {}
        self.line_stats: Dict[str, LineStats] = {}
        self.camera_stats: Dict[str, CameraStats] = {}
        
        # Event history (for windowed calculations)
        self.event_history: List[Tuple[float, Event]] = []
        self.max_history_size = 100000
        
        # Global stats
        self.total_events = 0
        self.start_time = time.time()
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Periodic cleanup
        self._last_cleanup = time.time()
        self._cleanup_interval = 60.0  # seconds
    
    def process_event(self, event: Event):
        """
        Process an event and update statistics.
        
        Args:
            event: Event to process
        """
        with self._lock:
            self.total_events += 1
            self.event_history.append((time.time(), event))
            
            # Process by event type
            if event.event_type == EventType.ZONE_ENTRY:
                self._process_zone_entry(event)
            elif event.event_type == EventType.ZONE_EXIT:
                self._process_zone_exit(event)
            elif event.event_type == EventType.LINE_CROSS:
                self._process_line_cross(event)
            elif event.event_type == EventType.CAMERA_ENTRY:
                self._process_camera_entry(event)
            elif event.event_type == EventType.CAMERA_EXIT:
                self._process_camera_exit(event)
            elif event.event_type == EventType.HANDOFF:
                self._process_handoff(event)
            
            # Periodic cleanup
            if time.time() - self._last_cleanup > self._cleanup_interval:
                self._cleanup()
    
    def _process_zone_entry(self, event: Event):
        """Process zone entry event."""
        zone_id = event.metadata.get('zone_id', '')
        if zone_id not in self.zone_stats:
            self.zone_stats[zone_id] = ZoneStats(
                zone_id=zone_id,
                zone_name=event.metadata.get('zone_name', zone_id),
                camera_id=event.camera_id
            )
        self.zone_stats[zone_id].record_entry()
    
    def _process_zone_exit(self, event: Event):
        """Process zone exit event."""
        zone_id = event.metadata.get('zone_id', '')
        if zone_id not in self.zone_stats:
            self.zone_stats[zone_id] = ZoneStats(
                zone_id=zone_id,
                zone_name=event.metadata.get('zone_name', zone_id),
                camera_id=event.camera_id
            )
        dwell_time = event.metadata.get('dwell_time', 0.0)
        self.zone_stats[zone_id].record_exit(dwell_time)
    
    def _process_line_cross(self, event: Event):
        """Process line crossing event."""
        line_id = event.metadata.get('line_id', '')
        if line_id not in self.line_stats:
            self.line_stats[line_id] = LineStats(
                line_id=line_id,
                line_name=event.metadata.get('line_name', line_id),
                camera_id=event.camera_id
            )
        direction = event.metadata.get('direction', 'positive')
        self.line_stats[line_id].record_crossing(direction)
    
    def _process_camera_entry(self, event: Event):
        """Process camera entry event."""
        cam_id = event.camera_id
        if cam_id not in self.camera_stats:
            self.camera_stats[cam_id] = CameraStats(camera_id=cam_id)
        self.camera_stats[cam_id].record_entry(event.global_id)
    
    def _process_camera_exit(self, event: Event):
        """Process camera exit event."""
        cam_id = event.camera_id
        if cam_id not in self.camera_stats:
            self.camera_stats[cam_id] = CameraStats(camera_id=cam_id)
        self.camera_stats[cam_id].record_exit()
    
    def _process_handoff(self, event: Event):
        """Process handoff event."""
        source = event.metadata.get('source_camera', '')
        target = event.metadata.get('target_camera', '')
        
        if source and source in self.camera_stats:
            self.camera_stats[source].handoffs_out += 1
        if target and target in self.camera_stats:
            self.camera_stats[target].handoffs_in += 1
    
    def _cleanup(self):
        """Clean up old events and reset windowed counters."""
        now = time.time()
        cutoff_minute = now - 60
        cutoff_hour = now - 3600
        
        # Reset minute counters
        for stats in self.zone_stats.values():
            stats.entries_last_minute = 0
        for stats in self.line_stats.values():
            stats.positive_last_minute = 0
            stats.negative_last_minute = 0
        for stats in self.camera_stats.values():
            stats.entries_last_minute = 0
        
        # Recount from history
        for ts, event in self.event_history:
            if ts >= cutoff_minute:
                if event.event_type == EventType.ZONE_ENTRY:
                    zone_id = event.metadata.get('zone_id', '')
                    if zone_id in self.zone_stats:
                        self.zone_stats[zone_id].entries_last_minute += 1
                elif event.event_type == EventType.LINE_CROSS:
                    line_id = event.metadata.get('line_id', '')
                    if line_id in self.line_stats:
                        direction = event.metadata.get('direction', 'positive')
                        if direction in ("positive", CrossDirection.POSITIVE.value):
                            self.line_stats[line_id].positive_last_minute += 1
                        else:
                            self.line_stats[line_id].negative_last_minute += 1
                elif event.event_type == EventType.CAMERA_ENTRY:
                    if event.camera_id in self.camera_stats:
                        self.camera_stats[event.camera_id].entries_last_minute += 1
        
        # Trim history
        if len(self.event_history) > self.max_history_size:
            self.event_history = self.event_history[-self.max_history_size // 2:]
        
        self._last_cleanup = now
    
    def get_summary(self) -> Dict:
        """Get overall summary statistics."""
        with self._lock:
            return {
                'uptime_seconds': time.time() - self.start_time,
                'total_events': self.total_events,
                'zones': {zid: s.to_dict() for zid, s in self.zone_stats.items()},
                'lines': {lid: s.to_dict() for lid, s in self.line_stats.items()},
                'cameras': {cid: s.to_dict() for cid, s in self.camera_stats.items()},
                'global': {
                    'total_zone_entries': sum(z.entry_count for z in self.zone_stats.values()),
                    'total_line_crossings': sum(l.total_count for l in self.line_stats.values()),
                    'total_unique_people': len(set().union(*[c.unique_identities for c in self.camera_stats.values()])) if self.camera_stats else 0,
                    'current_total_occupancy': sum(z.current_occupancy for z in self.zone_stats.values())
                }
            }
    
    def get_zone_stats(self, zone_id: str) -> Optional[Dict]:
        """Get statistics for a specific zone."""
        with self._lock:
            if zone_id in self.zone_stats:
                return self.zone_stats[zone_id].to_dict()
            return None
    
    def get_line_stats(self, line_id: str) -> Optional[Dict]:
        """Get statistics for a specific line."""
        with self._lock:
            if line_id in self.line_stats:
                return self.line_stats[line_id].to_dict()
            return None
    
    def get_camera_stats(self, camera_id: str) -> Optional[Dict]:
        """Get statistics for a specific camera."""
        with self._lock:
            if camera_id in self.camera_stats:
                return self.camera_stats[camera_id].to_dict()
            return None
    
    def get_recent_events(self, limit: int = 100) -> List[Dict]:
        """Get recent events."""
        with self._lock:
            recent = self.event_history[-limit:]
            return [event.to_dict() for _, event in recent]
    
    def save(self, path: Optional[str] = None):
        """Save statistics to file."""
        path = path or self.persist_path
        if not path:
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with self._lock:
            data = {
                'timestamp': time.time(),
                'summary': self.get_summary()
            }
            
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
        
        logger.info(f"Saved statistics to {path}")
    
    def reset(self):
        """Reset all statistics."""
        with self._lock:
            self.zone_stats.clear()
            self.line_stats.clear()
            self.camera_stats.clear()
            self.event_history.clear()
            self.total_events = 0
            self.start_time = time.time()

