"""
Event types for zone-based analytics.
Defines entry, exit, dwell, and line crossing events.
"""

import time
from enum import Enum
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import json


class EventType(Enum):
    """Types of analytics events."""
    ZONE_ENTRY = "zone_entry"
    ZONE_EXIT = "zone_exit"
    ZONE_DWELL = "zone_dwell"
    LINE_CROSS = "line_cross"
    CAMERA_ENTRY = "camera_entry"
    CAMERA_EXIT = "camera_exit"
    HANDOFF = "handoff"


class CrossDirection(Enum):
    """Direction of line crossing."""
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    TOP_TO_BOTTOM = "top_to_bottom"
    BOTTOM_TO_TOP = "bottom_to_top"
    POSITIVE = "positive"
    NEGATIVE = "negative"


@dataclass
class Event:
    """Base event class."""
    event_id: str
    event_type: EventType
    timestamp: float
    camera_id: str
    global_id: Optional[int] = None
    track_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to serializable dict."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'camera_id': self.camera_id,
            'global_id': self.global_id,
            'track_id': self.track_id,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Event':
        """Create from dict."""
        return cls(
            event_id=data['event_id'],
            event_type=EventType(data['event_type']),
            timestamp=data['timestamp'],
            camera_id=data['camera_id'],
            global_id=data.get('global_id'),
            track_id=data.get('track_id'),
            metadata=data.get('metadata', {})
        )


@dataclass
class ZoneEntryEvent(Event):
    """Event when a person enters a zone."""
    zone_id: str = ""
    zone_name: str = ""
    
    def __post_init__(self):
        self.event_type = EventType.ZONE_ENTRY
        self.metadata['zone_id'] = self.zone_id
        self.metadata['zone_name'] = self.zone_name


@dataclass
class ZoneExitEvent(Event):
    """Event when a person exits a zone."""
    zone_id: str = ""
    zone_name: str = ""
    dwell_time: float = 0.0  # Time spent in zone
    
    def __post_init__(self):
        self.event_type = EventType.ZONE_EXIT
        self.metadata['zone_id'] = self.zone_id
        self.metadata['zone_name'] = self.zone_name
        self.metadata['dwell_time'] = self.dwell_time


@dataclass
class ZoneDwellEvent(Event):
    """Event for dwell time in a zone (periodic update)."""
    zone_id: str = ""
    zone_name: str = ""
    dwell_time: float = 0.0
    
    def __post_init__(self):
        self.event_type = EventType.ZONE_DWELL
        self.metadata['zone_id'] = self.zone_id
        self.metadata['zone_name'] = self.zone_name
        self.metadata['dwell_time'] = self.dwell_time


@dataclass
class LineCrossEvent(Event):
    """Event when a person crosses a counting line."""
    line_id: str = ""
    line_name: str = ""
    direction: CrossDirection = CrossDirection.POSITIVE
    
    def __post_init__(self):
        self.event_type = EventType.LINE_CROSS
        self.metadata['line_id'] = self.line_id
        self.metadata['line_name'] = self.line_name
        self.metadata['direction'] = self.direction.value


@dataclass
class CameraEntryEvent(Event):
    """Event when a person first appears in a camera."""
    entry_zone: Optional[str] = None
    
    def __post_init__(self):
        self.event_type = EventType.CAMERA_ENTRY
        if self.entry_zone:
            self.metadata['entry_zone'] = self.entry_zone


@dataclass
class CameraExitEvent(Event):
    """Event when a person leaves a camera view."""
    exit_zone: Optional[str] = None
    time_in_view: float = 0.0
    
    def __post_init__(self):
        self.event_type = EventType.CAMERA_EXIT
        if self.exit_zone:
            self.metadata['exit_zone'] = self.exit_zone
        self.metadata['time_in_view'] = self.time_in_view


@dataclass
class HandoffEvent(Event):
    """Event when a person is handed off between cameras."""
    source_camera: str = ""
    target_camera: str = ""
    transfer_time: float = 0.0
    confidence: float = 0.0
    
    def __post_init__(self):
        self.event_type = EventType.HANDOFF
        self.metadata['source_camera'] = self.source_camera
        self.metadata['target_camera'] = self.target_camera
        self.metadata['transfer_time'] = self.transfer_time
        self.metadata['confidence'] = self.confidence


def generate_event_id() -> str:
    """Generate unique event ID."""
    import uuid
    return str(uuid.uuid4())[:8]

