"""
Pydantic schemas for API request/response models.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from datetime import datetime


# --- Enums ---

class StreamStatus(str, Enum):
    disconnected = "disconnected"
    connecting = "connecting"
    connected = "connected"
    reconnecting = "reconnecting"
    failed = "failed"


class EventType(str, Enum):
    zone_entry = "zone_entry"
    zone_exit = "zone_exit"
    zone_dwell = "zone_dwell"
    line_cross = "line_cross"
    camera_entry = "camera_entry"
    camera_exit = "camera_exit"
    handoff = "handoff"


# --- Stream Schemas ---

class StreamConfig(BaseModel):
    camera_id: str
    path: str
    queue_size: int = 30
    reconnect_delay: float = 2.0
    max_reconnects: int = 10


class StreamHealth(BaseModel):
    camera_id: str
    status: StreamStatus
    fps: float
    frame_count: int
    drop_count: int
    drop_rate: float
    reconnect_count: int
    error_message: str = ""


class StreamsResponse(BaseModel):
    streams: Dict[str, StreamHealth]


# --- Track Schemas ---

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class Track(BaseModel):
    track_id: int
    global_id: Optional[int] = None
    camera_id: str
    bbox: BoundingBox
    score: float


class TracksResponse(BaseModel):
    camera_id: str
    tracks: List[Track]
    timestamp: float


# --- Event Schemas ---

class EventBase(BaseModel):
    event_id: str
    event_type: EventType
    timestamp: float
    camera_id: str
    global_id: Optional[int] = None
    track_id: Optional[int] = None
    metadata: Dict[str, Any] = {}


class EventsResponse(BaseModel):
    events: List[EventBase]
    total_count: int


# --- Analytics Schemas ---

class ZoneStats(BaseModel):
    zone_id: str
    zone_name: str
    camera_id: str
    entry_count: int
    exit_count: int
    current_occupancy: int
    avg_dwell_time: float
    min_dwell_time: float
    max_dwell_time: float
    entries_last_minute: int


class LineStats(BaseModel):
    line_id: str
    line_name: str
    camera_id: str
    in_count: int
    out_count: int
    net_count: int
    total_count: int
    in_last_minute: int
    out_last_minute: int


class CameraStats(BaseModel):
    camera_id: str
    total_entries: int
    total_exits: int
    current_count: int
    unique_identities: int
    entries_last_minute: int
    handoffs_in: int
    handoffs_out: int


class GlobalStats(BaseModel):
    total_zone_entries: int
    total_line_crossings: int
    total_unique_people: int
    current_total_occupancy: int


class AnalyticsSummary(BaseModel):
    uptime_seconds: float
    total_events: int
    zones: Dict[str, ZoneStats]
    lines: Dict[str, LineStats]
    cameras: Dict[str, CameraStats]
    global_stats: GlobalStats = Field(alias="global")

    class Config:
        populate_by_name = True


# --- Zone Configuration Schemas ---

class ZoneConfig(BaseModel):
    zone_id: str
    camera_id: str
    name: str
    polygon: List[Tuple[float, float]]
    zone_type: str = "general"
    track_dwell: bool = True
    dwell_threshold: float = 5.0


class LineConfig(BaseModel):
    line_id: str
    camera_id: str
    name: str
    start: Tuple[float, float]
    end: Tuple[float, float]
    positive_direction: str = "in"


class ZonesConfigResponse(BaseModel):
    zones: List[ZoneConfig]
    lines: List[LineConfig]


# --- Pipeline Schemas ---

class PipelineConfig(BaseModel):
    detector: Dict[str, Any] = {"name": "yolov8"}
    tracker: Dict[str, Any] = {"name": "bytetrack"}
    reid: Dict[str, Any] = {"name": "deep"}


class PipelineStatus(BaseModel):
    running: bool
    fps: float
    total_frames: int
    active_cameras: int
    active_tracks: int


# --- WebSocket Messages ---

class WSMessage(BaseModel):
    type: str
    data: Any
    timestamp: float


class WSTrackUpdate(BaseModel):
    type: str = "track_update"
    camera_id: str
    tracks: List[Track]
    timestamp: float


class WSEventNotification(BaseModel):
    type: str = "event"
    event: EventBase


class WSStatsUpdate(BaseModel):
    type: str = "stats_update"
    summary: AnalyticsSummary

