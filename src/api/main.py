"""
FastAPI application for multi-camera tracking system.
Provides REST API and WebSocket endpoints.
"""

import os
import time
import json
import asyncio
import logging
from typing import Dict, List, Optional, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    StreamsResponse, StreamHealth, StreamStatus,
    TracksResponse, Track, BoundingBox,
    EventsResponse, EventBase,
    AnalyticsSummary, ZoneStats, LineStats, CameraStats, GlobalStats,
    ZonesConfigResponse, ZoneConfig, LineConfig,
    PipelineStatus,
    WSMessage, WSTrackUpdate, WSEventNotification, WSStatsUpdate
)

logger = logging.getLogger(__name__)

# Global state (will be populated by pipeline)
app_state = {
    'pipeline': None,
    'analytics': None,
    'aggregator': None,
    'zone_manager': None,
    'registry': None,
    'health_monitor': None,
    'start_time': time.time()
}


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.add(connection)
        
        for conn in disconnected:
            self.active_connections.discard(conn)
    
    async def send_personal(self, websocket: WebSocket, message: dict):
        await websocket.send_json(message)


manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Multi-Camera Tracking API...")
    app_state['start_time'] = time.time()
    yield
    logger.info("Shutting down Multi-Camera Tracking API...")


# Create FastAPI app
app = FastAPI(
    title="Multi-Camera Tracking API",
    description="Real-time multi-camera person tracking with cross-camera re-identification",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
static_path = os.path.join(os.path.dirname(__file__), "static")
templates_path = os.path.join(os.path.dirname(__file__), "templates")

if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

templates = Jinja2Templates(directory=templates_path) if os.path.exists(templates_path) else None


# --- REST Endpoints ---

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve dashboard."""
    if templates:
        return templates.TemplateResponse("dashboard.html", {"request": request})
    return HTMLResponse("<h1>Multi-Camera Tracking API</h1><p>Dashboard not configured.</p>")


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "uptime": time.time() - app_state['start_time'],
        "pipeline_active": app_state['pipeline'] is not None
    }


@app.get("/api/streams", response_model=StreamsResponse)
async def get_streams():
    """Get status of all video streams."""
    if app_state['health_monitor']:
        health_data = app_state['health_monitor'].get_summary()
        streams = {}
        for cam_id, h in health_data.get('streams', {}).items():
            streams[cam_id] = StreamHealth(
                camera_id=cam_id,
                status=StreamStatus(h.get('status', 'disconnected')),
                fps=h.get('fps', 0),
                frame_count=h.get('frame_count', 0),
                drop_count=h.get('drop_count', 0),
                drop_rate=h.get('drop_rate', 0),
                reconnect_count=h.get('reconnect_count', 0),
                error_message=h.get('error', '')
            )
        return StreamsResponse(streams=streams)
    
    return StreamsResponse(streams={})


@app.get("/api/tracks/{camera_id}", response_model=TracksResponse)
async def get_tracks(camera_id: str):
    """Get current tracks for a camera."""
    tracks = []
    
    if app_state['analytics']:
        track_states = app_state['analytics'].get_active_tracks(camera_id)
        for state in track_states:
            if state.last_bbox:
                x, y, w, h = state.last_bbox
                tracks.append(Track(
                    track_id=state.track_id,
                    global_id=state.global_id,
                    camera_id=camera_id,
                    bbox=BoundingBox(x=x, y=y, width=w, height=h),
                    score=1.0
                ))
    
    return TracksResponse(
        camera_id=camera_id,
        tracks=tracks,
        timestamp=time.time()
    )


@app.get("/api/events", response_model=EventsResponse)
async def get_events(limit: int = 100, camera_id: Optional[str] = None):
    """Get recent events."""
    events = []
    
    if app_state['aggregator']:
        recent = app_state['aggregator'].get_recent_events(limit)
        for e in recent:
            if camera_id and e.get('camera_id') != camera_id:
                continue
            events.append(EventBase(**e))
    
    return EventsResponse(events=events, total_count=len(events))


@app.get("/api/analytics/summary")
async def get_analytics_summary():
    """Get analytics summary."""
    if app_state['aggregator']:
        summary = app_state['aggregator'].get_summary()
        return summary
    
    return {
        "uptime_seconds": time.time() - app_state['start_time'],
        "total_events": 0,
        "zones": {},
        "lines": {},
        "cameras": {},
        "global": {
            "total_zone_entries": 0,
            "total_line_crossings": 0,
            "total_unique_people": 0,
            "current_total_occupancy": 0
        }
    }


@app.get("/api/analytics/zones/{zone_id}")
async def get_zone_analytics(zone_id: str):
    """Get analytics for a specific zone."""
    if app_state['aggregator']:
        stats = app_state['aggregator'].get_zone_stats(zone_id)
        if stats:
            return stats
    raise HTTPException(status_code=404, detail="Zone not found")


@app.get("/api/analytics/lines/{line_id}")
async def get_line_analytics(line_id: str):
    """Get analytics for a specific line."""
    if app_state['aggregator']:
        stats = app_state['aggregator'].get_line_stats(line_id)
        if stats:
            return stats
    raise HTTPException(status_code=404, detail="Line not found")


@app.get("/api/zones", response_model=ZonesConfigResponse)
async def get_zones_config():
    """Get zone and line configurations."""
    zones = []
    lines = []
    
    if app_state['zone_manager']:
        zm = app_state['zone_manager']
        for zone in zm.zones.values():
            zones.append(ZoneConfig(
                zone_id=zone.zone_id,
                camera_id=zone.camera_id,
                name=zone.name,
                polygon=[(float(p[0]), float(p[1])) for p in zone.polygon],
                zone_type=zone.zone_type,
                track_dwell=zone.track_dwell,
                dwell_threshold=zone.dwell_threshold
            ))
        
        for line in zm.lines.values():
            lines.append(LineConfig(
                line_id=line.line_id,
                camera_id=line.camera_id,
                name=line.name,
                start=(float(line.start[0]), float(line.start[1])),
                end=(float(line.end[0]), float(line.end[1])),
                positive_direction=line.positive_direction
            ))
    
    return ZonesConfigResponse(zones=zones, lines=lines)


@app.get("/api/registry/stats")
async def get_registry_stats():
    """Get global identity registry statistics."""
    if app_state['registry']:
        return app_state['registry'].get_stats()
    return {"current_identities": 0, "total_created": 0}


@app.get("/api/pipeline/status", response_model=PipelineStatus)
async def get_pipeline_status():
    """Get pipeline status."""
    return PipelineStatus(
        running=app_state['pipeline'] is not None,
        fps=0.0,  # Would be updated by pipeline
        total_frames=0,
        active_cameras=len(app_state.get('streams', {})),
        active_tracks=0
    )


# --- WebSocket Endpoints ---

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    
    try:
        while True:
            # Receive and handle messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            msg_type = message.get('type', '')
            
            if msg_type == 'ping':
                await manager.send_personal(websocket, {
                    'type': 'pong',
                    'timestamp': time.time()
                })
            
            elif msg_type == 'subscribe':
                # Client wants to subscribe to updates
                await manager.send_personal(websocket, {
                    'type': 'subscribed',
                    'timestamp': time.time()
                })
            
            elif msg_type == 'get_stats':
                # Send current stats
                if app_state['aggregator']:
                    summary = app_state['aggregator'].get_summary()
                    await manager.send_personal(websocket, {
                        'type': 'stats_update',
                        'data': summary,
                        'timestamp': time.time()
                    })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# --- Broadcast Functions ---

async def broadcast_track_update(camera_id: str, tracks: List[dict]):
    """Broadcast track update to all WebSocket clients."""
    message = {
        'type': 'track_update',
        'camera_id': camera_id,
        'tracks': tracks,
        'timestamp': time.time()
    }
    await manager.broadcast(message)


async def broadcast_event(event: dict):
    """Broadcast event to all WebSocket clients."""
    message = {
        'type': 'event',
        'event': event,
        'timestamp': time.time()
    }
    await manager.broadcast(message)


async def broadcast_stats_update(summary: dict):
    """Broadcast stats update to all WebSocket clients."""
    message = {
        'type': 'stats_update',
        'data': summary,
        'timestamp': time.time()
    }
    await manager.broadcast(message)


# --- Helper Functions ---

def set_pipeline_state(
    pipeline=None,
    analytics=None,
    aggregator=None,
    zone_manager=None,
    registry=None,
    health_monitor=None
):
    """Set global state from pipeline."""
    if pipeline is not None:
        app_state['pipeline'] = pipeline
    if analytics is not None:
        app_state['analytics'] = analytics
    if aggregator is not None:
        app_state['aggregator'] = aggregator
    if zone_manager is not None:
        app_state['zone_manager'] = zone_manager
    if registry is not None:
        app_state['registry'] = registry
    if health_monitor is not None:
        app_state['health_monitor'] = health_monitor


def run_api(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)

