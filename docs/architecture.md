# System Architecture

This document describes the architecture of the Multi-Camera People Tracking System.

## Overview

The system is designed as a modular pipeline that processes multiple video streams in real-time, maintaining consistent person identities across cameras.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Multi-Camera Tracking System                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Camera 1   │    │   Camera 2   │    │   Camera N   │               │
│  │   (RTSP)     │    │    (MP4)     │    │   (Webcam)   │               │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘               │
│         │                   │                   │                        │
│         └─────────┬─────────┴─────────┬─────────┘                        │
│                   │                   │                                  │
│           ┌───────▼───────────────────▼───────┐                         │
│           │        Stream Ingest Layer         │                         │
│           │  - Bounded Queues                  │                         │
│           │  - Auto-reconnect                  │                         │
│           │  - Health Monitoring               │                         │
│           └───────────────┬───────────────────┘                         │
│                           │                                              │
│           ┌───────────────▼───────────────────┐                         │
│           │      Per-Camera Processing         │                         │
│           │  ┌─────────────────────────────┐  │                         │
│           │  │    Person Detection         │  │                         │
│           │  │    (YOLOv8 / ONNX)          │  │                         │
│           │  └──────────────┬──────────────┘  │                         │
│           │                 │                  │                         │
│           │  ┌──────────────▼──────────────┐  │                         │
│           │  │    Object Tracking          │  │                         │
│           │  │    (ByteTrack)              │  │                         │
│           │  └──────────────┬──────────────┘  │                         │
│           │                 │                  │                         │
│           │  ┌──────────────▼──────────────┐  │                         │
│           │  │    ReID Embedding           │  │                         │
│           │  │    (OSNet)                  │  │                         │
│           │  └──────────────┬──────────────┘  │                         │
│           └─────────────────┼─────────────────┘                         │
│                             │                                            │
│           ┌─────────────────▼─────────────────┐                         │
│           │    Cross-Camera Association       │                         │
│           │  ┌────────────┐ ┌────────────┐   │                         │
│           │  │ Camera     │ │   Zone     │   │                         │
│           │  │ Graph      │ │   Gate     │   │                         │
│           │  └─────┬──────┘ └─────┬──────┘   │                         │
│           │        │              │          │                          │
│           │  ┌─────▼──────────────▼──────┐   │                         │
│           │  │   Hungarian Matcher       │   │                         │
│           │  └───────────┬───────────────┘   │                         │
│           │              │                    │                         │
│           │  ┌───────────▼───────────────┐   │                         │
│           │  │   Global ID Registry      │   │                         │
│           │  └───────────────────────────┘   │                         │
│           └─────────────────┬─────────────────┘                         │
│                             │                                            │
│           ┌─────────────────▼─────────────────┐                         │
│           │        Analytics Engine           │                         │
│           │  - Zone Entry/Exit                │                         │
│           │  - Dwell Time                     │                         │
│           │  - Line Crossing                  │                         │
│           │  - Event Aggregation              │                         │
│           └─────────────────┬─────────────────┘                         │
│                             │                                            │
│           ┌─────────────────▼─────────────────┐                         │
│           │         Serving Layer             │                         │
│           │  ┌─────────┐  ┌─────────────┐    │                         │
│           │  │ FastAPI │  │  Dashboard  │    │                         │
│           │  │ REST/WS │  │   (Jinja2)  │    │                         │
│           │  └─────────┘  └─────────────┘    │                         │
│           └───────────────────────────────────┘                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Module Descriptions

### 1. Stream Ingest Layer (`src/io/`)

Handles video input from multiple sources:

- **MultiVideoReader**: Manages multiple `VideoStream` instances
- **VideoStream**: Single stream with threaded capture and bounded queue
- **HealthMonitor**: Tracks FPS, drop rate, latency for each stream

Key features:
- Bounded queues with drop-oldest policy to maintain low latency
- Automatic reconnection for RTSP streams
- Adaptive resolution based on processing latency

### 2. Detection Module (`src/detection/`)

Person detection using deep learning:

- **YOLOv8Detector**: Full Ultralytics integration
- **ONNXDetector**: ONNX Runtime for CPU inference
- **DummyDetector**: Synthetic detections for testing

Output: List of `{bbox: (x,y,w,h), score: float, cls: int}`

### 3. Tracking Module (`src/tracking/`)

Single-camera multi-object tracking:

- **ByteTracker**: Two-stage association (high/low confidence)
- **KalmanFilter**: Motion prediction
- **CentroidTracker**: Simple IoU baseline

Output: List of `{tid: int, bbox: tuple, score: float}`

### 4. ReID Module (`src/reid/`)

Appearance-based re-identification:

- **DeepReID**: OSNet encoder with 512-dim embeddings
- **CosineReID**: Color histogram baseline
- EMA smoothing for stable appearance prototypes

Output: `{track_id: embedding}` dictionary

### 5. Association Module (`src/association/`)

Cross-camera identity matching:

- **CameraGraph**: Camera topology with transfer time windows
- **ZoneGate**: Entry/exit zone constraints
- **CrossCameraMatcher**: Hungarian algorithm matching
- **GlobalRegistry**: Global ID management

### 6. Analytics Module (`src/analytics/`)

Zone-based behavioral analytics:

- **AnalyticsZone**: Polygon region definitions
- **CountingLine**: Directional counting lines
- **TrackAnalytics**: Per-track zone state machine
- **EventAggregator**: Real-time statistics

### 7. API Module (`src/api/`)

REST API and dashboard:

- **FastAPI**: REST endpoints + WebSocket
- **Jinja2**: Dashboard templates
- **WebSocket Manager**: Real-time client updates

## Data Flow

```
Frame → Detection → Tracking → ReID → Association → Analytics → Output
                                          │
                                          ▼
                                   Global Registry
                                          │
                                          ▼
                                    API/Dashboard
```

### Per-Frame Processing

1. **Ingest**: Read frame from queue
2. **Detect**: Run YOLOv8 to get person bounding boxes
3. **Track**: Associate detections with existing tracks (ByteTrack)
4. **Encode**: Extract ReID embeddings for each track
5. **Associate**: Match with global identities using camera graph + zones
6. **Update**: Update global registry with new observations
7. **Analytics**: Check zone interactions, emit events
8. **Output**: Write annotated video and track data

## Configuration

Configuration is YAML-based with hierarchical structure:

```yaml
inputs:          # Stream sources
output:          # Output paths
modules:         # Component selection
camera_graph:    # Camera topology
analytics:       # Zones and lines
runtime:         # Processing parameters
```

## Performance Considerations

### GPU Utilization
- Detection and ReID run on GPU (when available)
- Tracking and association run on CPU

### Memory Management
- Bounded queues prevent memory growth
- Gallery pruning for ReID
- Identity timeout for registry cleanup

### Latency Optimization
- Skip frames when behind
- Adaptive resolution scaling
- Batched ReID inference

## Extension Points

1. **Custom Detector**: Implement `detect(frame) -> List[Dict]`
2. **Custom Tracker**: Extend `ByteTracker` or implement interface
3. **Custom ReID**: Implement `encode()` and `assign_global_ids()`
4. **Custom Analytics**: Add zones/lines via `AnalyticsZoneManager`

