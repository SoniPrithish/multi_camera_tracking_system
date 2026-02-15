# Multi-Camera People Tracking System

Real-time multi-camera person detection, tracking, re-identification, and cross-camera association system with a live web dashboard. Built for GPU-accelerated inference on consumer hardware.

## Architecture

```
Camera Feeds ──► Ingest ──► YOLOv8 Detection ──► ByteTrack ──► OSNet Re-ID ──► Cross-Camera Association ──► Event Detection ──► Dashboard
  (MP4/RTSP)    (async      (FP16 batch GPU)    (Kalman       (FP16 batch      (Hungarian matching,         (Entry/Exit,        (FastAPI +
                 workers,                         filter,       GPU, EMA          cosine similarity,           Dwell,              WebSocket
                 bounded                          stable        smoothing,        time-of-flight gating,       Line Crossing)      live video)
                 queues)                          local IDs)    skip-frame)       global ID registry)
```

### Pipeline Stages

| Stage | Module | Description |
|-------|--------|-------------|
| 1. Ingest | `src/io/streams.py` | Async threaded workers per camera, bounded queues (drop stale frames), auto-reconnect with backoff, supports RTSP/MP4/webcam |
| 2. Detection | `src/detection/yolov8.py` | YOLOv8-nano with FP16 half-precision, batch inference across all cameras, person-only filtering, tunable confidence/NMS |
| 3. Tracking | `src/tracking/byte_tracker.py` | ByteTrack algorithm with 8-state Kalman filter, two-stage association (high/low confidence), track lifecycle management |
| 4. Re-ID | `src/reid/osnet.py` | Standalone OSNet (MobileNetV3-based) feature extractor, 512-d L2-normalized embeddings, FP16 inference, EMA smoothing, skip-frame optimization |
| 5. Association | `src/association/cross_camera.py` | Cross-camera global ID assignment using cosine similarity + Hungarian matching, time-of-flight gating, gallery with TTL and maturity gate |
| 6. Events | `src/events/event_detector.py` | Entry/Exit, Dwell (configurable threshold), Line Crossing (direction-aware), per-person deduplication |
| 7. Serving | `src/server.py` | FastAPI server with WebSocket live video streams, real-time dashboard, REST API for stats and events |

## Performance

Benchmarked on **NVIDIA RTX 4060 Ti 16GB**, 24 CPU threads, 62GB RAM, 6 simultaneous camera streams:

| Metric | Value |
|--------|-------|
| FPS per camera | **27.5** |
| Global throughput | **164.9 FPS** |
| End-to-end latency | **26.7 ms** |
| Detection (YOLOv8 FP16) | 20.2 ms (batch) |
| Tracking (ByteTrack) | 2.8 ms |
| Re-ID (OSNet FP16) | ~0 ms (amortized, skip-frame) |
| Association (Hungarian) | 0.1 ms |
| Events | 0.6 ms |
| Memory (RSS) | 3.8 GB (stable, no leaks) |
| Stability test | 30 min passed |

## Quick Start

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on CUDA 8.9)
- 8GB+ GPU VRAM (16GB recommended for 6+ cameras)

### Installation

```bash
# Clone the repository
git clone git@github.com:SoniPrithish/multi_camera_tracking_system.git
cd multi_camera_tracking_system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Add Video Sources

Place MP4 files in `data/samples/` or configure RTSP URLs in the config:

```bash
data/samples/
├── cam1.mp4
├── cam2.mp4
├── cam3.mp4
├── cam4.mp4
├── cam5.mp4
└── cam6.mp4
```

### Run

```bash
# Start the production server
python -m src.server --config configs/demo.yaml

# Open dashboard
# http://localhost:8000
```

## Dashboard

The web dashboard (served at `http://localhost:8000`) provides:

- **Live multi-camera video** — WebSocket-streamed JPEG feeds from all cameras simultaneously
- **Global ID overlay** — colored bounding boxes with unique global IDs that persist across cameras
- **Real-time metrics** — FPS per camera, detection/tracking latency breakdown, gallery stats
- **Cross-camera stats** — number of global IDs, total handoff matches
- **Event timeline** — scrolling log of Entry, Exit, Dwell, and Line Crossing events
- **Per-camera counts** — detection and track counts per camera view

### REST API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard HTML |
| `/api/stats` | GET | Real-time pipeline statistics (JSON) |
| `/api/events` | GET | Recent events list (JSON) |
| `/ws/{camera_id}` | WS | Live JPEG video stream for a camera |
| `/ws/dashboard-stats` | WS | Real-time stats push |

## Configuration

All parameters are in `configs/demo.yaml`:

```yaml
# Camera inputs (MP4 or RTSP)
inputs:
  - path: data/samples/cam1.mp4
    camera_id: cam1
  - path: rtsp://user:pass@192.168.1.100:554/stream
    camera_id: entrance

# Detection
modules:
  detector: yolov8
  detector_model: yolov8n.pt    # nano model for speed
  detector_conf: 0.35           # confidence threshold
  detector_iou: 0.45            # NMS IoU threshold
  detector_device: "0"          # GPU device ("cpu" for CPU)

# Tracking
  tracker: byte
  tracker_max_age: 30           # frames before track deletion
  tracker_min_hits: 3           # hits before track confirmation

# Re-ID
  reid: deep
  reid_model: osnet_x0_25
  reid_sim_thresh: 0.45

# Cross-Camera Association
association:
  assoc_sim_thresh: 0.55        # cosine similarity threshold
  assoc_min_transit_sec: 0      # min transit time between cameras
  assoc_max_transit_sec: 120    # max transit time
  assoc_gallery_ttl: 90         # seconds before inactive entries pruned
  assoc_confirm_hits: 2         # hits before global ID confirmed

# Events
events:
  entry_exit:
    enabled: true
  dwell:
    enabled: true
    threshold_sec: 10
  line_crossing:
    enabled: true
    lines: []                   # define per-camera line segments
```

## Project Structure

```
multi_camera_tracking_system/
├── configs/
│   └── demo.yaml              # Main configuration
├── data/
│   └── samples/               # Video files (not tracked in git)
├── scripts/
│   ├── test_stage2_live.py    # Stage 2 test (detection)
│   ├── test_stage3_live.py    # Stage 3 test (tracking)
│   ├── test_stage4_live.py    # Stage 4 test (Re-ID)
│   ├── test_stage5_live.py    # Stage 5 test (cross-camera)
│   ├── test_stage6_live.py    # Stage 6 test (events)
│   └── preview_cams.py        # Camera preview utility
├── src/
│   ├── server.py              # Production FastAPI server (Stage 7)
│   ├── io/
│   │   └── streams.py         # Multi-camera async ingest (Stage 1)
│   ├── detection/
│   │   ├── yolov8.py          # YOLOv8 FP16 batch detector (Stage 2)
│   │   └── detector.py        # Dummy detector for testing
│   ├── tracking/
│   │   ├── byte_tracker.py    # ByteTrack implementation (Stage 3)
│   │   └── kalman_filter.py   # 8-state Kalman filter
│   ├── reid/
│   │   ├── osnet.py           # OSNet feature extractor (Stage 4)
│   │   ├── deep.py            # Deep Re-ID module
│   │   └── cosine.py          # Cosine similarity utilities
│   ├── association/
│   │   └── cross_camera.py    # Cross-camera associator (Stage 5)
│   └── events/
│       └── event_detector.py  # Event detection (Stage 6)
├── tests/
│   └── test_association.py
├── requirements.txt
└── .gitignore
```

## Key Design Decisions

- **Batch GPU Inference**: All cameras processed in a single GPU batch call for both detection and Re-ID, maximizing throughput
- **Skip-Frame Re-ID**: Embeddings extracted every 8th frame to reduce GPU load; cached embeddings used in between
- **Bounded Queues**: Each camera's ingest queue holds only 2 frames — stale frames are dropped, ensuring the pipeline always processes the freshest data
- **EMA Embeddings**: Gallery embeddings use exponential moving average for stability against appearance changes
- **Maturity Gate**: Global IDs require multiple confirmation hits before becoming match candidates, reducing false positives
- **Time-of-Flight Gating**: Cross-camera matches constrained by configurable transit time windows between camera pairs
- **Dedup Ring Buffer**: Events use a bounded deduplication set with automatic cleanup to prevent memory growth

## License

This project is for educational and research purposes.
