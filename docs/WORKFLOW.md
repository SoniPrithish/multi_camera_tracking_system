# Multi-Camera Tracking System - Workflow Guide

This document describes the end-to-end workflow for using the multi-camera tracking system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Pipeline](#running-the-pipeline)
5. [API and Dashboard](#api-and-dashboard)
6. [Evaluation](#evaluation)
7. [Customization](#customization)

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/SoniPrithish/multi_camera_tracking_system.git
cd multi_camera_tracking_system

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run with demo config
python -m src.app --config configs/demo.yaml
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.x (optional, for GPU acceleration)
- FFmpeg (for video processing)

### Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# For GPU acceleration (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install onnxruntime-gpu
```

### Verify Installation

```bash
# Run tests
pytest tests/ -v

# Check imports
python -c "from src.detection import build_detector; print('OK')"
```

---

## Configuration

### Main Configuration File

Create a YAML configuration file (see `configs/demo.yaml` for example):

```yaml
# Input streams
inputs:
  - path: rtsp://camera1.local/stream
    camera_id: cam1
    queue_size: 30
  - path: data/videos/cam2.mp4
    camera_id: cam2

# Output settings
output:
  video_dir: outputs/videos
  tracks_path: outputs/tracks.jsonl
  manifest_path: outputs/manifest.json
  stats_path: outputs/stats.json
  show: true  # Display live visualization

# Module configuration
modules:
  detector:
    name: yolov8
    model: yolov8n.pt
    conf: 0.25
    device: auto  # auto, cpu, cuda
  
  tracker:
    name: bytetrack
    track_high_thresh: 0.5
    track_low_thresh: 0.1
    track_buffer: 30
  
  reid:
    name: deep
    model: osnet_x0_25
    ema_alpha: 0.9
    sim_thresh: 0.6

# Camera topology (optional)
camera_graph:
  cameras:
    - id: cam1
      name: Entrance
    - id: cam2
      name: Hallway
  transitions:
    - from: cam1
      to: cam2
      min_time: 5
      max_time: 30

# Analytics zones (optional)
analytics:
  zones:
    - id: entrance_zone
      camera_id: cam1
      name: Entrance Area
      polygon: [[0, 300], [200, 300], [200, 480], [0, 480]]
      type: entry
  lines:
    - id: counter_line
      camera_id: cam1
      name: People Counter
      start: [0, 240]
      end: [640, 240]

# Runtime settings
runtime:
  max_frames: null  # null for unlimited
  skip_frames: 1
  min_fps: 10
  max_latency_ms: 200
  draw:
    show_box: true
    show_id: true
    show_cam: true
    show_tracklet: true
```

### Environment Variables

```bash
export MCTRACK_CONFIG=configs/production.yaml
export MCTRACK_LOG_LEVEL=INFO
export MCTRACK_DEVICE=cuda
```

---

## Running the Pipeline

### Basic Usage

```bash
# Run with default config
python -m src.app

# Run with custom config
python -m src.app --config configs/my_config.yaml
```

### Programmatic Usage

```python
import yaml
from src.pipeline import MultiCamPipeline

# Load configuration
with open('configs/demo.yaml') as f:
    cfg = yaml.safe_load(f)

# Create and run pipeline
pipeline = MultiCamPipeline(cfg)
pipeline.run()

# Get statistics
stats = pipeline.get_stats()
print(f"Processed {stats['frame_count']} frames at {stats['fps']:.1f} FPS")
```

### Output Files

After running, you'll find:

- `outputs/videos/{camera_id}.mp4` - Annotated video for each camera
- `outputs/tracks.jsonl` - Track data in JSON Lines format
- `outputs/manifest.json` - Global identity manifest
- `outputs/stats.json` - Analytics statistics

---

## API and Dashboard

### Starting the API Server

```bash
# Start FastAPI server
uvicorn src.api:app --host 0.0.0.0 --port 8000

# With auto-reload for development
uvicorn src.api:app --reload --port 8000
```

### Dashboard

Access the dashboard at: `http://localhost:8000`

Features:
- Multi-camera grid view
- Real-time track visualization
- Zone analytics
- Event timeline
- Statistics display

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/streams` | GET | Get stream status |
| `/api/tracks/{camera_id}` | GET | Get current tracks |
| `/api/events` | GET | Get recent events |
| `/api/analytics/summary` | GET | Get analytics summary |
| `/api/zones` | GET | Get zone configuration |
| `/ws` | WebSocket | Real-time updates |

### WebSocket Messages

```javascript
// Connect to WebSocket
const ws = new WebSocket('ws://localhost:8000/ws');

// Subscribe to updates
ws.send(JSON.stringify({ type: 'subscribe' }));

// Receive updates
ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === 'track_update') {
        // Handle track update
    } else if (msg.type === 'event') {
        // Handle event
    }
};
```

---

## Evaluation

### Download Datasets

```bash
# List available datasets
python scripts/download_datasets.py --list

# Download specific dataset
python scripts/download_datasets.py --dataset pets2009
python scripts/download_datasets.py --dataset wildtrack
```

### Run Benchmark

```bash
# Run evaluation on dataset
python -m src.evaluation.benchmark \
    --dataset data/datasets/pets2009 \
    --output outputs/benchmark \
    --max-frames 1000

# Results are saved to outputs/benchmark/
```

### Metrics

The evaluation computes:

- **MOTA** (Multiple Object Tracking Accuracy)
- **IDF1** (ID F1 Score)
- **HOTA** (Higher Order Tracking Accuracy)
- **Precision/Recall**
- **ID Switches**
- **Handoff Accuracy** (for multi-camera)

---

## Customization

### Custom Detector

```python
from src.detection import build_detector

class MyDetector:
    def detect(self, frame):
        # Your detection logic
        return [{'bbox': (x, y, w, h), 'score': conf, 'cls': 0}]

# Register and use
from src.detection import build_detector
# Add to __init__.py or use directly
```

### Custom Tracker

```python
from src.tracking import ByteTracker

class MyTracker(ByteTracker):
    def update(self, cam_id, detections):
        # Custom tracking logic
        return super().update(cam_id, detections)
```

### Custom ReID

```python
from src.reid import DeepReID

class MyReID(DeepReID):
    def encode(self, frame, tracks):
        # Custom embedding extraction
        pass
```

### Adding Zones

```python
from src.analytics import AnalyticsZoneManager

manager = AnalyticsZoneManager()
manager.add_zone(
    zone_id='my_zone',
    camera_id='cam1',
    name='Custom Zone',
    polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
    zone_type='roi'
)
```

---

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
```bash
# Use smaller model
detector:
  name: yolov8
  model: yolov8n.pt  # Use nano model

# Or run on CPU
detector:
  device: cpu
```

**2. Low FPS**
```yaml
# Reduce resolution or skip frames
runtime:
  skip_frames: 2
  
inputs:
  - path: ...
    resize: [640, 360]
```

**3. RTSP stream disconnects**
```yaml
inputs:
  - path: rtsp://...
    reconnect_delay: 2.0
    max_reconnects: 10
```

### Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## Support

- GitHub Issues: https://github.com/SoniPrithish/multi_camera_tracking_system/issues
- Documentation: See `docs/` folder

