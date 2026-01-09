# Multi-Camera People Tracking System

[![CI](https://github.com/SoniPrithish/multi_camera_tracking_system/actions/workflows/ci.yml/badge.svg)](https://github.com/SoniPrithish/multi_camera_tracking_system/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A real-time, end-to-end multi-camera people tracking system that maintains consistent global identities as people move between camera views. Designed for practical deployment in retail stores, campuses, transit hubs, and other multi-camera environments.

## Features

- **Multi-Camera Support**: Process 2-6 concurrent video streams (RTSP, MP4, webcam)
- **Real-Time Tracking**: 15-30 FPS on GPU, with CPU fallback
- **Cross-Camera ReID**: Consistent global IDs across camera views
- **Zone Analytics**: Entry/exit detection, dwell time, line crossing
- **Web Dashboard**: Real-time visualization and statistics
- **Evaluation Tools**: MOTA, IDF1, HOTA metrics on standard datasets

## Quick Start

```bash
# Clone repository
git clone https://github.com/SoniPrithish/multi_camera_tracking_system.git
cd multi_camera_tracking_system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run demo
python -m src.app --config configs/demo.yaml
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Video Streams                             │
│         (RTSP/MP4/Webcam) × N cameras                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────▼─────────────┐
    │   Per-Camera Processing   │
    │  ┌─────────────────────┐  │
    │  │  YOLOv8 Detection   │  │
    │  └──────────┬──────────┘  │
    │             │             │
    │  ┌──────────▼──────────┐  │
    │  │  ByteTrack Tracker  │  │
    │  └──────────┬──────────┘  │
    │             │             │
    │  ┌──────────▼──────────┐  │
    │  │   OSNet ReID        │  │
    │  └──────────┬──────────┘  │
    └─────────────┼─────────────┘
                  │
    ┌─────────────▼─────────────┐
    │  Cross-Camera Association │
    │  - Camera Graph           │
    │  - Zone Gating            │
    │  - Hungarian Matching     │
    │  - Global ID Registry     │
    └─────────────┬─────────────┘
                  │
    ┌─────────────▼─────────────┐
    │   Analytics & Output      │
    │  - Zone Events            │
    │  - REST API               │
    │  - Web Dashboard          │
    └───────────────────────────┘
```

## Installation

### Requirements

- Python 3.9+
- CUDA 11.x (optional, for GPU)
- FFmpeg

### Install Dependencies

```bash
# CPU only
pip install -r requirements.txt

# With GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Configuration

Create a YAML configuration file:

```yaml
inputs:
  - path: rtsp://camera1/stream
    camera_id: cam1
  - path: data/videos/cam2.mp4
    camera_id: cam2

output:
  video_dir: outputs/videos
  tracks_path: outputs/tracks.jsonl
  show: true

modules:
  detector:
    name: yolov8
    model: yolov8n.pt
    conf: 0.25
  tracker:
    name: bytetrack
  reid:
    name: deep
    model: osnet_x0_25

runtime:
  max_frames: null
  skip_frames: 1
```

## Usage

### Command Line

```bash
# Run with config
python -m src.app --config configs/demo.yaml

# Run API server
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Python API

```python
import yaml
from src.pipeline import MultiCamPipeline

with open('configs/demo.yaml') as f:
    cfg = yaml.safe_load(f)

pipeline = MultiCamPipeline(cfg)
pipeline.run()
```

### Dashboard

Access the web dashboard at `http://localhost:8000`

## Evaluation

```bash
# Download datasets
python scripts/download_datasets.py --dataset pets2009

# Run benchmark
python -m src.evaluation.benchmark \
    --dataset data/datasets/pets2009 \
    --output outputs/benchmark
```

## Project Structure

```
multi_camera_tracking_system/
├── configs/              # Configuration files
├── docs/                 # Documentation
├── scripts/              # Utility scripts
├── src/
│   ├── api/              # FastAPI + Dashboard
│   ├── analytics/        # Zone events & aggregation
│   ├── association/      # Cross-camera matching
│   ├── detection/        # YOLOv8 / ONNX detectors
│   ├── evaluation/       # Metrics & benchmarks
│   ├── io/               # Stream handling
│   ├── reid/             # Re-identification
│   ├── tracking/         # ByteTrack / Kalman
│   ├── utils/            # Visualization
│   └── pipeline.py       # Main pipeline
├── tests/                # Unit tests
├── requirements.txt
└── README.md
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html
```

## Performance

| Component | GPU (RTX 3080) | CPU (i7-12700) |
|-----------|----------------|----------------|
| Detection | 50+ FPS | 5-10 FPS |
| Tracking | 500+ FPS | 200+ FPS |
| ReID | 100+ FPS | 10-20 FPS |
| **Total** | **30+ FPS** | **5-8 FPS** |

## Documentation

- [Workflow Guide](docs/WORKFLOW.md) - Complete usage instructions
- [Architecture](docs/architecture.md) - System design details
- [Changelog](CHANGELOG.md) - Version history

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Prithish Soni**

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [OSNet](https://github.com/KaiyangZhou/deep-person-reid)
- [MOT Challenge](https://motchallenge.net/)

## Citation

```bibtex
@software{multi_camera_tracking,
  author = {Soni, Prithish},
  title = {Multi-Camera People Tracking System},
  year = {2026},
  url = {https://github.com/SoniPrithish/multi_camera_tracking_system}
}
```
