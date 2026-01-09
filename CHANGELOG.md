# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-08

### Added

#### Core Features
- **Multi-camera video ingestion** with support for RTSP, HTTP, MP4, and webcam sources
- **YOLOv8 person detection** with ONNX export for CPU acceleration
- **ByteTrack multi-object tracker** with Kalman filter motion prediction
- **Deep ReID module** using OSNet with EMA prototype updates
- **Cross-camera association** with camera graph topology and zone gating
- **Global identity registry** for consistent IDs across cameras

#### Stream Handling
- Bounded frame queues with drop-oldest policy
- RTSP auto-reconnect with configurable delays
- Per-stream health monitoring (FPS, drop rate, latency)
- Adaptive resolution scaling based on latency budget

#### Analytics
- Zone-based entry/exit detection
- Dwell time calculation
- Line crossing counting with direction detection
- Real-time event aggregation with windowed statistics

#### API & Dashboard
- FastAPI REST API with OpenAPI documentation
- WebSocket endpoint for real-time updates
- Jinja2-based dashboard with dark industrial theme
- Multi-camera grid view with live statistics
- Event timeline and analytics charts

#### Evaluation
- Dataset downloaders for PETS, EPFL, WILDTRACK, MOT17
- Data loaders for multiple dataset formats
- MOTA, IDF1, HOTA metric calculators
- Multi-camera metrics with handoff accuracy
- Benchmark runner with latency profiling

#### Testing & CI
- Comprehensive unit tests for all modules
- Pytest configuration with fixtures
- GitHub Actions CI pipeline

### Technical Details

#### Detection Module
- `YOLOv8Detector`: Full Ultralytics integration with GPU/CPU support
- `ONNXDetector`: ONNX Runtime inference for CPU acceleration
- `DummyDetector`: Testing detector with synthetic trajectories

#### Tracking Module
- `ByteTracker`: Two-stage association (high/low confidence)
- `KalmanFilter`: Constant velocity motion model
- `CentroidTracker`: Simple IoU-based tracking (baseline)

#### ReID Module
- `DeepReID`: OSNet/torchreid integration with ONNX fallback
- `CosineReID`: Color histogram baseline with cosine similarity
- EMA embedding smoothing for stable appearance features

#### Association Module
- `CameraGraph`: Camera topology with transfer time windows
- `ZoneGate`: Entry/exit zone constraints for matching
- `CrossCameraMatcher`: Hungarian algorithm for optimal assignment
- `GlobalRegistry`: Identity management with merge support

### Configuration
- YAML-based configuration for all components
- Environment variable support
- Modular component selection

### Dependencies
- Core: OpenCV, NumPy, PyYAML, scikit-learn
- Deep Learning: PyTorch, Ultralytics, torchreid, ONNX Runtime
- Tracking: lap, filterpy
- API: FastAPI, uvicorn, Jinja2, websockets
- Evaluation: motmetrics, pandas, matplotlib

---

## [0.1.0] - 2025-11-07

### Added
- Initial project scaffold
- Basic pipeline structure
- Dummy detector and centroid tracker
- Simple color histogram ReID
- Video I/O utilities

---

## Development Timeline

| Date | Milestone |
|------|-----------|
| Nov 7, 2025 | Project initialization, requirements setup |
| Nov 10, 2025 | YOLOv8 detector implementation |
| Nov 20, 2025 | ByteTrack and stream handling |
| Dec 5, 2025 | Deep ReID module |
| Dec 15, 2025 | Cross-camera association |
| Dec 28, 2025 | Analytics module |
| Jan 5, 2026 | FastAPI and dashboard |
| Jan 8, 2026 | Evaluation pipeline, tests, documentation |

---

## Future Plans

### v1.1.0 (Planned)
- [ ] RT-DETR detector support
- [ ] DeepSORT tracker option
- [ ] ArcFace ReID model
- [ ] Ground-plane calibration
- [ ] 3D position estimation

### v1.2.0 (Planned)
- [ ] Trajectory prediction
- [ ] Behavior analysis
- [ ] Anomaly detection
- [ ] Multi-GPU support
- [ ] Docker deployment

### v2.0.0 (Planned)
- [ ] Transformer-based tracking
- [ ] Self-supervised ReID
- [ ] Edge deployment (Jetson)
- [ ] Cloud-native architecture

