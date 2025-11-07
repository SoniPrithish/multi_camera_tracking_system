
# MULTI_CAMERA_TRACKING — Starter Kit

This is a minimal, **batteries-included scaffold** for a multi-camera person/object tracking project.
It's structured so you can swap in your preferred detector (YOLO, RT-DETR, etc.), single-camera tracker (BYTE, DeepSORT),
and cross-camera re-identification (ReID) modules without changing the app shell.

> TL;DR: Edit `configs/demo.yaml`, then run: `python -m src.app --config configs/demo.yaml`

## Quick Start
1. Create and activate a virtual environment (conda/venv).
2. `pip install -r requirements.txt`
3. Put a couple of short test clips under `data/samples/` (or use camera RTSP URLs in the config).
4. Run: `python -m src.app --config configs/demo.yaml`

This starter uses **dummy algorithms** (fast, CPU-only) so the pipeline is runnable everywhere.
Replace the stubs in `src/detection`, `src/tracking`, and `src/reid` with your real models when ready.
