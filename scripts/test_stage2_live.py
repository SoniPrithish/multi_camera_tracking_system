#!/usr/bin/env python3
"""
Stage 2 Test: LIVE video stream with YOLOv8 detection overlays.
Streams MJPEG via WebSocket — true live video, NOT images.

Usage:
    cd multi_camera_tracking_system
    python3 scripts/test_stage2_live.py

Then open http://<host>:8888 in your browser.
"""
import os, sys, time, json, threading, logging, yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("stage2_test")

import cv2
import numpy as np
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from src.io.streams import MultiVideoReader
from src.detection.yolov8 import YOLOv8Detector

# ── Load config ──
with open("configs/demo.yaml") as f:
    cfg = yaml.safe_load(f)

CAMERAS = cfg["inputs"]
INGEST_CFG = cfg.get("ingest", {})
INGEST_CFG["resize_width"] = cfg.get("runtime", {}).get("resize_width", 960)
MOD = cfg.get("modules", {})

app = FastAPI()

# ── Shared state ──
latest_frames = {}
frame_lock = threading.Lock()
det_count = {}
fps_per_cam = {}
latency_per_cam = {}
global_fps = 0.0


def pipeline_loop():
    """Background thread: ingest → detect → draw overlay → store latest frames."""
    global global_fps

    reader = MultiVideoReader(CAMERAS, ingest_cfg=INGEST_CFG)
    detector = YOLOv8Detector(
        model=MOD.get("detector_model", "yolov8n.pt"),
        device=str(MOD.get("detector_device", "0")),
        conf=float(MOD.get("detector_conf", 0.35)),
        iou=float(MOD.get("detector_iou", 0.45)),
        img_size=int(MOD.get("detector_img_size", 640)),
    )

    frame_count = 0
    t_start = time.monotonic()
    cam_counters = {}

    logger.info("Pipeline loop started — %d cameras", len(CAMERAS))

    while True:
        batch = reader.read()
        if batch is None:
            logger.warning("All cameras dead — stopping.")
            break
        if not batch:
            continue

        for cam_id, frame in batch.items():
            t0 = time.monotonic()

            # ── DETECT ──
            dets = detector.detect(frame)

            # ── DRAW overlays ──
            vis = frame.copy()
            for d in dets:
                x, y, w, h = map(int, d["bbox"])
                score = d["score"]
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = f"person {score:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(vis, (x, max(0, y - th - 6)), (x + tw + 4, y), (0, 255, 0), -1)
                cv2.putText(vis, label, (x + 2, max(th + 2, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            latency_ms = (time.monotonic() - t0) * 1000

            # Per-camera FPS
            cam_counters.setdefault(cam_id, {"n": 0, "t": time.monotonic()})
            cam_counters[cam_id]["n"] += 1
            cam_elapsed = time.monotonic() - cam_counters[cam_id]["t"]
            cam_fps = cam_counters[cam_id]["n"] / max(0.01, cam_elapsed)

            # HUD
            hud = f"{cam_id} | {len(dets)} ppl | {cam_fps:.1f} FPS | {latency_ms:.0f}ms"
            cv2.putText(vis, hud, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

            with frame_lock:
                latest_frames[cam_id] = vis
                det_count[cam_id] = len(dets)
                fps_per_cam[cam_id] = round(cam_fps, 1)
                latency_per_cam[cam_id] = round(latency_ms, 1)

            frame_count += 1
            elapsed = time.monotonic() - t_start
            global_fps = round(frame_count / max(0.01, elapsed), 1)


@app.on_event("startup")
async def startup():
    t = threading.Thread(target=pipeline_loop, daemon=True)
    t.start()
    logger.info("Pipeline background thread started.")


def build_dashboard_html():
    """Generate dashboard HTML dynamically for N cameras."""
    cam_ids = [c["camera_id"] for c in CAMERAS]
    n = len(cam_ids)

    # Choose grid cols: up to 3 cols, then wrap
    cols = min(n, 3)

    cam_divs = "\n".join(
        f'  <div class="cam-box"><h3>{cid.upper()}</h3>'
        f'<canvas id="canvas-{cid}"></canvas></div>'
        for cid in cam_ids
    )

    js_cams = json.dumps(cam_ids)

    return f"""<!DOCTYPE html>
<html>
<head>
<title>Stage 2: YOLOv8 Live Detection</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 16px; }}
  h1 {{ text-align: center; color: #58a6ff; margin-bottom: 6px; }}
  #stats {{ text-align: center; margin: 8px; padding: 10px; background: #161b22;
            border-radius: 8px; font-size: 16px; color: #7ee787; font-family: monospace; }}
  .grid {{ display: grid; grid-template-columns: repeat({cols}, 1fr); gap: 12px; }}
  .cam-box {{ background: #161b22; border-radius: 10px; overflow: hidden;
              border: 1px solid #30363d; }}
  .cam-box h3 {{ text-align: center; margin: 8px 0 4px 0; color: #79c0ff; font-size: 14px; }}
  canvas {{ width: 100%; display: block; }}
</style>
</head>
<body>
<h1>🎯 Stage 2: YOLOv8 Live Person Detection</h1>
<div id="stats">Connecting to pipeline...</div>
<div class="grid">
{cam_divs}
</div>
<script>
const cams = {js_cams};
const canvases = {{}};
const ctxs = {{}};
cams.forEach(c => {{
    canvases[c] = document.getElementById('canvas-' + c);
    ctxs[c] = canvases[c].getContext('2d');
}});

function connectCam(camId) {{
    const ws = new WebSocket('ws://' + location.host + '/ws/' + camId);
    ws.binaryType = 'arraybuffer';
    ws.onmessage = (evt) => {{
        const blob = new Blob([evt.data], {{type: 'image/jpeg'}});
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {{
            const canvas = canvases[camId];
            canvas.width = img.width;
            canvas.height = img.height;
            ctxs[camId].drawImage(img, 0, 0);
            URL.revokeObjectURL(url);
        }};
        img.src = url;
    }};
    ws.onclose = () => {{ setTimeout(() => connectCam(camId), 1000); }};
    ws.onerror = () => {{ ws.close(); }};
}}

function connectStats() {{
    const ws = new WebSocket('ws://' + location.host + '/ws/stats');
    ws.onmessage = (evt) => {{
        document.getElementById('stats').innerText = evt.data;
    }};
    ws.onclose = () => {{ setTimeout(connectStats, 1000); }};
}}

cams.forEach(connectCam);
connectStats();
</script>
</body>
</html>"""


DASHBOARD_HTML = None  # built on first request


@app.get("/", response_class=HTMLResponse)
async def index():
    global DASHBOARD_HTML
    if DASHBOARD_HTML is None:
        DASHBOARD_HTML = build_dashboard_html()
    return DASHBOARD_HTML


@app.websocket("/ws/{cam_id}")
async def ws_cam(websocket: WebSocket, cam_id: str):
    await websocket.accept()
    try:
        while True:
            with frame_lock:
                frame = latest_frames.get(cam_id)
            if frame is not None:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
                await websocket.send_bytes(buf.tobytes())
            await asyncio.sleep(0.05)  # ~20 FPS push rate
    except (WebSocketDisconnect, Exception):
        pass


@app.websocket("/ws/stats")
async def ws_stats(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with frame_lock:
                parts = []
                for cid in sorted(fps_per_cam.keys()):
                    parts.append(f"{cid}: {det_count.get(cid,0)} ppl, {fps_per_cam[cid]} FPS, {latency_per_cam.get(cid,0)}ms")
                msg = " | ".join(parts) if parts else "Warming up..."
                msg = f"Global: {global_fps} FPS  ━  {msg}"
            await websocket.send_text(msg)
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, Exception):
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888, log_level="info")
