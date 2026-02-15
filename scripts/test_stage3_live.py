#!/usr/bin/env python3
"""
Stage 3 Test: LIVE video with YOLOv8 Detection + ByteTrack Tracking overlays.
True WebSocket live video stream — each person gets a stable color-coded ID.

Usage:
    cd multi_camera_tracking_system
    python3 scripts/test_stage3_live.py

Then open http://<host>:8889 in your browser.
"""
import os, sys, time, json, threading, logging, yaml, hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("stage3_test")

import cv2
import numpy as np
import asyncio
from collections import defaultdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

from src.io.streams import MultiVideoReader
from src.detection.yolov8 import YOLOv8Detector
from src.tracking.byte_tracker import ByteTracker

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
stats = {"det_count": {}, "track_count": {}, "fps_per_cam": {}, "latency_per_cam": {}, "global_fps": 0.0}


def id_to_color(tid: int):
    """Deterministic bright color from a track ID."""
    h = int(hashlib.md5(str(tid).encode()).hexdigest()[:6], 16)
    # HSV: hue varies, high saturation + value
    hue = h % 180
    color_bgr = cv2.cvtColor(
        np.array([[[hue, 220, 230]]], dtype=np.uint8), cv2.COLOR_HSV2BGR
    )[0, 0]
    return tuple(int(c) for c in color_bgr)


def pipeline_loop():
    """Background thread: ingest → detect → track → draw overlay → store."""
    reader = MultiVideoReader(CAMERAS, ingest_cfg=INGEST_CFG)
    detector = YOLOv8Detector(
        model=MOD.get("detector_model", "yolov8n.pt"),
        device=str(MOD.get("detector_device", "0")),
        conf=float(MOD.get("detector_conf", 0.35)),
        iou=float(MOD.get("detector_iou", 0.45)),
        img_size=int(MOD.get("detector_img_size", 640)),
    )
    tracker = ByteTracker(
        max_age=int(MOD.get("tracker_max_age", 30)),
        min_hits=int(MOD.get("tracker_min_hits", 3)),
        iou_thresh=float(MOD.get("tracker_iou_thresh", 0.2)),
        high_thresh=float(MOD.get("tracker_high_thresh", 0.5)),
        low_thresh=float(MOD.get("tracker_low_thresh", 0.1)),
    )

    # Tracklet history for drawing trails
    trail_history = defaultdict(lambda: defaultdict(list))  # cam_id -> tid -> [(cx, cy)]
    MAX_TRAIL = 30

    frame_count = 0
    t_start = time.monotonic()
    cam_counters = {}

    logger.info("Pipeline loop started — %d cameras", len(CAMERAS))

    while True:
        batch = reader.read()
        if batch is None:
            break
        if not batch:
            continue

        for cam_id, frame in batch.items():
            t0 = time.monotonic()

            # ── DETECT ──
            dets = detector.detect(frame)

            # ── TRACK ──
            tracks = tracker.update(cam_id, dets)

            # ── DRAW overlays ──
            vis = frame.copy()

            for t in tracks:
                tid = t["tid"]
                x, y, w, h = map(int, t["bbox"])
                score = t["score"]
                color = id_to_color(tid)

                # Bounding box
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

                # Label with ID
                label = f"ID:{tid} ({score:.2f})"
                (tw, th_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(vis, (x, max(0, y - th_text - 8)), (x + tw + 4, y), color, -1)
                cv2.putText(vis, label, (x + 2, max(th_text + 2, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                # Trail (tracklet)
                cx, cy = x + w // 2, y + h
                trail_history[cam_id][tid].append((cx, cy))
                if len(trail_history[cam_id][tid]) > MAX_TRAIL:
                    trail_history[cam_id][tid] = trail_history[cam_id][tid][-MAX_TRAIL:]
                pts = trail_history[cam_id][tid]
                for i in range(1, len(pts)):
                    thickness = max(1, int(2 * i / MAX_TRAIL) + 1)
                    cv2.line(vis, pts[i - 1], pts[i], color, thickness)

            # Clean up trails for tracks no longer active
            active_tids = {t["tid"] for t in tracks}
            dead_tids = [tid for tid in trail_history.get(cam_id, {}) if tid not in active_tids]
            for tid in dead_tids:
                del trail_history[cam_id][tid]

            latency_ms = (time.monotonic() - t0) * 1000

            # Per-camera FPS
            cam_counters.setdefault(cam_id, {"n": 0, "t": time.monotonic()})
            cam_counters[cam_id]["n"] += 1
            cam_elapsed = time.monotonic() - cam_counters[cam_id]["t"]
            cam_fps = cam_counters[cam_id]["n"] / max(0.01, cam_elapsed)

            # HUD
            hud = f"{cam_id} | {len(dets)}det {len(tracks)}trk | {cam_fps:.1f}FPS | {latency_ms:.0f}ms"
            cv2.putText(vis, hud, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            with frame_lock:
                latest_frames[cam_id] = vis
                stats["det_count"][cam_id] = len(dets)
                stats["track_count"][cam_id] = len(tracks)
                stats["fps_per_cam"][cam_id] = round(cam_fps, 1)
                stats["latency_per_cam"][cam_id] = round(latency_ms, 1)

            frame_count += 1
            elapsed = time.monotonic() - t_start
            stats["global_fps"] = round(frame_count / max(0.01, elapsed), 1)


@app.on_event("startup")
async def startup():
    t = threading.Thread(target=pipeline_loop, daemon=True)
    t.start()
    logger.info("Pipeline background thread started.")


def build_dashboard_html():
    cam_ids = [c["camera_id"] for c in CAMERAS]
    n = len(cam_ids)
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
<title>Stage 3: ByteTrack Live Tracking</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 16px; }}
  h1 {{ text-align: center; color: #d2a8ff; margin-bottom: 6px; }}
  #stats {{ text-align: center; margin: 8px; padding: 10px; background: #161b22;
            border-radius: 8px; font-size: 15px; color: #7ee787; font-family: monospace; }}
  .grid {{ display: grid; grid-template-columns: repeat({cols}, 1fr); gap: 12px; }}
  .cam-box {{ background: #161b22; border-radius: 10px; overflow: hidden;
              border: 1px solid #30363d; }}
  .cam-box h3 {{ text-align: center; margin: 8px 0 4px 0; color: #79c0ff; font-size: 14px; }}
  canvas {{ width: 100%; display: block; }}
</style>
</head>
<body>
<h1>🏃 Stage 3: ByteTrack Multi-Object Tracking</h1>
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


DASHBOARD_HTML = None


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
            await asyncio.sleep(0.05)
    except (WebSocketDisconnect, Exception):
        pass


@app.websocket("/ws/stats")
async def ws_stats(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with frame_lock:
                parts = []
                for cid in sorted(stats["fps_per_cam"].keys()):
                    d = stats["det_count"].get(cid, 0)
                    t = stats["track_count"].get(cid, 0)
                    f = stats["fps_per_cam"].get(cid, 0)
                    l = stats["latency_per_cam"].get(cid, 0)
                    parts.append(f"{cid}:{d}det/{t}trk {f}FPS {l}ms")
                msg = " │ ".join(parts) if parts else "Warming up..."
                msg = f"Global {stats['global_fps']}FPS  ━━  {msg}"
            await websocket.send_text(msg)
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, Exception):
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8889, log_level="info")
