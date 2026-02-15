#!/usr/bin/env python3
"""
Production Multi-Camera Tracking Server — Stage 7
====================================================

Single entry-point for the full pipeline:
  Ingest → YOLOv8 Detection → ByteTrack → OSNet ReID → Cross-Camera Association → Events

Serves a polished dashboard via FastAPI with:
  • Live multi-camera WebSocket video streams (JPEG, ~30 FPS)
  • Global ID overlay with colored bounding boxes + trails
  • Real-time stats (FPS, latency, gallery, cross-cam matches)
  • Event timeline (entry/exit, dwell, line-crossing)
  • Per-camera detection/track counts
  • REST API endpoints for programmatic access

Usage:
    python -m src.server --config configs/demo.yaml
    # or
    python src/server.py --config configs/demo.yaml

Dashboard: http://<host>:8000
"""

from __future__ import annotations

import os
import sys
import time
import json
import hashlib
import logging
import argparse
import threading
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Dict, List, Any

import cv2
import numpy as np
import yaml
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# ── Path setup ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from src.io.streams import MultiVideoReader
from src.detection.yolov8 import YOLOv8Detector
from src.tracking.byte_tracker import ByteTracker
from src.reid.osnet import OSNetExtractor
from src.association.cross_camera import CrossCameraAssociator
from src.events.event_detector import EventDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("mct-server")


# ══════════════════════════════════════════════════════════════════════════════
# Shared State
# ══════════════════════════════════════════════════════════════════════════════

latest_frames: Dict[str, np.ndarray] = {}
frame_lock = threading.Lock()

stats: Dict[str, Any] = {
    "det_count": {},
    "track_count": {},
    "fps_per_cam": {},
    "global_fps": 0.0,
    "det_ms": 0.0,
    "track_ms": 0.0,
    "reid_ms": 0.0,
    "assoc_ms": 0.0,
    "events_ms": 0.0,
    "draw_ms": 0.0,
    "gallery_size": 0,
    "active_ids": 0,
    "inactive_ids": 0,
    "cross_cam_ids": 0,
    "cross_cam_matches": 0,
    "event_counts": {"entry": 0, "exit": 0, "dwell": 0, "line_crossing": 0, "total": 0},
    "uptime_sec": 0,
}
stats_lock = threading.Lock()

# Global references filled by pipeline_loop
event_detector_ref: EventDetector = None
pipeline_running = threading.Event()


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def gid_to_color(gid: int):
    h = int(hashlib.md5(str(gid).encode()).hexdigest()[:6], 16)
    hue = h % 180
    c = cv2.cvtColor(np.array([[[hue, 230, 240]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(x) for x in c)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline Loop (runs in a background thread)
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_loop(cfg: dict):
    global event_detector_ref

    cameras = cfg["inputs"]
    ingest_cfg = cfg.get("ingest", {})
    ingest_cfg["resize_width"] = cfg.get("runtime", {}).get("resize_width", 960)
    ingest_cfg["target_fps"] = 30
    mod = cfg.get("modules", {})
    assoc_cfg = cfg.get("association", {})
    events_cfg = cfg.get("events", {})
    server_cfg = cfg.get("server", {})

    REID_EVERY_N = 8
    MAX_REID_CROPS = 48
    JPEG_QUALITY = int(server_cfg.get("ws_jpeg_quality", 65))
    MAX_TRAIL = 30

    # ── Build components ──
    reader = MultiVideoReader(cameras, ingest_cfg=ingest_cfg)

    detector = YOLOv8Detector(
        model=mod.get("detector_model", "yolov8n.pt"),
        device=str(mod.get("detector_device", "0")),
        conf=float(mod.get("detector_conf", 0.35)),
        iou=float(mod.get("detector_iou", 0.45)),
        img_size=int(mod.get("detector_img_size", 640)),
        half=True,
    )
    tracker = ByteTracker(
        max_age=int(mod.get("tracker_max_age", 30)),
        min_hits=int(mod.get("tracker_min_hits", 3)),
        iou_thresh=float(mod.get("tracker_iou_thresh", 0.2)),
    )
    reid_extractor = OSNetExtractor(
        model_name=mod.get("reid_model", "osnet_x0_25"),
        device=str(mod.get("reid_device", mod.get("detector_device", "0"))),
        half=True,
    )
    cross_cam = CrossCameraAssociator(assoc_cfg)

    evt_det = EventDetector(events_cfg)
    event_detector_ref = evt_det

    trail_history = defaultdict(lambda: defaultdict(list))
    cached_gids: Dict[str, Dict[int, int]] = {}

    loop_count = 0
    total_frames = 0
    t_global_start = time.monotonic()
    cam_counters: Dict[str, dict] = {}

    pipeline_running.set()
    logger.info(
        "Pipeline started — %d cameras, FP16=True, JPEG quality=%d",
        len(cameras), JPEG_QUALITY,
    )

    while True:
        batch = reader.read()
        if batch is None:
            logger.info("All video sources exhausted. Pipeline stopping.")
            break
        if not batch:
            continue

        loop_count += 1
        cam_ids = list(batch.keys())
        frames = [batch[cid] for cid in cam_ids]

        # ━━━ BATCH DETECTION ━━━
        t0 = time.monotonic()
        batch_dets = detector.detect_batch(frames)
        det_ms = (time.monotonic() - t0) * 1000

        # ━━━ TRACKING ━━━
        t0 = time.monotonic()
        all_tracks: Dict[str, List[Dict[str, Any]]] = {}
        for i, cam_id in enumerate(cam_ids):
            all_tracks[cam_id] = tracker.update(cam_id, batch_dets[i])
        track_ms = (time.monotonic() - t0) * 1000

        # ━━━ ReID EMBEDDINGS ━━━
        run_reid = (loop_count % REID_EVERY_N == 1) or loop_count <= 2
        t0 = time.monotonic()
        all_embs: Dict[str, Dict[int, np.ndarray]] = defaultdict(dict)
        if run_reid:
            all_crops = []
            crop_map = []
            for cam_id in cam_ids:
                for t in all_tracks[cam_id]:
                    if len(all_crops) >= MAX_REID_CROPS:
                        break
                    x, y, w, h = map(int, t["bbox"])
                    fh, fw = batch[cam_id].shape[:2]
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(fw, x + w), min(fh, y + h)
                    crop = batch[cam_id][y1:y2, x1:x2]
                    if crop.size > 0 and crop.shape[0] >= 10 and crop.shape[1] >= 10:
                        all_crops.append(crop)
                        crop_map.append((cam_id, t["tid"]))
            if all_crops:
                raw_embs = reid_extractor.extract(all_crops)
                for idx, (cam_id, tid) in enumerate(crop_map):
                    all_embs[cam_id][tid] = raw_embs[idx]
        reid_ms = (time.monotonic() - t0) * 1000

        # ━━━ CROSS-CAMERA ASSOCIATION ━━━
        t0 = time.monotonic()
        if run_reid and all_embs:
            cached_gids = cross_cam.associate(all_tracks, all_embs)
        else:
            cached_gids = cross_cam.associate(all_tracks, defaultdict(dict))
        assoc_ms = (time.monotonic() - t0) * 1000

        # ━━━ EVENTS ━━━
        t0 = time.monotonic()
        for i, cam_id in enumerate(cam_ids):
            fh, fw = frames[i].shape[:2]
            evt_det.process(
                cam_id=cam_id,
                tracks=all_tracks[cam_id],
                gids=cached_gids.get(cam_id, {}),
                frame_shape=(fh, fw),
            )
        events_ms = (time.monotonic() - t0) * 1000

        # ━━━ DRAW ━━━
        t0 = time.monotonic()
        for i, cam_id in enumerate(cam_ids):
            frame = frames[i]
            tracks = all_tracks[cam_id]
            gids = cached_gids.get(cam_id, {})
            fh, fw = frame.shape[:2]

            vis = frame.copy()

            # Auto-generated crossing line at 60%
            line_y = int(fh * 0.6)
            cv2.line(vis, (0, line_y), (fw, line_y), (0, 180, 180), 1, cv2.LINE_AA)

            for t in tracks:
                tid = t["tid"]
                x, y, w, h = map(int, t["bbox"])
                gid = gids.get(tid)
                if gid is not None:
                    color = gid_to_color(gid)
                    label = f"G{gid}"
                else:
                    color = (100, 100, 100)
                    label = f"T{tid}"

                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(vis, (x, max(0, y - th_t - 6)), (x + tw + 4, y), color, -1)
                cv2.putText(vis, label, (x + 2, max(th_t + 2, y - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # Trail
                trail_key = gid if gid is not None else f"T{tid}"
                cx, cy = x + w // 2, y + h
                trail_history[cam_id][trail_key].append((cx, cy))
                if len(trail_history[cam_id][trail_key]) > MAX_TRAIL:
                    trail_history[cam_id][trail_key] = trail_history[cam_id][trail_key][-MAX_TRAIL:]
                pts = trail_history[cam_id][trail_key]
                for j in range(1, len(pts)):
                    thickness = max(1, int(2 * j / MAX_TRAIL) + 1)
                    cv2.line(vis, pts[j - 1], pts[j], color, thickness)

            # Cleanup stale trails
            active_keys = set()
            for t in tracks:
                g = gids.get(t["tid"])
                active_keys.add(g if g is not None else f"T{t['tid']}")
            for tk in list(trail_history.get(cam_id, {}).keys()):
                if tk not in active_keys:
                    del trail_history[cam_id][tk]

            # Per-camera HUD
            cam_counters.setdefault(cam_id, {"n": 0, "t": time.monotonic()})
            cam_counters[cam_id]["n"] += 1
            elapsed_cam = time.monotonic() - cam_counters[cam_id]["t"]
            cam_fps = cam_counters[cam_id]["n"] / max(0.01, elapsed_cam)
            reid_tag = "R" if run_reid else "C"
            hud = f"{cam_id} | {len(batch_dets[i])}d {len(tracks)}t | {cam_fps:.1f}FPS [{reid_tag}]"
            cv2.putText(vis, hud, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

            with frame_lock:
                latest_frames[cam_id] = vis

        draw_ms = (time.monotonic() - t0) * 1000

        # ━━━ UPDATE STATS ━━━
        total_frames += len(cam_ids)
        elapsed_total = time.monotonic() - t_global_start
        gfps = total_frames / max(0.01, elapsed_total)

        assoc_stats = cross_cam.get_stats()
        evt_counts = evt_det.get_counts()

        with stats_lock:
            for i, cam_id in enumerate(cam_ids):
                stats["det_count"][cam_id] = len(batch_dets[i])
                stats["track_count"][cam_id] = len(all_tracks[cam_id])
                cc = cam_counters[cam_id]
                stats["fps_per_cam"][cam_id] = round(
                    cc["n"] / max(0.01, time.monotonic() - cc["t"]), 1
                )
            stats["global_fps"] = round(gfps, 1)
            stats["det_ms"] = round(det_ms, 1)
            stats["track_ms"] = round(track_ms, 1)
            stats["reid_ms"] = round(reid_ms, 1)
            stats["assoc_ms"] = round(assoc_ms, 1)
            stats["events_ms"] = round(events_ms, 1)
            stats["draw_ms"] = round(draw_ms, 1)
            stats["gallery_size"] = assoc_stats["gallery_size"]
            stats["active_ids"] = assoc_stats["active"]
            stats["inactive_ids"] = assoc_stats["inactive"]
            stats["cross_cam_ids"] = assoc_stats["cross_camera_ids"]
            stats["cross_cam_matches"] = assoc_stats["cross_cam_matches"]
            stats["event_counts"] = evt_counts
            stats["uptime_sec"] = round(elapsed_total, 0)

    pipeline_running.clear()
    logger.info("Pipeline loop finished.")


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI Application
# ══════════════════════════════════════════════════════════════════════════════

CFG: dict = {}

@asynccontextmanager
async def lifespan(app_: FastAPI):
    t = threading.Thread(target=pipeline_loop, args=(CFG,), daemon=True)
    t.start()
    logger.info("Pipeline thread launched.")
    yield
    logger.info("Server shutting down.")

app = FastAPI(title="Multi-Camera Tracking System", lifespan=lifespan)


# ── Dashboard HTML ───────────────────────────────────────────────────────────

def build_dashboard_html(cfg: dict) -> str:
    cam_ids = [c["camera_id"] for c in cfg["inputs"]]
    cols = min(len(cam_ids), 3)
    cam_divs = "\n".join(
        f'  <div class="cam-box"><div class="cam-header">{cid.upper()}</div>'
        f'<canvas id="canvas-{cid}"></canvas></div>'
        for cid in cam_ids
    )
    js_cams = json.dumps(cam_ids)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multi-Camera People Tracking — Dashboard</title>
<style>
  :root {{
    --bg-primary: #0d1117;
    --bg-card: #161b22;
    --bg-hover: #1c2129;
    --border: #30363d;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --accent-orange: #ffa657;
    --accent-blue: #79c0ff;
    --accent-green: #3fb950;
    --accent-red: #f85149;
    --accent-pink: #f778ba;
    --accent-amber: #d29922;
    --accent-cyan: #a5d6ff;
    --font-mono: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    --font-sans: 'Inter', 'Segoe UI', system-ui, sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: var(--font-sans);
    min-height: 100vh;
  }}

  /* ── Header ── */
  .header {{
    text-align: center;
    padding: 12px 16px 8px;
    border-bottom: 1px solid var(--border);
  }}
  .header h1 {{
    font-size: 20px;
    font-weight: 700;
    color: var(--accent-orange);
    letter-spacing: -0.3px;
  }}
  .header .subtitle {{
    font-size: 11px;
    color: var(--text-secondary);
    margin-top: 2px;
  }}

  /* ── Stats bar ── */
  #stats-bar {{
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px 20px;
    padding: 10px 16px;
    background: var(--bg-card);
    border-bottom: 1px solid var(--border);
    font-family: var(--font-mono);
    font-size: 12px;
  }}
  .stat {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
  }}
  .stat-label {{ color: var(--text-secondary); }}
  .stat-val {{ font-weight: 600; }}
  .stat-val.orange {{ color: var(--accent-orange); }}
  .stat-val.green {{ color: var(--accent-green); }}
  .stat-val.pink {{ color: var(--accent-pink); }}
  .stat-val.cyan {{ color: var(--accent-cyan); }}
  .stat-val.blue {{ color: var(--accent-blue); }}
  .stat-val.red {{ color: var(--accent-red); }}

  /* ── Camera grid ── */
  .cam-grid {{
    display: grid;
    grid-template-columns: repeat({cols}, 1fr);
    gap: 6px;
    padding: 6px;
  }}
  .cam-box {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    transition: border-color 0.2s;
  }}
  .cam-box:hover {{ border-color: var(--accent-blue); }}
  .cam-header {{
    text-align: center;
    font-size: 12px;
    font-weight: 600;
    color: var(--accent-blue);
    padding: 4px 0;
    background: var(--bg-primary);
    border-bottom: 1px solid var(--border);
  }}
  canvas {{ width: 100%; display: block; }}

  /* ── Bottom panels ── */
  .bottom-panels {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 6px;
    padding: 6px;
  }}
  .panel {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
  }}
  .panel h3 {{
    font-size: 13px;
    color: var(--accent-orange);
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 6px;
  }}

  /* ── Event timeline ── */
  #event-list {{
    max-height: 200px;
    overflow-y: auto;
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 1.6;
  }}
  #event-list::-webkit-scrollbar {{ width: 6px; }}
  #event-list::-webkit-scrollbar-track {{ background: var(--bg-primary); border-radius: 3px; }}
  #event-list::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
  .evt-entry {{ color: var(--accent-green); }}
  .evt-exit {{ color: var(--accent-red); }}
  .evt-dwell {{ color: var(--accent-amber); }}
  .evt-line_crossing {{ color: var(--accent-cyan); }}

  /* ── Per-cam stats table ── */
  #cam-stats-table {{
    width: 100%;
    font-family: var(--font-mono);
    font-size: 11px;
    border-collapse: collapse;
  }}
  #cam-stats-table th {{
    text-align: left;
    color: var(--text-secondary);
    padding: 3px 8px;
    border-bottom: 1px solid var(--border);
    font-weight: 500;
  }}
  #cam-stats-table td {{
    padding: 3px 8px;
    border-bottom: 1px solid rgba(48,54,61,0.4);
  }}
  #cam-stats-table td.num {{ text-align: right; color: var(--accent-orange); }}

  /* ── Uptime + status ── */
  .status-bar {{
    text-align: center;
    font-size: 11px;
    color: var(--text-secondary);
    padding: 6px;
    border-top: 1px solid var(--border);
    font-family: var(--font-mono);
  }}
  .status-dot {{
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent-green);
    margin-right: 4px;
    animation: pulse 2s infinite;
  }}
  @keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.4; }}
  }}

  @media (max-width: 900px) {{
    .cam-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .bottom-panels {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div class="header">
  <h1>Multi-Camera People Tracking</h1>
  <div class="subtitle">YOLOv8 · ByteTrack · OSNet ReID · Cross-Camera Association · Events | RTX 4060 Ti FP16</div>
</div>

<div id="stats-bar">
  <span class="stat"><span class="stat-label">Gallery:</span> <span class="stat-val orange" id="s-gallery">-</span></span>
  <span class="stat"><span class="stat-label">Active:</span> <span class="stat-val green" id="s-active">-</span></span>
  <span class="stat"><span class="stat-label">Cross-Cam:</span> <span class="stat-val pink" id="s-xcam">-</span></span>
  <span class="stat"><span class="stat-label">Handoffs:</span> <span class="stat-val pink" id="s-handoffs">-</span></span>
  <span class="stat"><span class="stat-label">│</span></span>
  <span class="stat"><span class="stat-label">Entry:</span> <span class="stat-val cyan" id="s-entry">0</span></span>
  <span class="stat"><span class="stat-label">Exit:</span> <span class="stat-val red" id="s-exit">0</span></span>
  <span class="stat"><span class="stat-label">Dwell:</span> <span class="stat-val orange" id="s-dwell">0</span></span>
  <span class="stat"><span class="stat-label">LineCross:</span> <span class="stat-val cyan" id="s-lcross">0</span></span>
  <span class="stat"><span class="stat-label">│</span></span>
  <span class="stat"><span class="stat-label">FPS:</span> <span class="stat-val green" id="s-fps">-</span></span>
  <span class="stat"><span class="stat-label">Det:</span> <span class="stat-val blue" id="s-det">-</span></span>
  <span class="stat"><span class="stat-label">Trk:</span> <span class="stat-val blue" id="s-trk">-</span></span>
  <span class="stat"><span class="stat-label">ReID:</span> <span class="stat-val blue" id="s-reid">-</span></span>
  <span class="stat"><span class="stat-label">Assoc:</span> <span class="stat-val blue" id="s-assoc">-</span></span>
  <span class="stat"><span class="stat-label">Events:</span> <span class="stat-val blue" id="s-events">-</span></span>
</div>

<div class="cam-grid">
{cam_divs}
</div>

<div class="bottom-panels">
  <div class="panel">
    <h3>📝 Event Timeline</h3>
    <div id="event-list">Waiting for events...</div>
  </div>
  <div class="panel">
    <h3>📊 Per-Camera Stats</h3>
    <table id="cam-stats-table">
      <thead><tr><th>Camera</th><th>Dets</th><th>Tracks</th><th>FPS</th></tr></thead>
      <tbody id="cam-stats-body"><tr><td colspan="4" style="color:var(--text-secondary)">Loading...</td></tr></tbody>
    </table>
  </div>
</div>

<div class="status-bar">
  <span class="status-dot"></span>
  <span id="s-status">Pipeline starting...</span>
  <span style="margin-left:16px">Uptime: <span id="s-uptime">0s</span></span>
</div>

<script>
const cams = {js_cams};
const canvases = {{}};
const ctxs = {{}};

// Init canvases
cams.forEach(c => {{
  canvases[c] = document.getElementById('canvas-' + c);
  ctxs[c] = canvases[c].getContext('2d');
}});

// ── Camera streams ──
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
  ws.onclose = () => {{ setTimeout(() => connectCam(camId), 1500); }};
  ws.onerror = () => {{ ws.close(); }};
}}

// ── Stats + Events ──
const maxEvents = 100;
let eventLines = [];

function fmtUptime(sec) {{
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return h + 'h ' + m + 'm ' + s + 's';
  if (m > 0) return m + 'm ' + s + 's';
  return s + 's';
}}

function connectStats() {{
  const ws = new WebSocket('ws://' + location.host + '/ws/dashboard-stats');
  ws.onmessage = (evt) => {{
    try {{
      const d = JSON.parse(evt.data);

      // Stats bar
      document.getElementById('s-gallery').textContent = d.gallery_size;
      document.getElementById('s-active').textContent = d.active_ids;
      document.getElementById('s-xcam').textContent = d.cross_cam_ids;
      document.getElementById('s-handoffs').textContent = d.cross_cam_matches;

      const ec = d.event_counts || {{}};
      document.getElementById('s-entry').textContent = ec.entry || 0;
      document.getElementById('s-exit').textContent = ec.exit || 0;
      document.getElementById('s-dwell').textContent = ec.dwell || 0;
      document.getElementById('s-lcross').textContent = ec.line_crossing || 0;

      document.getElementById('s-fps').textContent = d.global_fps;
      document.getElementById('s-det').textContent = d.det_ms + 'ms';
      document.getElementById('s-trk').textContent = d.track_ms + 'ms';
      document.getElementById('s-reid').textContent = d.reid_ms + 'ms';
      document.getElementById('s-assoc').textContent = d.assoc_ms + 'ms';
      document.getElementById('s-events').textContent = d.events_ms + 'ms';

      document.getElementById('s-uptime').textContent = fmtUptime(d.uptime_sec || 0);
      document.getElementById('s-status').textContent =
        'Pipeline running — ' + cams.length + ' cameras';

      // Per-camera table
      const tbody = document.getElementById('cam-stats-body');
      if (d.cam_stats_rows) {{
        tbody.innerHTML = d.cam_stats_rows.map(r =>
          '<tr><td>' + r.cam + '</td>' +
          '<td class="num">' + r.dets + '</td>' +
          '<td class="num">' + r.tracks + '</td>' +
          '<td class="num">' + r.fps + '</td></tr>'
        ).join('');
      }}

      // Event timeline
      if (d.recent_events && d.recent_events.length > 0) {{
        d.recent_events.forEach(e => {{
          if (!eventLines.find(l => l.id === e.event_id)) {{
            const ts = new Date(e.timestamp * 1000).toLocaleTimeString();
            const cls = 'evt-' + e.event_type;
            const meta = e.metadata ? ' ' + JSON.stringify(e.metadata) : '';
            const html = '<span class="' + cls + '">[' + ts + '] ' +
                         e.event_type.toUpperCase() + ' G' + e.global_id +
                         ' @ ' + e.camera_id + meta + '</span>';
            eventLines.unshift({{id: e.event_id, html}});
          }}
        }});
        while (eventLines.length > maxEvents) eventLines.pop();
        document.getElementById('event-list').innerHTML =
          eventLines.map(l => l.html).join('<br>');
      }}
    }} catch(err) {{
      console.error('Stats parse error:', err);
    }}
  }};
  ws.onclose = () => {{ setTimeout(connectStats, 1500); }};
  ws.onerror = () => {{ ws.close(); }};
}}

cams.forEach(connectCam);
connectStats();
</script>
</body>
</html>"""


# Cache the HTML
_dashboard_html: str = None

@app.get("/", response_class=HTMLResponse)
async def index():
    global _dashboard_html
    if _dashboard_html is None:
        _dashboard_html = build_dashboard_html(CFG)
    return _dashboard_html


# ── REST API ─────────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def api_stats():
    """Return current pipeline stats as JSON."""
    with stats_lock:
        return JSONResponse(content=dict(stats))


@app.get("/api/events")
async def api_events():
    """Return the 100 most recent events."""
    if event_detector_ref is not None:
        return JSONResponse(content=event_detector_ref.get_recent_events(100))
    return JSONResponse(content=[])


# ── WebSocket: Stats (MUST be before /ws/{cam_id}) ──────────────────────────

@app.websocket("/ws/dashboard-stats")
async def ws_stats(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with stats_lock:
                cam_stats_rows = []
                for cid in sorted(stats["fps_per_cam"].keys()):
                    cam_stats_rows.append({
                        "cam": cid,
                        "dets": stats["det_count"].get(cid, 0),
                        "tracks": stats["track_count"].get(cid, 0),
                        "fps": stats["fps_per_cam"].get(cid, 0),
                    })
                msg = {
                    "gallery_size": stats["gallery_size"],
                    "active_ids": stats["active_ids"],
                    "inactive_ids": stats["inactive_ids"],
                    "cross_cam_ids": stats["cross_cam_ids"],
                    "cross_cam_matches": stats["cross_cam_matches"],
                    "global_fps": stats["global_fps"],
                    "det_ms": stats["det_ms"],
                    "track_ms": stats["track_ms"],
                    "reid_ms": stats["reid_ms"],
                    "assoc_ms": stats["assoc_ms"],
                    "events_ms": stats["events_ms"],
                    "draw_ms": stats["draw_ms"],
                    "event_counts": stats["event_counts"],
                    "uptime_sec": stats["uptime_sec"],
                    "cam_stats_rows": cam_stats_rows,
                }

            if event_detector_ref is not None:
                msg["recent_events"] = event_detector_ref.get_recent_events(30)
            else:
                msg["recent_events"] = []

            await websocket.send_text(json.dumps(msg))
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, Exception):
        pass


# ── WebSocket: Camera streams ────────────────────────────────────────────────

@app.websocket("/ws/{cam_id}")
async def ws_cam(websocket: WebSocket, cam_id: str):
    await websocket.accept()
    jpeg_quality = int(CFG.get("server", {}).get("ws_jpeg_quality", 65))
    try:
        while True:
            with frame_lock:
                frame = latest_frames.get(cam_id)
            if frame is not None:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                await websocket.send_bytes(buf.tobytes())
            await asyncio.sleep(0.033)  # ~30 FPS push
    except (WebSocketDisconnect, Exception):
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global CFG

    parser = argparse.ArgumentParser(description="Multi-Camera Tracking Server")
    parser.add_argument("--config", type=str, default="configs/demo.yaml")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        CFG = yaml.safe_load(f)

    server_cfg = CFG.get("server", {})
    host = args.host or server_cfg.get("host", "0.0.0.0")
    port = args.port or int(server_cfg.get("port", 8000))

    logger.info("Starting server at %s:%d (config: %s)", host, port, args.config)
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
