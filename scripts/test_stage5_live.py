#!/usr/bin/env python3
"""
Stage 5 Test: Cross-Camera Association — LIVE video
====================================================

Architecture (same batch-GPU pipeline as Stage 4, plus cross-camera matching):
  - Ingest: 6 async camera threads
  - Detection: BATCH inference (6 frames → 1 YOLO call, FP16)
  - Tracking: per-camera ByteTrack (CPU)
  - ReID: BATCH all crops (FP16), every Nth frame
  - Cross-Camera: Hungarian matching + time gating + global ID registry   <-- NEW
  - Serving: WebSocket push

What's new vs Stage 4:
  - CrossCameraAssociator replaces per-camera reid.assign_global_ids()
  - Global IDs are STABLE across cameras (same person = same color/ID everywhere)
  - Gallery maturity gate: entries need 2+ hits before matching (precision)
  - Time-of-flight gating: inactive entries expire after configurable TTL
  - Dashboard shows cross-camera stats (unique IDs, handoff count)

Open http://<host>:8891 in browser.
"""
import os, sys, time, json, threading, logging, yaml, hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("stage5")

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
from src.reid.osnet import OSNetExtractor
from src.association.cross_camera import CrossCameraAssociator

# ── Load config ──
with open("configs/demo.yaml") as f:
    cfg = yaml.safe_load(f)

CAMERAS = cfg["inputs"]
INGEST_CFG = cfg.get("ingest", {})
INGEST_CFG["resize_width"] = cfg.get("runtime", {}).get("resize_width", 960)
INGEST_CFG["target_fps"] = 30
MOD = cfg.get("modules", {})
ASSOC_CFG = cfg.get("association", {})

# ── Tuning ──
REID_EVERY_N = 8          # Run ReID every Nth pipeline loop
MAX_REID_CROPS = 48        # Max crops across all cameras per ReID batch

app = FastAPI()

# ── Shared state ──
latest_frames = {}
frame_lock = threading.Lock()
stats = {
    "det_count": {}, "track_count": {}, "fps_per_cam": {},
    "latency": {}, "global_fps": 0.0,
    "det_ms": 0.0, "track_ms": 0.0, "reid_ms": 0.0,
    "assoc_ms": 0.0, "draw_ms": 0.0,
    "gallery_size": 0, "active_ids": 0, "inactive_ids": 0,
    "cross_cam_ids": 0, "cross_cam_matches": 0,
}
stats_lock = threading.Lock()


def gid_to_color(gid: int):
    """Deterministic vivid color from global ID."""
    h = int(hashlib.md5(str(gid).encode()).hexdigest()[:6], 16)
    hue = h % 180
    c = cv2.cvtColor(np.array([[[hue, 230, 240]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(x) for x in c)


def pipeline_loop():
    """SINGLE pipeline thread — batch all cameras through GPU together."""
    reader = MultiVideoReader(CAMERAS, ingest_cfg=INGEST_CFG)

    # ── Detection (batch GPU, FP16) ──
    detector = YOLOv8Detector(
        model=MOD.get("detector_model", "yolov8n.pt"),
        device=str(MOD.get("detector_device", "0")),
        conf=float(MOD.get("detector_conf", 0.35)),
        iou=float(MOD.get("detector_iou", 0.45)),
        img_size=int(MOD.get("detector_img_size", 640)),
        half=True,
    )

    # ── Tracking (per-camera, CPU) ──
    tracker = ByteTracker(
        max_age=int(MOD.get("tracker_max_age", 30)),
        min_hits=int(MOD.get("tracker_min_hits", 3)),
        iou_thresh=float(MOD.get("tracker_iou_thresh", 0.2)),
    )

    # ── ReID feature extractor (batch GPU, FP16) ──
    reid_extractor = OSNetExtractor(
        model_name=MOD.get("reid_model", "osnet_x0_25"),
        device=str(MOD.get("reid_device", MOD.get("detector_device", "0"))),
        half=True,
    )

    # ── Cross-Camera Associator (Stage 5) ──
    cross_cam = CrossCameraAssociator(ASSOC_CFG)

    trail_history = defaultdict(lambda: defaultdict(list))
    MAX_TRAIL = 30
    cached_gids = {}   # cam_id -> {tid: gid}

    loop_count = 0
    total_frames = 0
    t_global_start = time.monotonic()
    cam_counters = {}

    logger.info(
        "STAGE 5 pipeline started — %d cameras, FP16=True, "
        "sim_thresh=%.2f, confirm=%d",
        len(CAMERAS), cross_cam.sim_thresh, cross_cam.confirm_hits,
    )

    while True:
        batch = reader.read()
        if batch is None:
            break
        if not batch:
            continue

        loop_count += 1
        cam_ids = list(batch.keys())
        frames = [batch[cid] for cid in cam_ids]

        # ━━━ BATCH DETECTION (all cameras in ONE GPU call) ━━━
        t_det = time.monotonic()
        batch_dets = detector.detect_batch(frames)
        det_ms = (time.monotonic() - t_det) * 1000

        # ━━━ TRACKING (per-camera, CPU — fast) ━━━
        t_trk = time.monotonic()
        all_tracks = {}
        for i, cam_id in enumerate(cam_ids):
            all_tracks[cam_id] = tracker.update(cam_id, batch_dets[i])
        track_ms = (time.monotonic() - t_trk) * 1000

        # ━━━ ReID embeddings (batched, only every Nth loop) ━━━
        run_reid = (loop_count % REID_EVERY_N == 1) or loop_count <= 2
        t_reid = time.monotonic()

        all_embs: dict = defaultdict(dict)   # cam_id -> {tid: np.ndarray}

        if run_reid:
            all_crops = []
            crop_map = []   # [(cam_id, tid), ...]

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

        reid_ms = (time.monotonic() - t_reid) * 1000

        # ━━━ CROSS-CAMERA ASSOCIATION (Stage 5) ━━━
        t_assoc = time.monotonic()

        if run_reid and all_embs:
            cached_gids = cross_cam.associate(all_tracks, all_embs)
        else:
            # Non-ReID frame: call associate with empty embs to maintain
            # track→gid mapping for continuing tracks
            cached_gids = cross_cam.associate(all_tracks, defaultdict(dict))

        assoc_ms = (time.monotonic() - t_assoc) * 1000

        # ━━━ DRAW ━━━
        t_draw = time.monotonic()

        for i, cam_id in enumerate(cam_ids):
            frame = frames[i]
            tracks = all_tracks[cam_id]
            gids = cached_gids.get(cam_id, {})

            vis = frame.copy()
            for t in tracks:
                tid = t["tid"]
                x, y, w, h = map(int, t["bbox"])
                gid = gids.get(tid)

                if gid is not None:
                    color = gid_to_color(gid)
                    label = f"G{gid}"
                else:
                    color = (100, 100, 100)   # grey for unassigned
                    label = f"T{tid}"

                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(vis, (x, max(0, y - th_t - 8)), (x + tw + 4, y), color, -1)
                cv2.putText(vis, label, (x + 2, max(th_t + 2, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                # Trail (keyed by global ID for consistency across cameras)
                trail_key = gid if gid is not None else f"T{tid}"
                cx, cy = x + w // 2, y + h
                trail_history[cam_id][trail_key].append((cx, cy))
                if len(trail_history[cam_id][trail_key]) > MAX_TRAIL:
                    trail_history[cam_id][trail_key] = trail_history[cam_id][trail_key][-MAX_TRAIL:]
                pts = trail_history[cam_id][trail_key]
                for j in range(1, len(pts)):
                    th = max(1, int(2 * j / MAX_TRAIL) + 1)
                    cv2.line(vis, pts[j - 1], pts[j], color, th)

            # Cleanup old trails
            active_trail_keys = set()
            for t in tracks:
                gid = gids.get(t["tid"])
                active_trail_keys.add(gid if gid is not None else f"T{t['tid']}")
            for tk in list(trail_history.get(cam_id, {}).keys()):
                if tk not in active_trail_keys:
                    del trail_history[cam_id][tk]

            # Per-camera FPS counter
            cam_counters.setdefault(cam_id, {"n": 0, "t": time.monotonic()})
            cam_counters[cam_id]["n"] += 1
            ce = time.monotonic() - cam_counters[cam_id]["t"]
            cam_fps = cam_counters[cam_id]["n"] / max(0.01, ce)

            reid_tag = "R" if run_reid else "C"
            hud = (f"{cam_id} | {len(batch_dets[i])}d {len(tracks)}t "
                   f"| {cam_fps:.1f}FPS [{reid_tag}]")
            cv2.putText(vis, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            with frame_lock:
                latest_frames[cam_id] = vis

        draw_ms = (time.monotonic() - t_draw) * 1000

        total_frames += len(cam_ids)
        elapsed = time.monotonic() - t_global_start
        gfps = total_frames / max(0.01, elapsed)

        # ── Update stats ──
        assoc_stats = cross_cam.get_stats()
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
            stats["draw_ms"] = round(draw_ms, 1)
            stats["gallery_size"] = assoc_stats["gallery_size"]
            stats["active_ids"] = assoc_stats["active"]
            stats["inactive_ids"] = assoc_stats["inactive"]
            stats["cross_cam_ids"] = assoc_stats["cross_camera_ids"]
            stats["cross_cam_matches"] = assoc_stats["cross_cam_matches"]


@app.on_event("startup")
async def startup():
    t = threading.Thread(target=pipeline_loop, daemon=True)
    t.start()
    logger.info("Stage 5 pipeline thread started.")


def build_dashboard_html():
    cam_ids = [c["camera_id"] for c in CAMERAS]
    cols = min(len(cam_ids), 3)
    cam_divs = "\n".join(
        f'  <div class="cam-box"><h3>{cid.upper()}</h3>'
        f'<canvas id="canvas-{cid}"></canvas></div>'
        for cid in cam_ids
    )
    js_cams = json.dumps(cam_ids)

    return f"""<!DOCTYPE html>
<html>
<head>
<title>Stage 5: Cross-Camera Association</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif;
         margin: 0; padding: 12px; }}
  h1 {{ text-align: center; color: #f78166; margin-bottom: 4px; font-size: 22px; }}
  .sub {{ text-align: center; color: #8b949e; font-size: 12px; margin-bottom: 6px; }}
  #stats {{ text-align: center; margin: 6px; padding: 10px; background: #161b22;
            border-radius: 8px; font-size: 13px; color: #7ee787; font-family: monospace;
            line-height: 1.6; }}
  .stat-row {{ display: flex; justify-content: center; gap: 24px; flex-wrap: wrap; }}
  .stat-label {{ color: #8b949e; }}
  .stat-val {{ color: #ffa657; font-weight: bold; }}
  .stat-xcam {{ color: #f778ba; font-weight: bold; }}
  .grid {{ display: grid; grid-template-columns: repeat({cols}, 1fr); gap: 8px; }}
  .cam-box {{ background: #161b22; border-radius: 8px; overflow: hidden;
              border: 1px solid #30363d; }}
  .cam-box h3 {{ text-align: center; margin: 6px 0 2px 0; color: #79c0ff; font-size: 13px; }}
  canvas {{ width: 100%; display: block; }}
</style>
</head>
<body>
<h1>🔗 Stage 5: Cross-Camera Association</h1>
<p class="sub">Hungarian Matching · Time-of-Flight Gating · Global ID Registry ·
  sim≥0.55 · confirm≥2 · RTX 4060 Ti FP16</p>
<div id="stats">Connecting...</div>
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
    const ws = new WebSocket('ws://' + location.host + '/ws/dashboard-stats');
    ws.onmessage = (evt) => {{
        try {{
            const d = JSON.parse(evt.data);
            let html = '<div class="stat-row">';
            html += '<span><span class="stat-label">Gallery:</span> <span class="stat-val">' + d.gallery_size + '</span></span>';
            html += '<span><span class="stat-label">Active:</span> <span class="stat-val">' + d.active_ids + '</span></span>';
            html += '<span><span class="stat-label">Inactive:</span> <span class="stat-val">' + d.inactive_ids + '</span></span>';
            html += '<span><span class="stat-label">Cross-Cam IDs:</span> <span class="stat-xcam">' + d.cross_cam_ids + '</span></span>';
            html += '<span><span class="stat-label">Handoffs:</span> <span class="stat-xcam">' + d.cross_cam_matches + '</span></span>';
            html += '</div>';
            html += '<div class="stat-row" style="margin-top:4px">';
            html += '<span><span class="stat-label">FPS:</span> <span class="stat-val">' + d.global_fps + '</span></span>';
            html += '<span><span class="stat-label">Det:</span> ' + d.det_ms + 'ms</span>';
            html += '<span><span class="stat-label">Trk:</span> ' + d.track_ms + 'ms</span>';
            html += '<span><span class="stat-label">ReID:</span> ' + d.reid_ms + 'ms</span>';
            html += '<span><span class="stat-label">Assoc:</span> ' + d.assoc_ms + 'ms</span>';
            html += '</div>';
            // Per-camera stats
            if (d.cam_stats) {{
                html += '<div class="stat-row" style="margin-top:4px; font-size:12px">';
                d.cam_stats.forEach(cs => {{
                    html += '<span>' + cs + '</span>';
                }});
                html += '</div>';
            }}
            document.getElementById('stats').innerHTML = html;
        }} catch(e) {{
            document.getElementById('stats').innerText = evt.data;
        }}
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


@app.websocket("/ws/dashboard-stats")
async def ws_stats(websocket: WebSocket):
    """Stats endpoint — MUST be before /ws/{cam_id} to avoid route capture."""
    await websocket.accept()
    try:
        while True:
            with stats_lock:
                # Build per-camera summary
                cam_stats_list = []
                for cid in sorted(stats["fps_per_cam"].keys()):
                    d = stats["det_count"].get(cid, 0)
                    t = stats["track_count"].get(cid, 0)
                    f = stats["fps_per_cam"].get(cid, 0)
                    cam_stats_list.append(f"{cid}:{d}d/{t}t {f}fps")

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
                    "cam_stats": cam_stats_list,
                }
            await websocket.send_text(json.dumps(msg))
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, Exception):
        pass


@app.websocket("/ws/{cam_id}")
async def ws_cam(websocket: WebSocket, cam_id: str):
    await websocket.accept()
    try:
        while True:
            with frame_lock:
                frame = latest_frames.get(cam_id)
            if frame is not None:
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                await websocket.send_bytes(buf.tobytes())
            await asyncio.sleep(0.033)  # ~30 FPS push
    except (WebSocketDisconnect, Exception):
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8891, log_level="info")
