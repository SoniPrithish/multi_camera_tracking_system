#!/usr/bin/env python3
"""
Stage 4 Test: LIVE video — Detection + ByteTrack + DeepReID
FULLY OPTIMIZED for RTX 4060 Ti 16GB + 24 CPU threads.

Architecture:
  - Ingest: 6 async camera threads (already fast)
  - Pipeline: SINGLE thread, BATCH all 6 cameras through GPU together
  - Detection: BATCH inference (6 frames → 1 YOLO call, FP16)
  - ReID: BATCH all crops (up to 64 at once, FP16), only every Nth frame
  - Tracking: per-camera ByteTrack (CPU, instant)
  - Serving: WebSocket push in async loop

Open http://<host>:8890 in browser.
"""
import os, sys, time, json, threading, logging, yaml, hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("stage4_opt")

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
from src.reid.deep import DeepReID

# ── Load config ──
with open("configs/demo.yaml") as f:
    cfg = yaml.safe_load(f)

CAMERAS = cfg["inputs"]
INGEST_CFG = cfg.get("ingest", {})
INGEST_CFG["resize_width"] = cfg.get("runtime", {}).get("resize_width", 960)
INGEST_CFG["target_fps"] = 30  # Push ingest higher — we can handle it
MOD = cfg.get("modules", {})

# ── Tuning ──
REID_EVERY_N = 8          # Run ReID every Nth pipeline loop (others use cache)
MAX_REID_CROPS = 40        # Max total crops across all cameras per ReID batch

app = FastAPI()

# ── Shared state ──
latest_frames = {}
frame_lock = threading.Lock()
stats = {
    "det_count": {}, "track_count": {}, "fps_per_cam": {},
    "latency": {}, "global_fps": 0.0, "gallery_size": 0,
    "det_ms": 0.0, "track_ms": 0.0, "reid_ms": 0.0, "draw_ms": 0.0,
}
stats_lock = threading.Lock()


def gid_to_color(gid: int):
    h = int(hashlib.md5(str(gid).encode()).hexdigest()[:6], 16)
    hue = h % 180
    c = cv2.cvtColor(np.array([[[hue, 230, 240]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(x) for x in c)


def pipeline_loop():
    """SINGLE pipeline thread — batch all cameras through GPU together."""
    reader = MultiVideoReader(CAMERAS, ingest_cfg=INGEST_CFG)

    detector = YOLOv8Detector(
        model=MOD.get("detector_model", "yolov8n.pt"),
        device=str(MOD.get("detector_device", "0")),
        conf=float(MOD.get("detector_conf", 0.35)),
        iou=float(MOD.get("detector_iou", 0.45)),
        img_size=int(MOD.get("detector_img_size", 640)),
        half=True,
    )
    tracker = ByteTracker(
        max_age=int(MOD.get("tracker_max_age", 30)),
        min_hits=int(MOD.get("tracker_min_hits", 3)),
        iou_thresh=float(MOD.get("tracker_iou_thresh", 0.2)),
    )
    reid = DeepReID(
        model=MOD.get("reid_model", "osnet_x0_25"),
        device=str(MOD.get("reid_device", MOD.get("detector_device", "0"))),
        sim_thresh=float(MOD.get("reid_sim_thresh", 0.45)),
        ema_alpha=float(MOD.get("reid_ema_alpha", 0.7)),
        gallery_ttl=float(MOD.get("reid_gallery_ttl", 120)),
        max_gallery_size=int(MOD.get("reid_max_gallery", 500)),
    )

    trail_history = defaultdict(lambda: defaultdict(list))
    MAX_TRAIL = 30
    cached_gids = {}  # cam_id -> {tid: gid}

    loop_count = 0
    total_frames = 0
    t_global_start = time.monotonic()
    cam_counters = {}

    logger.info("BATCH pipeline started — %d cameras, FP16=True", len(CAMERAS))

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

        # ━━━ ReID (batched, only every Nth loop) ━━━
        run_reid = (loop_count % REID_EVERY_N == 1) or loop_count <= 2
        t_reid = time.monotonic()

        if run_reid:
            # Collect all crops across all cameras
            all_crops = []
            crop_map = []  # (cam_id, tid, index_in_all_crops)
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
                # ONE batch GPU call for ALL crops
                raw_embs = reid.extractor.extract(all_crops)

                # Distribute embeddings back to per-camera tracks
                cam_embs = defaultdict(dict)
                for idx, (cam_id, tid) in enumerate(crop_map):
                    cam_embs[cam_id][tid] = raw_embs[idx]

                # Assign global IDs per camera
                new_cached_gids = {}
                for cam_id in cam_ids:
                    embs_dict = cam_embs.get(cam_id, {})
                    # Fill missing with zeros
                    for t in all_tracks[cam_id]:
                        if t["tid"] not in embs_dict:
                            embs_dict[t["tid"]] = np.zeros(reid.embed_dim, dtype=np.float32)
                    gids = reid.assign_global_ids(cam_id, all_tracks[cam_id], embs_dict)
                    new_cached_gids[cam_id] = gids

                cached_gids = new_cached_gids

        reid_ms = (time.monotonic() - t_reid) * 1000

        # ━━━ DRAW (parallel with CPU threads for speed) ━━━
        t_draw = time.monotonic()

        for i, cam_id in enumerate(cam_ids):
            frame = frames[i]
            tracks = all_tracks[cam_id]
            gids = cached_gids.get(cam_id, {})

            vis = frame.copy()
            for t in tracks:
                tid = t["tid"]
                x, y, w, h = map(int, t["bbox"])
                gid = gids.get(tid, tid + 10000)
                color = gid_to_color(gid)

                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                label = f"G{gid}"
                (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(vis, (x, max(0, y - th_t - 8)), (x + tw + 4, y), color, -1)
                cv2.putText(vis, label, (x + 2, max(th_t + 2, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                # Trail
                cx, cy = x + w // 2, y + h
                trail_history[cam_id][gid].append((cx, cy))
                if len(trail_history[cam_id][gid]) > MAX_TRAIL:
                    trail_history[cam_id][gid] = trail_history[cam_id][gid][-MAX_TRAIL:]
                pts = trail_history[cam_id][gid]
                for j in range(1, len(pts)):
                    th = max(1, int(2 * j / MAX_TRAIL) + 1)
                    cv2.line(vis, pts[j - 1], pts[j], color, th)

            # Cleanup trails
            active_gids = set(gids.values())
            for gid_key in list(trail_history.get(cam_id, {}).keys()):
                if gid_key not in active_gids:
                    del trail_history[cam_id][gid_key]

            # Per-camera FPS
            cam_counters.setdefault(cam_id, {"n": 0, "t": time.monotonic()})
            cam_counters[cam_id]["n"] += 1
            ce = time.monotonic() - cam_counters[cam_id]["t"]
            cam_fps = cam_counters[cam_id]["n"] / max(0.01, ce)

            reid_tag = "R" if run_reid else "C"
            hud = f"{cam_id} | {len(batch_dets[i])}d {len(tracks)}t | {cam_fps:.1f}FPS | det:{det_ms:.0f} trk:{track_ms:.0f} reid:{reid_ms:.0f}ms [{reid_tag}]"
            cv2.putText(vis, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            with frame_lock:
                latest_frames[cam_id] = vis

        draw_ms = (time.monotonic() - t_draw) * 1000

        total_frames += len(cam_ids)
        elapsed = time.monotonic() - t_global_start
        gfps = total_frames / max(0.01, elapsed)

        with stats_lock:
            for i, cam_id in enumerate(cam_ids):
                stats["det_count"][cam_id] = len(batch_dets[i])
                stats["track_count"][cam_id] = len(all_tracks[cam_id])
                stats["fps_per_cam"][cam_id] = round(cam_counters[cam_id]["n"] / max(0.01, time.monotonic() - cam_counters[cam_id]["t"]), 1)
            stats["global_fps"] = round(gfps, 1)
            stats["gallery_size"] = reid.gallery_size
            stats["det_ms"] = round(det_ms, 1)
            stats["track_ms"] = round(track_ms, 1)
            stats["reid_ms"] = round(reid_ms, 1)
            stats["draw_ms"] = round(draw_ms, 1)


@app.on_event("startup")
async def startup():
    t = threading.Thread(target=pipeline_loop, daemon=True)
    t.start()
    logger.info("Batch pipeline thread started.")


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
<title>Stage 4: DeepReID Live (Batch GPU)</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif; margin: 0; padding: 12px; }}
  h1 {{ text-align: center; color: #ffa657; margin-bottom: 4px; font-size: 22px; }}
  .sub {{ text-align: center; color: #8b949e; font-size: 12px; margin-bottom: 6px; }}
  #stats {{ text-align: center; margin: 6px; padding: 8px; background: #161b22;
            border-radius: 8px; font-size: 14px; color: #7ee787; font-family: monospace; }}
  .grid {{ display: grid; grid-template-columns: repeat({cols}, 1fr); gap: 8px; }}
  .cam-box {{ background: #161b22; border-radius: 8px; overflow: hidden;
              border: 1px solid #30363d; }}
  .cam-box h3 {{ text-align: center; margin: 6px 0 2px 0; color: #79c0ff; font-size: 13px; }}
  canvas {{ width: 100%; display: block; }}
</style>
</head>
<body>
<h1>🔍 Stage 4: Deep Re-ID — Batch GPU Pipeline</h1>
<p class="sub">RTX 4060 Ti 16GB · FP16 · Batch detect (6 cams) · Batch ReID · ByteTrack</p>
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
    const ws = new WebSocket('ws://' + location.host + '/ws/stats');
    ws.onmessage = (evt) => {{ document.getElementById('stats').innerText = evt.data; }};
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
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                await websocket.send_bytes(buf.tobytes())
            await asyncio.sleep(0.033)  # ~30 FPS push
    except (WebSocketDisconnect, Exception):
        pass


@app.websocket("/ws/stats")
async def ws_stats(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            with stats_lock:
                parts = []
                for cid in sorted(stats["fps_per_cam"].keys()):
                    d = stats["det_count"].get(cid, 0)
                    t = stats["track_count"].get(cid, 0)
                    f = stats["fps_per_cam"].get(cid, 0)
                    parts.append(f"{cid}:{d}d/{t}t {f}fps")
                cam_str = " │ ".join(parts) if parts else "Starting..."
                msg = (f"Gallery:{stats['gallery_size']} │ {stats['global_fps']}FPS total │ "
                       f"det:{stats['det_ms']}ms trk:{stats['track_ms']}ms "
                       f"reid:{stats['reid_ms']}ms draw:{stats['draw_ms']}ms ━━ {cam_str}")
            await websocket.send_text(msg)
            await asyncio.sleep(0.4)
    except (WebSocketDisconnect, Exception):
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8890, log_level="info")
