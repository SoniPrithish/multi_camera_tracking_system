#!/usr/bin/env python3
"""
Stage 6 Test: Events — Entry/Exit, Dwell, Line Crossing
========================================================

Full pipeline: Ingest → Detection → Tracking → ReID → Cross-Camera → Events
All stages running on RTX 4060 Ti with batch FP16 GPU inference.

New in Stage 6:
  - Entry events: fires when a global ID first appears in a camera
  - Exit events: fires when a global ID disappears from a camera (3s timeout)
  - Dwell events: fires when a person stays in view for > 10 seconds
  - Line crossing: auto-generated horizontal line at 60% frame height
  - Event timeline in the dashboard (scrolling log)
  - Event counters (entry/exit/dwell/crossing)

Open http://<host>:8892 in browser.
"""
import os, sys, time, json, threading, logging, yaml, hashlib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("stage6")

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
from src.events.event_detector import EventDetector

# ── Load config ──
with open("configs/demo.yaml") as f:
    cfg = yaml.safe_load(f)

CAMERAS = cfg["inputs"]
INGEST_CFG = cfg.get("ingest", {})
INGEST_CFG["resize_width"] = cfg.get("runtime", {}).get("resize_width", 960)
INGEST_CFG["target_fps"] = 30
MOD = cfg.get("modules", {})
ASSOC_CFG = cfg.get("association", {})
EVENTS_CFG = cfg.get("events", {})

# ── Tuning ──
REID_EVERY_N = 8
MAX_REID_CROPS = 48

app = FastAPI()

# ── Shared state ──
latest_frames = {}
frame_lock = threading.Lock()
stats = {
    "det_count": {}, "track_count": {}, "fps_per_cam": {},
    "global_fps": 0.0,
    "det_ms": 0.0, "track_ms": 0.0, "reid_ms": 0.0,
    "assoc_ms": 0.0, "events_ms": 0.0, "draw_ms": 0.0,
    "gallery_size": 0, "active_ids": 0, "inactive_ids": 0,
    "cross_cam_ids": 0, "cross_cam_matches": 0,
    "event_counts": {"entry": 0, "exit": 0, "dwell": 0, "line_crossing": 0, "total": 0},
}
stats_lock = threading.Lock()

# Event detector (shared)
event_detector: EventDetector = None
event_detector_lock = threading.Lock()


def gid_to_color(gid: int):
    h = int(hashlib.md5(str(gid).encode()).hexdigest()[:6], 16)
    hue = h % 180
    c = cv2.cvtColor(np.array([[[hue, 230, 240]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(x) for x in c)


def pipeline_loop():
    global event_detector

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
    reid_extractor = OSNetExtractor(
        model_name=MOD.get("reid_model", "osnet_x0_25"),
        device=str(MOD.get("reid_device", MOD.get("detector_device", "0"))),
        half=True,
    )
    cross_cam = CrossCameraAssociator(ASSOC_CFG)
    event_detector = EventDetector(EVENTS_CFG)

    trail_history = defaultdict(lambda: defaultdict(list))
    MAX_TRAIL = 30
    cached_gids = {}

    loop_count = 0
    total_frames = 0
    t_global_start = time.monotonic()
    cam_counters = {}

    logger.info(
        "STAGE 6 pipeline started — %d cameras, events=%s",
        len(CAMERAS),
        {k: v.get("enabled", True) for k, v in EVENTS_CFG.items() if isinstance(v, dict)},
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

        # ━━━ BATCH DETECTION ━━━
        t_det = time.monotonic()
        batch_dets = detector.detect_batch(frames)
        det_ms = (time.monotonic() - t_det) * 1000

        # ━━━ TRACKING ━━━
        t_trk = time.monotonic()
        all_tracks = {}
        for i, cam_id in enumerate(cam_ids):
            all_tracks[cam_id] = tracker.update(cam_id, batch_dets[i])
        track_ms = (time.monotonic() - t_trk) * 1000

        # ━━━ ReID ━━━
        run_reid = (loop_count % REID_EVERY_N == 1) or loop_count <= 2
        t_reid = time.monotonic()

        all_embs: dict = defaultdict(dict)
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
        reid_ms = (time.monotonic() - t_reid) * 1000

        # ━━━ CROSS-CAMERA ASSOCIATION ━━━
        t_assoc = time.monotonic()
        if run_reid and all_embs:
            cached_gids = cross_cam.associate(all_tracks, all_embs)
        else:
            cached_gids = cross_cam.associate(all_tracks, defaultdict(dict))
        assoc_ms = (time.monotonic() - t_assoc) * 1000

        # ━━━ EVENTS (Stage 6) ━━━
        t_events = time.monotonic()
        all_new_events = []
        for i, cam_id in enumerate(cam_ids):
            fh, fw = frames[i].shape[:2]
            new_events = event_detector.process(
                cam_id=cam_id,
                tracks=all_tracks[cam_id],
                gids=cached_gids.get(cam_id, {}),
                frame_shape=(fh, fw),
            )
            all_new_events.extend(new_events)
        events_ms = (time.monotonic() - t_events) * 1000

        # ━━━ DRAW ━━━
        t_draw = time.monotonic()

        for i, cam_id in enumerate(cam_ids):
            frame = frames[i]
            tracks = all_tracks[cam_id]
            gids = cached_gids.get(cam_id, {})
            fh, fw = frame.shape[:2]

            vis = frame.copy()

            # Draw auto-generated line at 60% height
            line_y = int(fh * 0.6)
            cv2.line(vis, (0, line_y), (fw, line_y), (0, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(vis, "LINE", (fw - 55, line_y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)

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
                (tw, th_t), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(vis, (x, max(0, y - th_t - 8)), (x + tw + 4, y), color, -1)
                cv2.putText(vis, label, (x + 2, max(th_t + 2, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

                # Trail
                trail_key = gid if gid is not None else f"T{tid}"
                cx, cy = x + w // 2, y + h
                trail_history[cam_id][trail_key].append((cx, cy))
                if len(trail_history[cam_id][trail_key]) > MAX_TRAIL:
                    trail_history[cam_id][trail_key] = trail_history[cam_id][trail_key][-MAX_TRAIL:]
                pts = trail_history[cam_id][trail_key]
                for j in range(1, len(pts)):
                    th = max(1, int(2 * j / MAX_TRAIL) + 1)
                    cv2.line(vis, pts[j - 1], pts[j], color, th)

            # Cleanup trails
            active_trail_keys = set()
            for t in tracks:
                gid = gids.get(t["tid"])
                active_trail_keys.add(gid if gid is not None else f"T{t['tid']}")
            for tk in list(trail_history.get(cam_id, {}).keys()):
                if tk not in active_trail_keys:
                    del trail_history[cam_id][tk]

            # HUD
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

        # Update stats
        assoc_stats = cross_cam.get_stats()
        evt_counts = event_detector.get_counts()
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


@app.on_event("startup")
async def startup():
    t = threading.Thread(target=pipeline_loop, daemon=True)
    t.start()
    logger.info("Stage 6 pipeline thread started.")


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
<title>Stage 6: Events — Entry/Exit, Dwell, Line Crossing</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', sans-serif;
         margin: 0; padding: 10px; }}
  h1 {{ text-align: center; color: #f0883e; margin-bottom: 2px; font-size: 20px; }}
  .sub {{ text-align: center; color: #8b949e; font-size: 11px; margin-bottom: 4px; }}
  #stats {{ text-align: center; margin: 4px; padding: 8px; background: #161b22;
            border-radius: 8px; font-size: 12px; color: #7ee787; font-family: monospace;
            line-height: 1.5; }}
  .stat-row {{ display: flex; justify-content: center; gap: 18px; flex-wrap: wrap; }}
  .stat-label {{ color: #8b949e; }}
  .stat-val {{ color: #ffa657; font-weight: bold; }}
  .stat-xcam {{ color: #f778ba; font-weight: bold; }}
  .stat-evt {{ color: #a5d6ff; font-weight: bold; }}
  .grid {{ display: grid; grid-template-columns: repeat({cols}, 1fr); gap: 6px; }}
  .cam-box {{ background: #161b22; border-radius: 8px; overflow: hidden;
              border: 1px solid #30363d; }}
  .cam-box h3 {{ text-align: center; margin: 4px 0 2px 0; color: #79c0ff; font-size: 12px; }}
  canvas {{ width: 100%; display: block; }}
  #event-timeline {{
    margin: 6px; padding: 8px; background: #161b22; border-radius: 8px;
    border: 1px solid #30363d; max-height: 180px; overflow-y: auto;
    font-family: monospace; font-size: 11px; line-height: 1.5;
  }}
  #event-timeline h3 {{ color: #f0883e; margin: 0 0 4px 0; font-size: 13px; }}
  .evt-entry {{ color: #3fb950; }}
  .evt-exit {{ color: #f85149; }}
  .evt-dwell {{ color: #d29922; }}
  .evt-line_crossing {{ color: #a5d6ff; }}
</style>
</head>
<body>
<h1>📋 Stage 6: Events — Entry/Exit · Dwell · Line Crossing</h1>
<p class="sub">Full Pipeline: Detect→Track→ReID→CrossCam→Events | RTX 4060 Ti FP16</p>
<div id="stats">Connecting...</div>
<div class="grid">
{cam_divs}
</div>
<div id="event-timeline"><h3>📝 Event Timeline</h3><div id="event-list">Waiting for events...</div></div>
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
const maxEventLines = 100;
let eventLines = [];
function connectStats() {{
    const ws = new WebSocket('ws://' + location.host + '/ws/dashboard-stats');
    ws.onmessage = (evt) => {{
        try {{
            const d = JSON.parse(evt.data);
            // Stats bar
            let html = '<div class="stat-row">';
            html += '<span><span class="stat-label">Gallery:</span> <span class="stat-val">' + d.gallery_size + '</span></span>';
            html += '<span><span class="stat-label">Active:</span> <span class="stat-val">' + d.active_ids + '</span></span>';
            html += '<span><span class="stat-label">Cross-Cam:</span> <span class="stat-xcam">' + d.cross_cam_ids + '</span></span>';
            html += '<span><span class="stat-label">Handoffs:</span> <span class="stat-xcam">' + d.cross_cam_matches + '</span></span>';
            const ec = d.event_counts || {{}};
            html += '<span><span class="stat-label">Entry:</span> <span class="stat-evt">' + (ec.entry||0) + '</span></span>';
            html += '<span><span class="stat-label">Exit:</span> <span class="stat-evt">' + (ec.exit||0) + '</span></span>';
            html += '<span><span class="stat-label">Dwell:</span> <span class="stat-evt">' + (ec.dwell||0) + '</span></span>';
            html += '<span><span class="stat-label">LineCross:</span> <span class="stat-evt">' + (ec.line_crossing||0) + '</span></span>';
            html += '</div>';
            html += '<div class="stat-row" style="margin-top:3px">';
            html += '<span><span class="stat-label">FPS:</span> <span class="stat-val">' + d.global_fps + '</span></span>';
            html += '<span><span class="stat-label">Det:</span> ' + d.det_ms + 'ms</span>';
            html += '<span><span class="stat-label">Trk:</span> ' + d.track_ms + 'ms</span>';
            html += '<span><span class="stat-label">ReID:</span> ' + d.reid_ms + 'ms</span>';
            html += '<span><span class="stat-label">Assoc:</span> ' + d.assoc_ms + 'ms</span>';
            html += '<span><span class="stat-label">Events:</span> ' + d.events_ms + 'ms</span>';
            html += '</div>';
            if (d.cam_stats) {{
                html += '<div class="stat-row" style="margin-top:3px; font-size:11px">';
                d.cam_stats.forEach(cs => {{ html += '<span>' + cs + '</span>'; }});
                html += '</div>';
            }}
            document.getElementById('stats').innerHTML = html;
            // Event timeline
            if (d.recent_events && d.recent_events.length > 0) {{
                d.recent_events.forEach(e => {{
                    const ts = new Date(e.timestamp * 1000).toLocaleTimeString();
                    const cls = 'evt-' + e.event_type;
                    const meta = e.metadata ? JSON.stringify(e.metadata) : '';
                    const line = '<span class="' + cls + '">[' + ts + '] ' +
                                 e.event_type.toUpperCase() + ' G' + e.global_id +
                                 ' @ ' + e.camera_id + ' ' + meta + '</span>';
                    // Deduplicate by event_id
                    if (!eventLines.find(l => l.id === e.event_id)) {{
                        eventLines.unshift({{id: e.event_id, html: line}});
                    }}
                }});
                while (eventLines.length > maxEventLines) eventLines.pop();
                document.getElementById('event-list').innerHTML =
                    eventLines.map(l => l.html).join('<br>');
            }}
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
    """Stats + events endpoint — BEFORE /ws/{{cam_id}} to avoid capture."""
    await websocket.accept()
    try:
        while True:
            with stats_lock:
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
                    "events_ms": stats["events_ms"],
                    "draw_ms": stats["draw_ms"],
                    "event_counts": stats["event_counts"],
                    "cam_stats": cam_stats_list,
                }

            # Get recent events
            if event_detector is not None:
                msg["recent_events"] = event_detector.get_recent_events(30)
            else:
                msg["recent_events"] = []

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
            await asyncio.sleep(0.033)
    except (WebSocketDisconnect, Exception):
        pass


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8892, log_level="info")
