"""
Event Detection — Stage 6
=========================

Detects three event types from tracked persons:

1. **Entry / Exit** — fires once when a global ID first appears in a camera
   (entry) and once when it disappears for > `exit_timeout` seconds (exit).
2. **Dwell** — fires once when a person stays in a camera view for longer
   than `threshold_sec` seconds.
3. **Line Crossing** — fires when a person's bounding-box center crosses a
   virtual line segment.  Direction-aware (A→B or B→A).

Deduplication
-------------
Each event is uniquely keyed by ``(type, global_id, camera_id, ...)``.
The same event will NEVER fire twice for the same person in the same camera
until they fully exit and re-enter.

Thread-safe (single internal lock).
"""

from __future__ import annotations

import time
import logging
import threading
import hashlib
from typing import Dict, List, Set, Tuple, Any, Optional
from collections import deque
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


# ── Event data class ─────────────────────────────────────────────────────────

@dataclass
class Event:
    """A single detected event."""
    event_id: str            # Unique hash
    event_type: str          # entry | exit | dwell | line_crossing
    global_id: int           # Person global ID
    camera_id: str           # Which camera
    timestamp: float         # time.time() when event fired
    bbox: Tuple[float, ...]  # (x, y, w, h) at event time
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = round(d["timestamp"], 3)
        return d


# ── Per-person state in one camera ───────────────────────────────────────────

class _PersonCamState:
    """Tracks a single (global_id, camera_id) pair for event logic."""

    __slots__ = [
        "global_id", "camera_id",
        "first_seen", "last_seen",
        "entry_emitted", "exit_emitted", "dwell_emitted",
        "prev_center", "curr_center",
        "line_crossings_emitted",
        "last_bbox",
    ]

    def __init__(
        self,
        global_id: int,
        camera_id: str,
        bbox_center: Tuple[float, float],
        bbox: Tuple[float, ...],
        timestamp: float,
    ):
        self.global_id = global_id
        self.camera_id = camera_id
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.entry_emitted = False
        self.exit_emitted = False
        self.dwell_emitted = False
        self.prev_center = bbox_center
        self.curr_center = bbox_center
        self.line_crossings_emitted: Set[str] = set()
        self.last_bbox = bbox


# ── Line crossing helper ─────────────────────────────────────────────────────

def _cross_product_sign(
    ax: float, ay: float, bx: float, by: float,
    px: float, py: float,
) -> float:
    """Sign of cross product (B-A) × (P-A).  >0 left, <0 right, 0 on line."""
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def _segments_intersect(
    p1: Tuple[float, float], p2: Tuple[float, float],
    q1: Tuple[float, float], q2: Tuple[float, float],
) -> bool:
    """True if line segment p1-p2 intersects line segment q1-q2."""
    d1 = _cross_product_sign(q1[0], q1[1], q2[0], q2[1], p1[0], p1[1])
    d2 = _cross_product_sign(q1[0], q1[1], q2[0], q2[1], p2[0], p2[1])
    d3 = _cross_product_sign(p1[0], p1[1], p2[0], p2[1], q1[0], q1[1])
    d4 = _cross_product_sign(p1[0], p1[1], p2[0], p2[1], q2[0], q2[1])

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _crossing_direction(
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
    person_prev: Tuple[float, float],
    person_curr: Tuple[float, float],
) -> str:
    """Determine crossing direction: 'in' (right→left of line) or 'out'."""
    sign_prev = _cross_product_sign(
        line_start[0], line_start[1],
        line_end[0], line_end[1],
        person_prev[0], person_prev[1],
    )
    return "in" if sign_prev > 0 else "out"


# ── Main event detector ─────────────────────────────────────────────────────

class EventDetector:
    """Detects entry/exit, dwell, and line-crossing events.

    Parameters (from ``cfg["events"]``):
        entry_exit.enabled : bool (default True)
        entry_exit.zones   : list of polygons ([] = full frame edge)
        dwell.enabled      : bool (default True)
        dwell.threshold_sec: float (default 10)
        line_crossing.enabled : bool (default True)
        line_crossing.lines   : list of line definitions
        exit_timeout_sec   : float — seconds of absence before firing exit (default 3)
    """

    def __init__(self, cfg: dict):
        ee_cfg = cfg.get("entry_exit", {})
        dw_cfg = cfg.get("dwell", {})
        lc_cfg = cfg.get("line_crossing", {})

        self.entry_exit_enabled = ee_cfg.get("enabled", True)
        self.dwell_enabled = dw_cfg.get("enabled", True)
        self.dwell_threshold = float(dw_cfg.get("threshold_sec", 10))
        self.line_crossing_enabled = lc_cfg.get("enabled", True)

        # Parse line definitions: [{name, x1, y1, x2, y2, cam_id?}]
        raw_lines = lc_cfg.get("lines", [])
        self.lines: List[Dict[str, Any]] = []
        for ln in raw_lines:
            self.lines.append({
                "name": ln.get("name", f"line_{len(self.lines)}"),
                "p1": (float(ln["x1"]), float(ln["y1"])),
                "p2": (float(ln["x2"]), float(ln["y2"])),
                "cam_id": ln.get("cam_id"),   # None = all cameras
            })

        self.exit_timeout = float(cfg.get("exit_timeout_sec", 3.0))

        # State: (cam_id, global_id) → _PersonCamState
        self._states: Dict[Tuple[str, int], _PersonCamState] = {}
        self._lock = threading.Lock()

        # Event log (bounded ring buffer)
        self._event_log: deque = deque(maxlen=2000)
        self._emitted_keys: Set[str] = set()   # dedup set

        # Counters
        self.entry_count = 0
        self.exit_count = 0
        self.dwell_count = 0
        self.line_cross_count = 0

        logger.info(
            "EventDetector — entry_exit=%s dwell=%s(%.0fs) "
            "line_crossing=%s(%d lines) exit_timeout=%.1fs",
            self.entry_exit_enabled, self.dwell_enabled, self.dwell_threshold,
            self.line_crossing_enabled, len(self.lines), self.exit_timeout,
        )

    def process(
        self,
        cam_id: str,
        tracks: List[Dict[str, Any]],
        gids: Dict[int, int],
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> List[Event]:
        """Process one camera's tracks and return new events.

        Parameters
        ----------
        cam_id : camera identifier
        tracks : [{'tid': int, 'bbox': (x,y,w,h), 'score': float}, ...]
        gids   : {local_tid: global_id}
        frame_shape : (height, width) for auto-generating default lines

        Returns
        -------
        List of newly fired events (no duplicates).
        """
        with self._lock:
            return self._process_impl(cam_id, tracks, gids, frame_shape)

    def get_recent_events(self, n: int = 50) -> List[dict]:
        """Return the N most recent events (newest first)."""
        with self._lock:
            return [e.to_dict() for e in list(self._event_log)[-n:]][::-1]

    def get_counts(self) -> dict:
        return {
            "entry": self.entry_count,
            "exit": self.exit_count,
            "dwell": self.dwell_count,
            "line_crossing": self.line_cross_count,
            "total": self.entry_count + self.exit_count +
                     self.dwell_count + self.line_cross_count,
        }

    # ── internal ─────────────────────────────────────────────────────────

    def _process_impl(
        self, cam_id, tracks, gids, frame_shape,
    ) -> List[Event]:
        now = time.time()
        new_events: List[Event] = []

        # Build set of currently visible (cam_id, global_id) pairs
        current_gids: Set[Tuple[str, int]] = set()

        for t in tracks:
            tid = t["tid"]
            gid = gids.get(tid)
            if gid is None:
                continue

            x, y, w, h = (float(v) for v in t["bbox"])
            cx, cy = x + w / 2, y + h / 2
            bbox = (x, y, w, h)
            key = (cam_id, gid)
            current_gids.add(key)

            if key in self._states:
                state = self._states[key]
                state.prev_center = state.curr_center
                state.curr_center = (cx, cy)
                state.last_seen = now
                state.last_bbox = bbox

                # If this person had exited and came back, reset for re-entry
                if state.exit_emitted:
                    state.entry_emitted = False
                    state.exit_emitted = False
                    state.dwell_emitted = False
                    state.first_seen = now
                    state.line_crossings_emitted.clear()
            else:
                state = _PersonCamState(gid, cam_id, (cx, cy), bbox, now)
                self._states[key] = state

            # ── 1. Entry event ──
            if self.entry_exit_enabled and not state.entry_emitted:
                evt = self._emit_event(
                    "entry", gid, cam_id, now, bbox,
                    metadata={"direction": "in"},
                )
                if evt:
                    new_events.append(evt)
                    self.entry_count += 1
                state.entry_emitted = True

            # ── 2. Dwell event ──
            if self.dwell_enabled and not state.dwell_emitted:
                dwell_sec = now - state.first_seen
                if dwell_sec >= self.dwell_threshold:
                    evt = self._emit_event(
                        "dwell", gid, cam_id, now, bbox,
                        metadata={"dwell_seconds": round(dwell_sec, 1)},
                    )
                    if evt:
                        new_events.append(evt)
                        self.dwell_count += 1
                    state.dwell_emitted = True

            # ── 3. Line crossing ──
            if self.line_crossing_enabled:
                # Use configured lines + auto-generated default lines
                lines_to_check = list(self.lines)

                # Auto-generate a default horizontal line at 60% height
                # if no lines are configured for this camera
                cam_has_line = any(
                    ln["cam_id"] is None or ln["cam_id"] == cam_id
                    for ln in self.lines
                )
                if not cam_has_line and frame_shape is not None:
                    fh, fw = frame_shape
                    lines_to_check.append({
                        "name": f"{cam_id}_auto_h",
                        "p1": (0, fh * 0.6),
                        "p2": (fw, fh * 0.6),
                        "cam_id": cam_id,
                    })

                for ln in lines_to_check:
                    if ln["cam_id"] is not None and ln["cam_id"] != cam_id:
                        continue

                    line_name = ln["name"]
                    if line_name in state.line_crossings_emitted:
                        continue   # already fired for this crossing

                    crossed = _segments_intersect(
                        state.prev_center, state.curr_center,
                        ln["p1"], ln["p2"],
                    )
                    if crossed:
                        direction = _crossing_direction(
                            ln["p1"], ln["p2"],
                            state.prev_center, state.curr_center,
                        )
                        evt = self._emit_event(
                            "line_crossing", gid, cam_id, now, bbox,
                            metadata={
                                "line": line_name,
                                "direction": direction,
                            },
                        )
                        if evt:
                            new_events.append(evt)
                            self.line_cross_count += 1
                        state.line_crossings_emitted.add(line_name)

        # ── 4. Exit events — persons no longer visible ──
        if self.entry_exit_enabled:
            for key, state in list(self._states.items()):
                if key[0] != cam_id:
                    continue
                if key not in current_gids:
                    elapsed = now - state.last_seen
                    if elapsed >= self.exit_timeout and not state.exit_emitted:
                        evt = self._emit_event(
                            "exit", state.global_id, cam_id, now, state.last_bbox,
                            metadata={
                                "direction": "out",
                                "total_dwell_seconds": round(
                                    state.last_seen - state.first_seen, 1
                                ),
                            },
                        )
                        if evt:
                            new_events.append(evt)
                            self.exit_count += 1
                        state.exit_emitted = True

            # Garbage collect very old states (> 5 minutes since exit)
            stale_keys = [
                k for k, s in self._states.items()
                if k[0] == cam_id and s.exit_emitted and (now - s.last_seen) > 300
            ]
            for k in stale_keys:
                del self._states[k]

        return new_events

    def _emit_event(
        self,
        event_type: str,
        global_id: int,
        camera_id: str,
        timestamp: float,
        bbox: Tuple[float, ...],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Event]:
        """Create event if not already emitted (dedup)."""
        # Dedup key: type + gid + cam + a time bucket (1s resolution) for
        # entry/exit, or line_name for crossings
        if event_type == "line_crossing":
            dedup_suffix = metadata.get("line", "") if metadata else ""
        elif event_type == "dwell":
            dedup_suffix = "once"
        else:
            dedup_suffix = str(int(timestamp))

        dedup_key = f"{event_type}:{global_id}:{camera_id}:{dedup_suffix}"

        if dedup_key in self._emitted_keys:
            return None

        event_id = hashlib.sha1(dedup_key.encode()).hexdigest()[:12]
        evt = Event(
            event_id=event_id,
            event_type=event_type,
            global_id=global_id,
            camera_id=camera_id,
            timestamp=timestamp,
            bbox=bbox,
            metadata=metadata or {},
        )
        self._event_log.append(evt)
        self._emitted_keys.add(dedup_key)

        # Bound the dedup set (keep last 10000)
        if len(self._emitted_keys) > 10000:
            # Just clear old ones — the state-based flags prevent true
            # duplicates anyway
            self._emitted_keys.clear()

        logger.debug(
            "EVENT %s: G%d @ %s — %s",
            event_type.upper(), global_id, camera_id, metadata,
        )
        return evt
