"""
Async multi-camera ingest with bounded queues, auto-reconnect, and frame-drop.

Each camera runs in its own daemon thread.  A bounded queue (default size 2)
ensures that only the *freshest* frames are ever consumed — stale frames are
silently dropped so the pipeline never falls behind.

Supports:  RTSP, HTTP, MP4/AVI file paths, and integer webcam indices.
"""

from __future__ import annotations

import time
import threading
import logging
from queue import Queue, Full, Empty
from typing import Dict, Optional, List

import cv2

logger = logging.getLogger(__name__)


# ── single-camera worker ────────────────────────────────────────────────────

class CameraWorker:
    """Threaded reader for a single camera / video source."""

    def __init__(
        self,
        camera_id: str,
        path: str,
        queue_size: int = 2,
        target_fps: float = 15.0,
        reconnect_delay: float = 3.0,
        max_reconnect: int = 10,
        resize_width: Optional[int] = None,
    ):
        self.camera_id = camera_id
        self.path = path
        self.queue: Queue = Queue(maxsize=queue_size)
        self.target_fps = target_fps
        self.reconnect_delay = reconnect_delay
        self.max_reconnect = max_reconnect
        self.resize_width = resize_width

        self._cap: Optional[cv2.VideoCapture] = None
        self._source_fps: float = 25.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._reconnect_count = 0
        self._frames_read = 0
        self._frames_dropped = 0
        self._last_ts = 0.0

    # ── public API ──

    def start(self):
        """Open source and start the reader thread."""
        self._open()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"cam-{self.camera_id}", daemon=True
        )
        self._thread.start()
        logger.info("CameraWorker %s started (source fps=%.1f, target fps=%.1f)",
                     self.camera_id, self._source_fps, self.target_fps)

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._release()
        logger.info("CameraWorker %s stopped  (read=%d, dropped=%d)",
                     self.camera_id, self._frames_read, self._frames_dropped)

    def get(self, timeout: float = 0.5) -> Optional[object]:
        """Return the latest frame (numpy array) or None if queue is empty."""
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None

    @property
    def source_fps(self) -> float:
        return self._source_fps

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── internals ──

    def _open(self):
        """Open (or reopen) the video source."""
        self._release()
        src = self.path
        if src.isdigit():
            src = int(src)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            logger.warning("CameraWorker %s: cannot open %s", self.camera_id, self.path)
            self._cap = None
            return False
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._source_fps = fps if fps and fps > 0 else 25.0
        self._cap = cap
        self._reconnect_count = 0
        return True

    def _release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _reconnect(self) -> bool:
        """Try to reopen the source with back-off."""
        while not self._stop_event.is_set() and self._reconnect_count < self.max_reconnect:
            self._reconnect_count += 1
            logger.warning("CameraWorker %s: reconnect attempt %d/%d …",
                           self.camera_id, self._reconnect_count, self.max_reconnect)
            time.sleep(self.reconnect_delay)
            if self._open():
                logger.info("CameraWorker %s: reconnected.", self.camera_id)
                return True
        logger.error("CameraWorker %s: gave up reconnecting.", self.camera_id)
        return False

    def _run(self):
        """Main loop — reads frames, rate-limits, drops stale."""
        frame_interval = 1.0 / max(1, self.target_fps)
        # how many source frames to skip per target frame
        skip = max(1, int(round(self._source_fps / self.target_fps)))

        frame_counter = 0
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                if not self._reconnect():
                    break
                skip = max(1, int(round(self._source_fps / self.target_fps)))
                continue

            ok, frame = self._cap.read()
            if not ok:
                # end-of-file for video — loop back to start (simulate live)
                if not str(self.path).startswith("rtsp"):
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    logger.info("CameraWorker %s: looping video.", self.camera_id)
                    continue
                # RTSP / real stream → try reconnect
                if not self._reconnect():
                    break
                continue

            frame_counter += 1
            if frame_counter % skip != 0:
                continue  # skip to stay at target_fps

            # optional resize
            if self.resize_width and frame.shape[1] > self.resize_width:
                h, w = frame.shape[:2]
                new_h = int(h * self.resize_width / w)
                frame = cv2.resize(frame, (self.resize_width, new_h))

            # bounded put — drop stale if queue full
            self._frames_read += 1
            try:
                self.queue.put_nowait(frame)
            except Full:
                # drop oldest, put new
                try:
                    self.queue.get_nowait()
                except Empty:
                    pass
                try:
                    self.queue.put_nowait(frame)
                except Full:
                    pass
                self._frames_dropped += 1

            # rate-limit to avoid busy-spin
            now = time.monotonic()
            elapsed = now - self._last_ts
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            self._last_ts = time.monotonic()


# ── multi-camera reader ─────────────────────────────────────────────────────

class MultiVideoReader:
    """Drop-in replacement for the original synchronous reader.

    Spawns one ``CameraWorker`` thread per source.  ``read()`` returns
    the latest frame from each camera (non-blocking, may be None for
    cameras that haven't produced a new frame yet).
    """

    def __init__(self, inputs: List[dict], ingest_cfg: Optional[dict] = None):
        ingest_cfg = ingest_cfg or {}
        self.workers: Dict[str, CameraWorker] = {}
        for item in inputs:
            cam_id = item["camera_id"]
            path = str(item["path"])
            w = CameraWorker(
                camera_id=cam_id,
                path=path,
                queue_size=int(ingest_cfg.get("queue_size", 2)),
                target_fps=float(ingest_cfg.get("target_fps", 15)),
                reconnect_delay=float(ingest_cfg.get("reconnect_delay", 3)),
                max_reconnect=int(ingest_cfg.get("max_reconnect", 10)),
                resize_width=ingest_cfg.get("resize_width"),
            )
            self.workers[cam_id] = w

        # start all workers
        for w in self.workers.values():
            w.start()

    def read(self) -> Optional[Dict[str, object]]:
        """Return dict {cam_id: frame} with latest frames.

        Returns None only when *all* workers have died.
        """
        frames: Dict[str, object] = {}
        any_alive = False
        for cam_id, w in self.workers.items():
            if w.alive:
                any_alive = True
            f = w.get(timeout=0.05)
            if f is not None:
                frames[cam_id] = f
        if not any_alive:
            return None
        return frames if frames else {}

    def fps(self, cam_id: str) -> float:
        w = self.workers.get(cam_id)
        return w.source_fps if w else 25.0

    def close(self):
        for w in self.workers.values():
            w.stop()
