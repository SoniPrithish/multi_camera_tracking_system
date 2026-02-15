"""
ByteTrack: Multi-Object Tracking by Associating Every Detection Box.

Two-stage association:
  1. High-confidence detections matched to existing tracks via IoU + Kalman.
  2. Low-confidence detections matched to remaining unmatched tracks.

This gives stable IDs through occlusions and crowded scenes.

Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
Every Detection Box", ECCV 2022.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman_filter import KalmanFilter

logger = logging.getLogger(__name__)

# ── Track states ──
class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


# ── Helper functions ──

def xywh_to_xyxy(bbox):
    """(x, y, w, h) → (x1, y1, x2, y2)."""
    x, y, w, h = bbox
    return np.array([x, y, x + w, y + h], dtype=np.float64)


def xyxy_to_xywh(bbox):
    """(x1, y1, x2, y2) → (x, y, w, h)."""
    x1, y1, x2, y2 = bbox
    return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float64)


def xywh_to_cxcyah(bbox):
    """(x, y, w, h) → (cx, cy, aspect_ratio, height) for Kalman filter."""
    x, y, w, h = bbox
    return np.array([x + w / 2, y + h / 2, w / max(h, 1e-6), h], dtype=np.float64)


def cxcyah_to_xywh(state):
    """(cx, cy, a, h) → (x, y, w, h)."""
    cx, cy, a, h = state
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, w, h], dtype=np.float64)


def iou_batch(bboxes_a: np.ndarray, bboxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of xyxy bboxes.

    Parameters
    ----------
    bboxes_a : (M, 4) in xyxy format
    bboxes_b : (N, 4) in xyxy format

    Returns
    -------
    iou_matrix : (M, N)
    """
    M, N = len(bboxes_a), len(bboxes_b)
    if M == 0 or N == 0:
        return np.zeros((M, N), dtype=np.float64)

    a = bboxes_a[:, None, :]  # (M, 1, 4)
    b = bboxes_b[None, :, :]  # (1, N, 4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter = np.maximum(0, inter_x2 - inter_x1) * np.maximum(0, inter_y2 - inter_y1)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter

    return inter / np.maximum(union, 1e-6)


def linear_assignment(cost_matrix: np.ndarray, thresh: float):
    """Solve the linear assignment problem.

    Returns
    -------
    matches : list of (row, col) tuples
    unmatched_a : list of row indices
    unmatched_b : list of col indices
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_a = list(range(cost_matrix.shape[0]))
    unmatched_b = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] > thresh:
            continue
        matches.append((r, c))
        if r in unmatched_a:
            unmatched_a.remove(r)
        if c in unmatched_b:
            unmatched_b.remove(c)

    return matches, unmatched_a, unmatched_b


# ── Single track ──

class STrack:
    """A single object track with Kalman filter state."""

    _count = 0  # class-level ID counter

    @staticmethod
    def next_id():
        STrack._count += 1
        return STrack._count

    @staticmethod
    def reset_id():
        STrack._count = 0

    def __init__(self, bbox_xywh, score: float, cls: int = 0):
        self.track_id = 0  # assigned when activated
        self.bbox = np.array(bbox_xywh, dtype=np.float64)  # (x, y, w, h)
        self.score = score
        self.cls = cls

        self.kalman_filter = None
        self.mean = None
        self.covariance = None

        self.state = TrackState.New
        self.is_activated = False

        self.frame_id = 0
        self.start_frame = 0
        self.tracklet_len = 0

    def activate(self, kalman_filter: KalmanFilter, frame_id: int):
        """Initialize a new track."""
        self.kalman_filter = kalman_filter
        self.track_id = STrack.next_id()

        measurement = xywh_to_cxcyah(self.bbox)
        self.mean, self.covariance = self.kalman_filter.initiate(measurement)

        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 0

    def re_activate(self, new_track: "STrack", frame_id: int, new_id: bool = False):
        """Re-activate a lost track with a new detection."""
        measurement = xywh_to_cxcyah(new_track.bbox)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, measurement
        )
        self.bbox = new_track.bbox.copy()
        self.score = new_track.score

        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.tracklet_len = 0
        if new_id:
            self.track_id = STrack.next_id()

    def predict(self):
        """Kalman filter prediction step."""
        if self.state != TrackState.Tracked:
            self.mean[7] = 0  # zero velocity for lost tracks
        self.mean, self.covariance = self.kalman_filter.predict(
            self.mean, self.covariance
        )

    def update(self, new_track: "STrack", frame_id: int):
        """Kalman filter update step with a matched detection."""
        measurement = xywh_to_cxcyah(new_track.bbox)
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, measurement
        )
        self.bbox = new_track.bbox.copy()
        self.score = new_track.score

        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.tracklet_len += 1

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed

    @property
    def predicted_bbox(self) -> np.ndarray:
        """Get the predicted xywh bbox from the Kalman state."""
        if self.mean is None:
            return self.bbox
        return cxcyah_to_xywh(self.mean[:4])

    @property
    def xyxy(self) -> np.ndarray:
        return xywh_to_xyxy(self.bbox)

    @property
    def predicted_xyxy(self) -> np.ndarray:
        return xywh_to_xyxy(self.predicted_bbox)


# ── ByteTrack multi-camera wrapper ──

class ByteTracker:
    """Full ByteTrack implementation with per-camera state.

    Parameters (all from config):
        max_age        : frames to keep a lost track before removing (default 30)
        min_hits       : minimum hits to output a track (default 3)
        iou_thresh     : IoU threshold for first association (default 0.2)
        high_thresh    : confidence threshold for "high" detections (default 0.5)
        low_thresh     : confidence threshold for "low" detections (default 0.1)
        match_thresh   : cost threshold for assignment (default 0.8)
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_thresh: float = 0.2,
        high_thresh: float = 0.5,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh

        # Per-camera state
        self._cam_state: Dict[str, _CamState] = {}

    def update(self, cam_id: str, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run one tracking step for a camera.

        Parameters
        ----------
        cam_id : str
        detections : list of {'bbox': (x,y,w,h), 'score': float, 'cls': int}

        Returns
        -------
        tracks : list of {'tid': int, 'bbox': (x,y,w,h), 'score': float}
        """
        if cam_id not in self._cam_state:
            self._cam_state[cam_id] = _CamState(
                max_age=self.max_age,
                min_hits=self.min_hits,
                iou_thresh=self.iou_thresh,
                high_thresh=self.high_thresh,
                low_thresh=self.low_thresh,
                match_thresh=self.match_thresh,
            )
        return self._cam_state[cam_id].update(detections)


class _CamState:
    """Tracking state for a single camera (internal)."""

    def __init__(self, max_age, min_hits, iou_thresh, high_thresh, low_thresh, match_thresh):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_thresh = iou_thresh
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh

        self.kalman_filter = KalmanFilter()
        self.frame_id = 0

        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.frame_id += 1

        # ── Step 0: Create STrack objects from detections ──
        det_stracks = []
        for d in detections:
            s = STrack(d["bbox"], d["score"], d.get("cls", 0))
            det_stracks.append(s)

        # Split detections into high-confidence and low-confidence
        high_dets = [s for s in det_stracks if s.score >= self.high_thresh]
        low_dets = [s for s in det_stracks if self.low_thresh <= s.score < self.high_thresh]

        # ── Step 1: Predict existing tracks ──
        # Combine currently tracked + recently lost
        strack_pool = self.tracked_stracks + self.lost_stracks
        for t in strack_pool:
            t.predict()

        # ── Step 2: First association — high-conf dets vs all pool tracks ──
        if len(strack_pool) > 0 and len(high_dets) > 0:
            track_xyxy = np.array([t.predicted_xyxy for t in strack_pool])
            det_xyxy = np.array([xywh_to_xyxy(d.bbox) for d in high_dets])
            iou_matrix = iou_batch(track_xyxy, det_xyxy)
            cost_matrix = 1.0 - iou_matrix

            matches, unmatched_tracks, unmatched_dets = linear_assignment(
                cost_matrix, thresh=self.match_thresh
            )
        else:
            matches = []
            unmatched_tracks = list(range(len(strack_pool)))
            unmatched_dets = list(range(len(high_dets)))

        # Apply first-round matches
        for t_idx, d_idx in matches:
            track = strack_pool[t_idx]
            det = high_dets[d_idx]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
            else:
                track.re_activate(det, self.frame_id, new_id=False)

        # Remaining tracked (not lost) tracks for second association
        remaining_tracked = [
            strack_pool[i] for i in unmatched_tracks
            if strack_pool[i].state == TrackState.Tracked
        ]
        # Remaining lost tracks go directly to lost pool
        remaining_lost_from_first = [
            strack_pool[i] for i in unmatched_tracks
            if strack_pool[i].state != TrackState.Tracked
        ]

        # ── Step 3: Second association — low-conf dets vs remaining tracked ──
        if len(remaining_tracked) > 0 and len(low_dets) > 0:
            track_xyxy = np.array([t.predicted_xyxy for t in remaining_tracked])
            det_xyxy = np.array([xywh_to_xyxy(d.bbox) for d in low_dets])
            iou_matrix = iou_batch(track_xyxy, det_xyxy)
            cost_matrix = 1.0 - iou_matrix

            matches2, unmatched_tracks2, _ = linear_assignment(
                cost_matrix, thresh=0.5  # more lenient for low-conf
            )
        else:
            matches2 = []
            unmatched_tracks2 = list(range(len(remaining_tracked)))

        for t_idx, d_idx in matches2:
            track = remaining_tracked[t_idx]
            det = low_dets[d_idx]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
            else:
                track.re_activate(det, self.frame_id, new_id=False)

        # Tracks unmatched after both rounds → mark lost
        for i in unmatched_tracks2:
            track = remaining_tracked[i]
            if track.state == TrackState.Tracked:
                track.mark_lost()

        # Also mark lost tracks that weren't re-activated
        for t in remaining_lost_from_first:
            # They stay lost (already)
            pass

        # ── Step 4: Initialize new tracks from unmatched high-conf dets ──
        for i in unmatched_dets:
            det = high_dets[i]
            det.activate(self.kalman_filter, self.frame_id)

        # ── Step 5: Update track lists ──
        new_tracked = []
        new_lost = []

        for t in self.tracked_stracks:
            if t.state == TrackState.Tracked:
                new_tracked.append(t)
            elif t.state == TrackState.Lost:
                new_lost.append(t)

        # Add re-activated tracks from pool
        for t in self.lost_stracks:
            if t.state == TrackState.Tracked:
                new_tracked.append(t)
            elif t.state == TrackState.Lost:
                new_lost.append(t)

        # Add newly initialized tracks
        for i in unmatched_dets:
            t = high_dets[i]
            if t.is_activated:
                new_tracked.append(t)

        # Remove long-lost tracks
        for t in new_lost:
            if self.frame_id - t.frame_id > self.max_age:
                t.mark_removed()

        self.tracked_stracks = [t for t in new_tracked if t.state == TrackState.Tracked]
        self.lost_stracks = [t for t in new_lost if t.state == TrackState.Lost]

        # Deduplicate by track_id
        seen_ids = set()
        deduped = []
        for t in self.tracked_stracks:
            if t.track_id not in seen_ids:
                seen_ids.add(t.track_id)
                deduped.append(t)
        self.tracked_stracks = deduped

        # ── Step 6: Output active tracks ──
        outputs = []
        for t in self.tracked_stracks:
            if t.is_activated and t.tracklet_len >= self.min_hits or self.frame_id <= self.min_hits + 1:
                outputs.append({
                    "tid": t.track_id,
                    "bbox": tuple(map(float, t.bbox)),
                    "score": float(t.score),
                })

        return outputs
