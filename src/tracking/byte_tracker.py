"""
ByteTrack implementation for multi-object tracking.
Based on: ByteTrack: Multi-Object Tracking by Associating Every Detection Box
Paper: https://arxiv.org/abs/2110.06864
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from enum import Enum
from collections import defaultdict

from .kalman import KalmanFilter, bbox_to_xyah, xyah_to_bbox


class TrackState(Enum):
    """Track lifecycle states."""
    TENTATIVE = 1  # Newly created, not yet confirmed
    CONFIRMED = 2  # Confirmed track
    LOST = 3       # Lost track (not matched for some frames)


class STrack:
    """Single object track."""
    
    _count = 0
    shared_kalman = KalmanFilter()
    
    def __init__(self, bbox: Tuple[float, float, float, float], score: float):
        """
        Initialize track.
        
        Args:
            bbox: (x, y, w, h) bounding box
            score: Detection confidence score
        """
        self.track_id = 0  # Will be assigned when activated
        self._bbox = bbox
        self.score = score
        self.state = TrackState.TENTATIVE
        
        # Kalman filter state
        self.mean = None
        self.covariance = None
        
        # Track history
        self.frame_id = 0
        self.start_frame = 0
        self.time_since_update = 0
        self.hits = 0
        
        # Feature storage (for ReID)
        self.features = []
        self.smooth_feat = None
        self.curr_feat = None
        self.alpha = 0.9  # EMA smoothing factor
    
    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """Get current bounding box (x, y, w, h)."""
        if self.mean is not None:
            return xyah_to_bbox(self.mean[:4])
        return self._bbox
    
    @property
    def tlbr(self) -> np.ndarray:
        """Get bounding box as [x1, y1, x2, y2]."""
        x, y, w, h = self.bbox
        return np.array([x, y, x + w, y + h])
    
    def predict(self):
        """Propagate state using Kalman filter."""
        if self.mean is not None:
            self.mean, self.covariance = self.shared_kalman.predict(
                self.mean, self.covariance
            )
    
    def activate(self, frame_id: int):
        """Activate a new track."""
        STrack._count += 1
        self.track_id = STrack._count
        
        # Initialize Kalman filter
        self.mean, self.covariance = self.shared_kalman.initiate(
            bbox_to_xyah(self._bbox)
        )
        
        self.state = TrackState.TENTATIVE
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.hits = 1
    
    def re_activate(self, new_track: 'STrack', frame_id: int, new_id: bool = False):
        """Re-activate a lost track with new detection."""
        self.mean, self.covariance = self.shared_kalman.update(
            self.mean, self.covariance, bbox_to_xyah(new_track._bbox)
        )
        
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        
        self.state = TrackState.CONFIRMED
        self.frame_id = frame_id
        self.time_since_update = 0
        self.hits += 1
        self.score = new_track.score
        
        if new_id:
            STrack._count += 1
            self.track_id = STrack._count
    
    def update(self, new_track: 'STrack', frame_id: int):
        """Update track with matched detection."""
        self.mean, self.covariance = self.shared_kalman.update(
            self.mean, self.covariance, bbox_to_xyah(new_track._bbox)
        )
        
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        
        self.state = TrackState.CONFIRMED
        self.frame_id = frame_id
        self.time_since_update = 0
        self.hits += 1
        self.score = new_track.score
    
    def update_features(self, feat: np.ndarray):
        """Update feature with EMA smoothing."""
        feat = feat / (np.linalg.norm(feat) + 1e-6)
        self.curr_feat = feat
        
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
            self.smooth_feat = self.smooth_feat / (np.linalg.norm(self.smooth_feat) + 1e-6)
        
        self.features.append(feat)
        if len(self.features) > 100:
            self.features.pop(0)
    
    def mark_lost(self):
        """Mark track as lost."""
        self.state = TrackState.LOST
    
    def is_activated(self) -> bool:
        """Check if track is activated."""
        return self.state == TrackState.CONFIRMED
    
    @staticmethod
    def reset_id():
        """Reset global track ID counter."""
        STrack._count = 0


class ByteTracker:
    """
    ByteTrack multi-object tracker.
    
    Two-stage association:
    1. Match high-confidence detections with tracks using IoU
    2. Match remaining low-confidence detections with unmatched tracks
    """
    
    def __init__(
        self,
        track_high_thresh: float = 0.5,
        track_low_thresh: float = 0.1,
        new_track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        min_hits: int = 3,
    ):
        """
        Initialize ByteTracker.
        
        Args:
            track_high_thresh: Threshold for high-confidence detections
            track_low_thresh: Threshold for low-confidence detections
            new_track_thresh: Threshold for creating new tracks
            match_thresh: IoU threshold for matching
            track_buffer: Frames to keep lost tracks
            min_hits: Minimum hits to confirm track
        """
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.min_hits = min_hits
        
        # Per-camera track storage
        self.tracked_stracks: Dict[str, List[STrack]] = defaultdict(list)
        self.lost_stracks: Dict[str, List[STrack]] = defaultdict(list)
        self.removed_stracks: Dict[str, List[STrack]] = defaultdict(list)
        
        self.frame_id: Dict[str, int] = defaultdict(int)
    
    def update(self, cam_id: str, detections: List[Dict]) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            cam_id: Camera identifier
            detections: List of detections with 'bbox' and 'score' keys
            
        Returns:
            List of active tracks with 'tid', 'bbox', 'score' keys
        """
        self.frame_id[cam_id] += 1
        frame_id = self.frame_id[cam_id]
        
        # Get current tracks
        tracked_stracks = self.tracked_stracks[cam_id]
        lost_stracks = self.lost_stracks[cam_id]
        
        # Convert detections to STracks
        if len(detections) > 0:
            det_stracks = [
                STrack(d['bbox'], d.get('score', 1.0)) 
                for d in detections
            ]
            scores = np.array([d.get('score', 1.0) for d in detections])
        else:
            det_stracks = []
            scores = np.array([])
        
        # Split detections by confidence
        high_inds = scores >= self.track_high_thresh
        low_inds = (scores >= self.track_low_thresh) & (scores < self.track_high_thresh)
        
        det_high = [det_stracks[i] for i in np.where(high_inds)[0]]
        det_low = [det_stracks[i] for i in np.where(low_inds)[0]]
        
        # Predict all tracked tracks
        strack_pool = tracked_stracks + lost_stracks
        for track in strack_pool:
            track.predict()
        
        # === First association: high-confidence detections ===
        dists = self._iou_distance(strack_pool, det_high)
        matches, u_track, u_det_high = self._linear_assignment(
            dists, thresh=self.match_thresh
        )
        
        activated_stracks = []
        refind_stracks = []
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = det_high[idet]
            
            if track.state == TrackState.CONFIRMED:
                track.update(det, frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, frame_id)
                refind_stracks.append(track)
        
        # === Second association: low-confidence detections with remaining tracks ===
        r_tracked = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.CONFIRMED]
        
        dists = self._iou_distance(r_tracked, det_low)
        matches, u_track_2, u_det_low = self._linear_assignment(
            dists, thresh=0.5
        )
        
        for itracked, idet in matches:
            track = r_tracked[itracked]
            det = det_low[idet]
            track.update(det, frame_id)
            activated_stracks.append(track)
        
        # Mark unmatched tracks as lost
        for i in u_track_2:
            track = r_tracked[i]
            if track.state != TrackState.LOST:
                track.mark_lost()
                lost_stracks.append(track)
        
        # === Initialize new tracks ===
        for i in u_det_high:
            track = det_high[i]
            if track.score >= self.new_track_thresh:
                track.activate(frame_id)
                activated_stracks.append(track)
        
        # Remove old lost tracks
        lost_stracks = [t for t in lost_stracks 
                       if frame_id - t.frame_id <= self.track_buffer]
        
        # Update track lists
        self.tracked_stracks[cam_id] = [
            t for t in activated_stracks + refind_stracks 
            if t.state == TrackState.CONFIRMED
        ]
        self.lost_stracks[cam_id] = lost_stracks
        
        # Return confirmed tracks
        outputs = []
        for track in self.tracked_stracks[cam_id]:
            if track.is_activated() and track.hits >= self.min_hits:
                outputs.append({
                    'tid': track.track_id,
                    'bbox': track.bbox,
                    'score': track.score
                })
        
        return outputs
    
    def _iou_distance(
        self, 
        tracks: List[STrack], 
        detections: List[STrack]
    ) -> np.ndarray:
        """Compute IoU distance matrix."""
        if len(tracks) == 0 or len(detections) == 0:
            return np.zeros((len(tracks), len(detections)))
        
        track_boxes = np.array([t.tlbr for t in tracks])
        det_boxes = np.array([d.tlbr for d in detections])
        
        ious = self._batch_iou(track_boxes, det_boxes)
        return 1 - ious
    
    def _batch_iou(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU between two sets of boxes."""
        # boxes: [x1, y1, x2, y2]
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Intersection
        lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = np.clip(rb - lt, 0, None)
        inter = wh[:, :, 0] * wh[:, :, 1]
        
        # Union
        union = area1[:, None] + area2[None, :] - inter
        
        return inter / (union + 1e-6)
    
    def _linear_assignment(
        self, 
        cost_matrix: np.ndarray, 
        thresh: float
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Solve linear assignment problem.
        
        Returns:
            matches: List of (track_idx, det_idx) pairs
            unmatched_tracks: List of unmatched track indices
            unmatched_detections: List of unmatched detection indices
        """
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))
        
        try:
            import lap
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        except ImportError:
            # Fallback to scipy
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            x = np.full(cost_matrix.shape[0], -1)
            y = np.full(cost_matrix.shape[1], -1)
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= thresh:
                    x[r] = c
                    y[c] = r
        
        matches = []
        unmatched_tracks = []
        unmatched_dets = []
        
        for i, j in enumerate(x):
            if j >= 0:
                matches.append((i, j))
            else:
                unmatched_tracks.append(i)
        
        for j, i in enumerate(y):
            if i < 0:
                unmatched_dets.append(j)
        
        return matches, unmatched_tracks, unmatched_dets
    
    def reset(self, cam_id: Optional[str] = None):
        """Reset tracker state."""
        if cam_id is None:
            self.tracked_stracks.clear()
            self.lost_stracks.clear()
            self.removed_stracks.clear()
            self.frame_id.clear()
            STrack.reset_id()
        else:
            self.tracked_stracks[cam_id] = []
            self.lost_stracks[cam_id] = []
            self.removed_stracks[cam_id] = []
            self.frame_id[cam_id] = 0
