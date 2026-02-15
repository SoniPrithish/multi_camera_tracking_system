"""
Deep Re-Identification module with:
  - OSNet / MobileNet feature extraction (512-d)
  - EMA (exponential moving average) embeddings per track
  - Time-windowed gallery with auto-pruning
  - Cosine similarity + Hungarian matching for global ID assignment
"""

from __future__ import annotations

import time
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from .osnet import OSNetExtractor

logger = logging.getLogger(__name__)


class GalleryEntry:
    """A gallery entry for one global identity."""

    __slots__ = ["global_id", "embedding", "last_seen", "camera_id", "hits"]

    def __init__(self, global_id: int, embedding: np.ndarray, camera_id: str):
        self.global_id = global_id
        self.embedding = embedding.copy()
        self.last_seen = time.monotonic()
        self.camera_id = camera_id
        self.hits = 1

    def update_embedding(self, new_emb: np.ndarray, alpha: float = 0.7):
        """EMA update of the gallery embedding."""
        self.embedding = alpha * self.embedding + (1 - alpha) * new_emb
        # Re-normalize
        norm = np.linalg.norm(self.embedding)
        if norm > 0:
            self.embedding = self.embedding / norm
        self.last_seen = time.monotonic()
        self.hits += 1


class DeepReID:
    """Full deep ReID module with gallery and global ID assignment.

    Parameters
    ----------
    model : str
        Feature extractor model name (osnet_x0_25, mobilenet, resnet18).
    device : str
        "0" for GPU, "cpu" for CPU.
    sim_thresh : float
        Minimum cosine similarity to match a gallery entry.
    ema_alpha : float
        EMA smoothing factor for embedding updates (higher = more weight on old).
    gallery_ttl : float
        Time-to-live in seconds for gallery entries (auto-prune stale entries).
    max_gallery_size : int
        Maximum gallery entries (prevents unbounded memory).
    """

    def __init__(
        self,
        model: str = "osnet_x0_25",
        device: str = "0",
        sim_thresh: float = 0.45,
        ema_alpha: float = 0.7,
        gallery_ttl: float = 120.0,
        max_gallery_size: int = 500,
    ):
        self.extractor = OSNetExtractor(model_name=model, device=device)
        self.sim_thresh = sim_thresh
        self.ema_alpha = ema_alpha
        self.gallery_ttl = gallery_ttl
        self.max_gallery_size = max_gallery_size

        # Gallery: global_id -> GalleryEntry
        self._gallery: Dict[int, GalleryEntry] = {}
        self._next_gid = 1
        self._lock = threading.Lock()

        # Per-track EMA embeddings: (cam_id, local_tid) -> np.ndarray
        self._track_embs: Dict[Tuple[str, int], np.ndarray] = {}

        self.embed_dim = self.extractor.embed_dim
        logger.info("DeepReID initialized — sim_thresh=%.2f, ema=%.2f, ttl=%.0fs, max=%d",
                     sim_thresh, ema_alpha, gallery_ttl, max_gallery_size)

    def encode(self, frame: np.ndarray, tracks: List[Dict[str, Any]]) -> Dict[int, np.ndarray]:
        """Extract + EMA-smooth embeddings for all tracks.

        Parameters
        ----------
        frame : BGR image
        tracks : list of {'tid': int, 'bbox': (x,y,w,h), 'score': float}

        Returns
        -------
        embs : dict {tid: 512-d np.ndarray}
        """
        if not tracks:
            return {}

        # Crop all persons
        crops = []
        valid_indices = []
        for i, t in enumerate(tracks):
            x, y, w, h = map(int, t["bbox"])
            fh, fw = frame.shape[:2]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(fw, x + w), min(fh, y + h)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0 and crop.shape[0] >= 10 and crop.shape[1] >= 10:
                crops.append(crop)
                valid_indices.append(i)

        if not crops:
            return {t["tid"]: np.zeros(self.embed_dim, dtype=np.float32) for t in tracks}

        # Batch extract
        raw_embs = self.extractor.extract(crops)

        # Map back and apply EMA smoothing per track
        result = {}
        emb_idx = 0
        for i, t in enumerate(tracks):
            tid = t["tid"]
            if i in valid_indices:
                raw = raw_embs[emb_idx]
                emb_idx += 1
            else:
                raw = np.zeros(self.embed_dim, dtype=np.float32)

            # EMA smooth with previous embedding for this track
            key = tid  # simplified key
            if key in self._track_embs and np.any(self._track_embs[key]):
                smoothed = self.ema_alpha * self._track_embs[key] + (1 - self.ema_alpha) * raw
                norm = np.linalg.norm(smoothed)
                if norm > 0:
                    smoothed = smoothed / norm
            else:
                smoothed = raw

            self._track_embs[key] = smoothed
            result[tid] = smoothed

        return result

    def assign_global_ids(
        self, cam_id: str, tracks: List[Dict[str, Any]], embs: Dict[int, np.ndarray]
    ) -> Dict[int, int]:
        """Assign global IDs by matching track embeddings against the gallery.

        Uses cosine similarity + Hungarian matching.

        Returns
        -------
        gids : dict {local_tid: global_id}
        """
        with self._lock:
            # Prune stale gallery entries
            self._prune_gallery()

            if not tracks:
                return {}

            # Get valid embeddings
            valid_tracks = []
            valid_embs = []
            for t in tracks:
                e = embs.get(t["tid"])
                if e is not None and np.any(e):
                    valid_tracks.append(t)
                    valid_embs.append(e)

            if not valid_embs:
                # All tracks have zero embeddings → assign new global IDs
                gids = {}
                for t in tracks:
                    gid = self._next_gid
                    self._next_gid += 1
                    gids[t["tid"]] = gid
                return gids

            query_embs = np.array(valid_embs, dtype=np.float32)  # (M, D)

            # Build gallery matrix
            gallery_ids = list(self._gallery.keys())
            if gallery_ids:
                gallery_embs = np.array(
                    [self._gallery[gid].embedding for gid in gallery_ids],
                    dtype=np.float32,
                )  # (G, D)

                # Cosine similarity matrix (M, G)
                sim_matrix = query_embs @ gallery_embs.T

                # Hungarian matching on cost = 1 - sim
                cost_matrix = 1.0 - sim_matrix
                row_idx, col_idx = linear_sum_assignment(cost_matrix)

                matched_query = set()
                matched_gallery = set()
                gids = {}

                for r, c in zip(row_idx, col_idx):
                    sim = sim_matrix[r, c]
                    if sim >= self.sim_thresh:
                        gid = gallery_ids[c]
                        tid = valid_tracks[r]["tid"]
                        gids[tid] = gid
                        # EMA update gallery
                        self._gallery[gid].update_embedding(valid_embs[r], self.ema_alpha)
                        self._gallery[gid].camera_id = cam_id
                        matched_query.add(r)
                        matched_gallery.add(c)

                # Unmatched queries → new global IDs
                for i, t in enumerate(valid_tracks):
                    if i not in matched_query:
                        gid = self._next_gid
                        self._next_gid += 1
                        gids[t["tid"]] = gid
                        self._gallery[gid] = GalleryEntry(gid, valid_embs[i], cam_id)

            else:
                # Empty gallery — initialize all
                gids = {}
                for i, t in enumerate(valid_tracks):
                    gid = self._next_gid
                    self._next_gid += 1
                    gids[t["tid"]] = gid
                    self._gallery[gid] = GalleryEntry(gid, valid_embs[i], cam_id)

            # Handle tracks that had zero embeddings (not in valid_tracks)
            valid_tids = {t["tid"] for t in valid_tracks}
            for t in tracks:
                if t["tid"] not in valid_tids:
                    gid = self._next_gid
                    self._next_gid += 1
                    gids[t["tid"]] = gid

            return gids

    def _prune_gallery(self):
        """Remove stale gallery entries older than gallery_ttl."""
        now = time.monotonic()
        stale = [
            gid for gid, entry in self._gallery.items()
            if now - entry.last_seen > self.gallery_ttl
        ]
        for gid in stale:
            del self._gallery[gid]

        # If still too large, remove oldest
        if len(self._gallery) > self.max_gallery_size:
            sorted_entries = sorted(self._gallery.items(), key=lambda x: x[1].last_seen)
            to_remove = len(self._gallery) - self.max_gallery_size
            for gid, _ in sorted_entries[:to_remove]:
                del self._gallery[gid]

    @property
    def gallery_size(self) -> int:
        return len(self._gallery)
