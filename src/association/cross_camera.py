"""
Cross-Camera Association — Stage 5
===================================

Assigns **consistent global IDs** to the same person seen across multiple
cameras.  Designed for HIGH PRECISION (fewer false matches).

Architecture
------------
1. **Camera graph** — all-to-all by default (configurable adjacency).
2. **Time-of-flight gating** — rejects matches where the elapsed time since
   the gallery entry was last seen is outside [min_transit, max_transit].
3. **Cosine similarity + Hungarian matching** — on L2-normalized 512-d
   embeddings from OSNet / MobileNet backbone.
4. **Global ID Registry** with lifecycle management:
   ACTIVE → INACTIVE (track lost) → PRUNED (expired TTL).
5. **Maturity gate** — a gallery entry must have ≥ `confirm_hits` appearances
   before it becomes a match candidate (prevents one-off noise).

Data flow (per pipeline loop)
-----------------------------
Phase 1  Continuing tracks (already have a global ID) → keep ID, EMA-update
         the gallery embedding.
Phase 2  New tracks → build cost matrix (appearance + temporal), Hungarian
         match against mature gallery entries, create new IDs for unmatched.
Phase 3  Cleanup disappeared tracks → mark INACTIVE → prune stale entries.

Thread-safe (single internal lock).
"""

from __future__ import annotations

import time
import logging
import threading
from typing import Dict, List, Set, Tuple, Any

import numpy as np
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# ── Global ID entry ──────────────────────────────────────────────────────────

class GlobalIDEntry:
    """A single global identity in the registry."""

    __slots__ = [
        "global_id", "embedding", "state",
        "last_camera", "last_seen", "first_seen",
        "last_bbox_center", "appearance_count",
        "camera_set",
    ]

    def __init__(
        self,
        global_id: int,
        embedding: np.ndarray,
        camera_id: str,
        bbox_center: Tuple[float, float],
    ):
        self.global_id = global_id
        self.embedding = embedding.copy().astype(np.float32)
        norm = np.linalg.norm(self.embedding)
        if norm > 1e-6:
            self.embedding /= norm

        now = time.monotonic()
        self.state: str = "active"       # active | inactive
        self.last_camera: str = camera_id
        self.last_seen: float = now
        self.first_seen: float = now
        self.last_bbox_center: Tuple[float, float] = bbox_center
        self.appearance_count: int = 1
        self.camera_set: Set[str] = {camera_id}

    # ── helpers ──

    def update_embedding(self, new_emb: np.ndarray, alpha: float):
        """EMA update: gallery ← α·gallery + (1-α)·new."""
        self.embedding = alpha * self.embedding + (1 - alpha) * new_emb
        norm = np.linalg.norm(self.embedding)
        if norm > 1e-6:
            self.embedding = self.embedding / norm

    def touch(self, camera_id: str, timestamp: float, bbox_center: Tuple[float, float]):
        """Update last-seen metadata."""
        self.last_camera = camera_id
        self.last_seen = timestamp
        self.last_bbox_center = bbox_center
        self.appearance_count += 1
        self.camera_set.add(camera_id)

    @property
    def is_mature(self) -> bool:
        """Mature = enough appearances to be a reliable match candidate."""
        return self.appearance_count >= 2

    @property
    def is_cross_camera(self) -> bool:
        return len(self.camera_set) > 1


# ── Cross-camera associator ─────────────────────────────────────────────────

class CrossCameraAssociator:
    """
    Cross-camera global ID assignment using appearance + temporal constraints.

    Parameters (from ``cfg`` dict, typically ``cfg["association"]``):
        assoc_sim_thresh       : float   Minimum cosine similarity (default 0.55).
        assoc_ema_alpha        : float   EMA smoothing factor (default 0.7).
        assoc_min_transit_sec  : float   Min seconds between camera transitions (default 0).
        assoc_max_transit_sec  : float   Max seconds for inactive entry matching (default 120).
        assoc_gallery_ttl      : float   TTL for inactive entries (default 90).
        assoc_max_gallery_size : int     Gallery cap (default 500).
        assoc_confirm_hits     : int     Hits before entry is a match candidate (default 2).
    """

    def __init__(self, cfg: dict):
        self.sim_thresh = float(cfg.get("assoc_sim_thresh", 0.55))
        self.ema_alpha = float(cfg.get("assoc_ema_alpha", 0.7))
        self.min_transit_sec = float(cfg.get("assoc_min_transit_sec", 0))
        self.max_transit_sec = float(cfg.get("assoc_max_transit_sec", 120))
        self.gallery_ttl = float(cfg.get("assoc_gallery_ttl", 90))
        self.max_gallery_size = int(cfg.get("assoc_max_gallery_size", 500))
        self.confirm_hits = int(cfg.get("assoc_confirm_hits", 2))

        # Gallery: global_id → GlobalIDEntry
        self._gallery: Dict[int, GlobalIDEntry] = {}
        self._next_gid: int = 1
        self._lock = threading.Lock()

        # Persistent mapping (cam_id, local_tid) → global_id
        self._track_to_gid: Dict[Tuple[str, int], int] = {}

        # Stats
        self._cross_cam_matches: int = 0

        logger.info(
            "CrossCameraAssociator — sim=%.2f ema=%.2f transit=[%.0f,%.0f]s "
            "ttl=%.0fs max=%d confirm=%d",
            self.sim_thresh, self.ema_alpha, self.min_transit_sec,
            self.max_transit_sec, self.gallery_ttl, self.max_gallery_size,
            self.confirm_hits,
        )

    # ── public API ───────────────────────────────────────────────────────

    def associate(
        self,
        all_tracks: Dict[str, List[Dict[str, Any]]],
        all_embs: Dict[str, Dict[int, np.ndarray]],
    ) -> Dict[str, Dict[int, int]]:
        """Assign global IDs to all tracks across all cameras.

        Parameters
        ----------
        all_tracks : {cam_id: [{'tid': int, 'bbox': (x,y,w,h), 'score': float}, ...]}
        all_embs   : {cam_id: {tid: np.ndarray(D,)}}  — may be empty on
                     non-ReID frames; in that case only Phase 1 runs.

        Returns
        -------
        {cam_id: {local_tid: global_id}}
        """
        with self._lock:
            return self._associate_impl(all_tracks, all_embs)

    def get_stats(self) -> dict:
        with self._lock:
            active = sum(1 for e in self._gallery.values() if e.state == "active")
            inactive = sum(1 for e in self._gallery.values() if e.state == "inactive")
            cross = sum(1 for e in self._gallery.values() if e.is_cross_camera)
            return {
                "gallery_size": len(self._gallery),
                "active": active,
                "inactive": inactive,
                "cross_camera_ids": cross,
                "cross_cam_matches": self._cross_cam_matches,
                "tracked_mappings": len(self._track_to_gid),
            }

    @property
    def gallery_size(self) -> int:
        return len(self._gallery)

    # ── internal implementation ──────────────────────────────────────────

    def _associate_impl(
        self,
        all_tracks: Dict[str, List[Dict[str, Any]]],
        all_embs: Dict[str, Dict[int, np.ndarray]],
    ) -> Dict[str, Dict[int, int]]:
        now = time.monotonic()
        result: Dict[str, Dict[int, int]] = {}
        current_active_keys: Set[Tuple[str, int]] = set()

        new_tracks: List[Tuple[str, int, np.ndarray, Tuple[float, float]]] = []
        have_new_embs = any(len(v) > 0 for v in all_embs.values())

        # ── Phase 1: Continuing tracks — keep existing GIDs ─────────────
        for cam_id, tracks in all_tracks.items():
            result[cam_id] = {}
            for t in tracks:
                tid = t["tid"]
                key = (cam_id, tid)
                current_active_keys.add(key)

                if key in self._track_to_gid:
                    gid = self._track_to_gid[key]
                    result[cam_id][tid] = gid

                    # EMA update gallery embedding if new embedding available
                    emb = all_embs.get(cam_id, {}).get(tid)
                    if emb is not None and np.any(emb) and gid in self._gallery:
                        entry = self._gallery[gid]
                        entry.update_embedding(emb, self.ema_alpha)
                        cx = float(t["bbox"][0]) + float(t["bbox"][2]) / 2
                        cy = float(t["bbox"][1]) + float(t["bbox"][3]) / 2
                        entry.touch(cam_id, now, (cx, cy))
                        entry.state = "active"
                else:
                    # New track — needs matching (only if we have embeddings)
                    emb = all_embs.get(cam_id, {}).get(tid)
                    if emb is not None and np.any(emb):
                        cx = float(t["bbox"][0]) + float(t["bbox"][2]) / 2
                        cy = float(t["bbox"][1]) + float(t["bbox"][3]) / 2
                        new_tracks.append((cam_id, tid, emb, (cx, cy)))
                    elif have_new_embs:
                        # Embeddings were computed this frame but this track
                        # didn't get one (too small crop, etc.) → assign fresh GID
                        gid = self._create_entry(
                            np.zeros(512, dtype=np.float32), cam_id, (0, 0), now
                        )
                        result[cam_id][tid] = gid
                        self._track_to_gid[key] = gid
                    else:
                        # Non-ReID frame: no embeddings available.
                        # Keep old GID if we had one, else skip (will get
                        # assigned on the next ReID frame).
                        pass

        if not new_tracks:
            self._cleanup_stale(current_active_keys, now)
            return result

        # ── Phase 2: Match new tracks against gallery ────────────────────

        # Candidates: mature gallery entries (enough hits for precision)
        gallery_candidates = [
            e for e in self._gallery.values() if e.is_mature
        ]

        if not gallery_candidates:
            # No mature entries — create fresh IDs
            for cam_id, tid, emb, center in new_tracks:
                gid = self._create_entry(emb, cam_id, center, now)
                result[cam_id][tid] = gid
                self._track_to_gid[(cam_id, tid)] = gid
            self._cleanup_stale(current_active_keys, now)
            return result

        # Build matrices
        query_embs = np.array([t[2] for t in new_tracks], dtype=np.float32)
        gallery_embs = np.array(
            [e.embedding for e in gallery_candidates], dtype=np.float32
        )

        # Cosine similarity (both sides L2-normalized)
        sim_matrix = query_embs @ gallery_embs.T            # (M, G)
        cost_matrix = 1.0 - sim_matrix                      # lower = better

        # Apply time-gating penalties
        for i, (cam_id, tid, _emb, _center) in enumerate(new_tracks):
            for j, entry in enumerate(gallery_candidates):
                if entry.last_camera == cam_id:
                    # Same camera, different track — possible fragmentation.
                    # Allow matching, but only if the entry is inactive
                    # (the old track disappeared and a new one appeared).
                    if entry.state == "active":
                        # Entry is active in the same camera under a different
                        # track → probably a different person. Block match.
                        #
                        # Check if the entry's track is still active.
                        entry_still_tracked = any(
                            v == entry.global_id
                            for k, v in self._track_to_gid.items()
                            if k in current_active_keys and k[0] == cam_id
                        )
                        if entry_still_tracked:
                            cost_matrix[i, j] = 2.0  # block: another track owns this GID in same cam
                else:
                    # Different camera — apply transit time constraints
                    elapsed = now - entry.last_seen
                    if elapsed < self.min_transit_sec:
                        cost_matrix[i, j] = 2.0      # too fast
                    elif entry.state == "inactive" and elapsed > self.max_transit_sec:
                        cost_matrix[i, j] = 2.0      # too old

        # Hungarian matching
        M, G = cost_matrix.shape
        if M > 0 and G > 0:
            row_idx, col_idx = linear_sum_assignment(cost_matrix)
        else:
            row_idx, col_idx = np.array([], dtype=int), np.array([], dtype=int)

        matched_queries: Set[int] = set()

        for r, c in zip(row_idx, col_idx):
            if sim_matrix[r, c] < self.sim_thresh:
                continue

            cam_id, tid, emb, center = new_tracks[r]
            entry = gallery_candidates[c]
            gid = entry.global_id

            # Accept match
            entry.update_embedding(emb, self.ema_alpha)
            entry.touch(cam_id, now, center)
            entry.state = "active"

            result[cam_id][tid] = gid
            self._track_to_gid[(cam_id, tid)] = gid
            matched_queries.add(r)

            if entry.is_cross_camera:
                self._cross_cam_matches += 1
                logger.debug(
                    "Cross-cam match G%d: cams=%s sim=%.3f",
                    gid, entry.camera_set, sim_matrix[r, c],
                )

        # Unmatched new tracks → fresh global IDs
        for i, (cam_id, tid, emb, center) in enumerate(new_tracks):
            if i not in matched_queries:
                gid = self._create_entry(emb, cam_id, center, now)
                result[cam_id][tid] = gid
                self._track_to_gid[(cam_id, tid)] = gid

        # ── Phase 3: Cleanup ─────────────────────────────────────────────
        self._cleanup_stale(current_active_keys, now)

        return result

    # ── helpers ───────────────────────────────────────────────────────────

    def _create_entry(
        self, emb: np.ndarray, cam_id: str,
        center: Tuple[float, float], timestamp: float,
    ) -> int:
        gid = self._next_gid
        self._next_gid += 1
        entry = GlobalIDEntry(gid, emb, cam_id, center)
        entry.last_seen = timestamp
        entry.first_seen = timestamp
        self._gallery[gid] = entry
        return gid

    def _cleanup_stale(self, current_active_keys: Set[Tuple[str, int]], now: float):
        """Mark disappeared tracks as inactive and prune old entries."""

        # ── Mark disappeared tracks ──
        stale_keys = [k for k in self._track_to_gid if k not in current_active_keys]
        for key in stale_keys:
            gid = self._track_to_gid.pop(key)
            if gid in self._gallery:
                # Only mark inactive if no other active track uses this GID
                still_active = any(
                    v == gid for k, v in self._track_to_gid.items()
                    if k in current_active_keys
                )
                if not still_active:
                    self._gallery[gid].state = "inactive"

        # ── Prune expired inactive entries ──
        to_prune = [
            gid for gid, e in self._gallery.items()
            if e.state == "inactive" and (now - e.last_seen) > self.gallery_ttl
        ]
        for gid in to_prune:
            del self._gallery[gid]

        # ── Enforce gallery cap ──
        if len(self._gallery) > self.max_gallery_size:
            sorted_entries = sorted(
                self._gallery.items(), key=lambda x: x[1].last_seen
            )
            excess = len(self._gallery) - self.max_gallery_size
            for gid, _ in sorted_entries[:excess]:
                del self._gallery[gid]
                # Clean up any orphaned track→gid mappings
                orphan_keys = [k for k, v in self._track_to_gid.items() if v == gid]
                for k in orphan_keys:
                    del self._track_to_gid[k]
