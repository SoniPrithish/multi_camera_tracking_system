"""
Global identity registry for cross-camera tracking.
Maintains consistent global IDs across all cameras.
"""

import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class GlobalIdentity:
    """Represents a globally tracked identity."""
    global_id: int
    first_seen_time: float
    last_seen_time: float
    first_seen_camera: str
    last_seen_camera: str
    embedding: Optional[np.ndarray] = None
    
    # Track associations per camera
    camera_track_ids: Dict[str, List[int]] = field(default_factory=dict)
    
    # Appearance history
    embedding_history: List[np.ndarray] = field(default_factory=list)
    
    # Movement history
    camera_history: List[Tuple[str, float]] = field(default_factory=list)
    
    def update(
        self,
        camera_id: str,
        track_id: int,
        timestamp: float,
        embedding: Optional[np.ndarray] = None,
        ema_alpha: float = 0.9
    ):
        """Update identity with new observation."""
        self.last_seen_time = timestamp
        self.last_seen_camera = camera_id
        
        # Track association
        if camera_id not in self.camera_track_ids:
            self.camera_track_ids[camera_id] = []
        if track_id not in self.camera_track_ids[camera_id]:
            self.camera_track_ids[camera_id].append(track_id)
        
        # Camera history
        if len(self.camera_history) == 0 or self.camera_history[-1][0] != camera_id:
            self.camera_history.append((camera_id, timestamp))
        
        # Update embedding with EMA
        if embedding is not None:
            if self.embedding is None:
                self.embedding = embedding.copy()
            else:
                self.embedding = ema_alpha * self.embedding + (1 - ema_alpha) * embedding
                self.embedding = self.embedding / (np.linalg.norm(self.embedding) + 1e-6)
            
            self.embedding_history.append(embedding)
            if len(self.embedding_history) > 100:
                self.embedding_history.pop(0)
    
    @property
    def age(self) -> float:
        """Time since first seen."""
        return time.time() - self.first_seen_time
    
    @property
    def time_since_seen(self) -> float:
        """Time since last seen."""
        return time.time() - self.last_seen_time
    
    @property
    def camera_count(self) -> int:
        """Number of cameras this identity has been seen in."""
        return len(self.camera_track_ids)
    
    def was_in_camera(self, camera_id: str) -> bool:
        """Check if identity was ever seen in a camera."""
        return camera_id in self.camera_track_ids
    
    def to_dict(self) -> Dict:
        """Convert to serializable dict."""
        return {
            'global_id': self.global_id,
            'first_seen_time': self.first_seen_time,
            'last_seen_time': self.last_seen_time,
            'first_seen_camera': self.first_seen_camera,
            'last_seen_camera': self.last_seen_camera,
            'camera_track_ids': self.camera_track_ids,
            'camera_history': self.camera_history,
            'camera_count': self.camera_count,
            'age': self.age,
            'has_embedding': self.embedding is not None
        }


class GlobalRegistry:
    """
    Central registry for global identities.
    Handles ID assignment, merging, and persistence.
    """
    
    def __init__(
        self,
        ema_alpha: float = 0.9,
        lost_timeout: float = 300.0,  # 5 minutes
        max_identities: int = 10000,
        manifest_path: Optional[str] = None
    ):
        """
        Initialize global registry.
        
        Args:
            ema_alpha: EMA smoothing factor for embeddings
            lost_timeout: Time after which identities are considered lost
            max_identities: Maximum number of identities to track
            manifest_path: Path to save identity manifest
        """
        self.ema_alpha = ema_alpha
        self.lost_timeout = lost_timeout
        self.max_identities = max_identities
        self.manifest_path = manifest_path
        
        # Identity storage
        self.identities: Dict[int, GlobalIdentity] = {}
        self.next_id = 1
        
        # Lookup indices
        self._camera_to_gids: Dict[str, Set[int]] = defaultdict(set)
        self._track_to_gid: Dict[str, Dict[int, int]] = defaultdict(dict)  # cam -> {tid -> gid}
        
        # Statistics
        self.stats = {
            'total_created': 0,
            'total_merged': 0,
            'total_removed': 0,
            'handoffs': 0
        }
    
    def get_or_create(
        self,
        camera_id: str,
        track_id: int,
        embedding: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None
    ) -> int:
        """
        Get existing global ID or create new one.
        
        Args:
            camera_id: Camera identifier
            track_id: Local track ID
            embedding: Optional appearance embedding
            timestamp: Optional timestamp
            
        Returns:
            Global ID
        """
        timestamp = timestamp or time.time()
        
        # Check if track already has global ID
        existing_gid = self._track_to_gid[camera_id].get(track_id)
        if existing_gid is not None and existing_gid in self.identities:
            identity = self.identities[existing_gid]
            identity.update(camera_id, track_id, timestamp, embedding, self.ema_alpha)
            return existing_gid
        
        # Create new identity
        gid = self._create_identity(camera_id, track_id, timestamp, embedding)
        return gid
    
    def _create_identity(
        self,
        camera_id: str,
        track_id: int,
        timestamp: float,
        embedding: Optional[np.ndarray]
    ) -> int:
        """Create a new global identity."""
        gid = self.next_id
        self.next_id += 1
        
        identity = GlobalIdentity(
            global_id=gid,
            first_seen_time=timestamp,
            last_seen_time=timestamp,
            first_seen_camera=camera_id,
            last_seen_camera=camera_id,
            embedding=embedding.copy() if embedding is not None else None
        )
        identity.camera_track_ids[camera_id] = [track_id]
        identity.camera_history.append((camera_id, timestamp))
        
        self.identities[gid] = identity
        self._camera_to_gids[camera_id].add(gid)
        self._track_to_gid[camera_id][track_id] = gid
        
        self.stats['total_created'] += 1
        
        # Cleanup if too many identities
        if len(self.identities) > self.max_identities:
            self._cleanup_old_identities()
        
        logger.debug(f"Created global ID {gid} for {camera_id}:{track_id}")
        return gid
    
    def assign_global_id(
        self,
        camera_id: str,
        track_id: int,
        global_id: int,
        embedding: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None
    ):
        """
        Assign an existing global ID to a track.
        Used for cross-camera handoffs.
        """
        timestamp = timestamp or time.time()
        
        if global_id not in self.identities:
            logger.warning(f"Global ID {global_id} not found")
            return
        
        identity = self.identities[global_id]
        identity.update(camera_id, track_id, timestamp, embedding, self.ema_alpha)
        
        self._camera_to_gids[camera_id].add(global_id)
        self._track_to_gid[camera_id][track_id] = global_id
        
        self.stats['handoffs'] += 1
        logger.debug(f"Assigned global ID {global_id} to {camera_id}:{track_id}")
    
    def get_identity(self, global_id: int) -> Optional[GlobalIdentity]:
        """Get identity by global ID."""
        return self.identities.get(global_id)
    
    def get_global_id(self, camera_id: str, track_id: int) -> Optional[int]:
        """Get global ID for a camera track."""
        return self._track_to_gid[camera_id].get(track_id)
    
    def get_embedding(self, global_id: int) -> Optional[np.ndarray]:
        """Get embedding for a global ID."""
        identity = self.identities.get(global_id)
        return identity.embedding if identity else None
    
    def get_active_identities(
        self,
        max_age: Optional[float] = None
    ) -> List[GlobalIdentity]:
        """Get all active (recently seen) identities."""
        max_age = max_age or self.lost_timeout
        now = time.time()
        
        return [
            identity for identity in self.identities.values()
            if now - identity.last_seen_time <= max_age
        ]
    
    def get_lost_identities(
        self,
        min_lost_time: float = 5.0,
        max_lost_time: Optional[float] = None
    ) -> List[GlobalIdentity]:
        """Get identities that are lost but not timed out."""
        max_lost_time = max_lost_time or self.lost_timeout
        now = time.time()
        
        return [
            identity for identity in self.identities.values()
            if min_lost_time <= now - identity.last_seen_time <= max_lost_time
        ]
    
    def merge_identities(self, keep_gid: int, merge_gid: int):
        """
        Merge two identities into one.
        Used when two separate tracks are determined to be the same person.
        """
        if keep_gid not in self.identities or merge_gid not in self.identities:
            return
        
        keep = self.identities[keep_gid]
        merge = self.identities[merge_gid]
        
        # Merge camera tracks
        for cam, tids in merge.camera_track_ids.items():
            if cam not in keep.camera_track_ids:
                keep.camera_track_ids[cam] = []
            keep.camera_track_ids[cam].extend(tids)
            
            # Update track-to-gid mapping
            for tid in tids:
                self._track_to_gid[cam][tid] = keep_gid
        
        # Merge camera history
        keep.camera_history.extend(merge.camera_history)
        keep.camera_history.sort(key=lambda x: x[1])
        
        # Merge embeddings
        keep.embedding_history.extend(merge.embedding_history)
        
        # Update times
        keep.first_seen_time = min(keep.first_seen_time, merge.first_seen_time)
        keep.last_seen_time = max(keep.last_seen_time, merge.last_seen_time)
        
        # Remove merged identity
        del self.identities[merge_gid]
        for cam_gids in self._camera_to_gids.values():
            cam_gids.discard(merge_gid)
        
        self.stats['total_merged'] += 1
        logger.info(f"Merged identity {merge_gid} into {keep_gid}")
    
    def _cleanup_old_identities(self):
        """Remove very old identities to save memory."""
        now = time.time()
        to_remove = []
        
        for gid, identity in self.identities.items():
            if now - identity.last_seen_time > self.lost_timeout * 2:
                to_remove.append(gid)
        
        for gid in to_remove:
            self._remove_identity(gid)
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old identities")
    
    def _remove_identity(self, global_id: int):
        """Remove an identity from registry."""
        if global_id not in self.identities:
            return
        
        identity = self.identities[global_id]
        
        # Remove from indices
        for cam in identity.camera_track_ids.keys():
            self._camera_to_gids[cam].discard(global_id)
            for tid in identity.camera_track_ids[cam]:
                if self._track_to_gid[cam].get(tid) == global_id:
                    del self._track_to_gid[cam][tid]
        
        del self.identities[global_id]
        self.stats['total_removed'] += 1
    
    def save_manifest(self, path: Optional[str] = None):
        """Save identity manifest to file."""
        path = path or self.manifest_path
        if path is None:
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        manifest = {
            'timestamp': time.time(),
            'stats': self.stats,
            'identities': [
                identity.to_dict() for identity in self.identities.values()
            ]
        }
        
        with open(path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Saved manifest with {len(self.identities)} identities to {path}")
    
    def get_stats(self) -> Dict:
        """Get registry statistics."""
        active = self.get_active_identities()
        lost = self.get_lost_identities()
        
        return {
            **self.stats,
            'current_identities': len(self.identities),
            'active_identities': len(active),
            'lost_identities': len(lost),
            'next_id': self.next_id
        }
    
    def reset(self):
        """Reset all state."""
        self.identities.clear()
        self._camera_to_gids.clear()
        self._track_to_gid.clear()
        self.next_id = 1
        self.stats = {
            'total_created': 0,
            'total_merged': 0,
            'total_removed': 0,
            'handoffs': 0
        }

