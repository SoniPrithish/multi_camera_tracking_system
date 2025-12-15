"""
Cross-camera matching using Hungarian algorithm.
Combines appearance, temporal, and spatial cues for optimal assignment.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MatchCandidate:
    """Represents a potential cross-camera match."""
    source_cam: str
    source_track_id: int
    source_global_id: Optional[int]
    target_cam: str
    target_track_id: int
    embedding_similarity: float
    time_score: float
    zone_valid: bool
    
    @property
    def combined_score(self) -> float:
        """Combined matching score."""
        if not self.zone_valid:
            return 0.0
        return self.embedding_similarity * self.time_score


class CrossCameraMatcher:
    """
    Performs cross-camera matching using appearance and constraints.
    Uses Hungarian algorithm for optimal assignment.
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.5,
        time_weight: float = 0.3,
        appearance_weight: float = 0.7,
    ):
        """
        Initialize matcher.
        
        Args:
            similarity_threshold: Minimum similarity for a valid match
            time_weight: Weight for temporal score
            appearance_weight: Weight for appearance similarity
        """
        self.similarity_threshold = similarity_threshold
        self.time_weight = time_weight
        self.appearance_weight = appearance_weight
    
    def compute_cost_matrix(
        self,
        lost_tracks: List[Dict],
        new_tracks: List[Dict],
        camera_graph: Any,
        zone_gate: Any
    ) -> Tuple[np.ndarray, List[Dict], List[Dict]]:
        """
        Compute cost matrix for Hungarian matching.
        
        Args:
            lost_tracks: List of lost tracks with embeddings and metadata
            new_tracks: List of new/unmatched tracks
            camera_graph: CameraGraph for temporal constraints
            zone_gate: ZoneGate for spatial constraints
            
        Returns:
            cost_matrix: (N_lost, N_new) cost matrix
            valid_lost: Filtered lost tracks
            valid_new: Filtered new tracks
        """
        if len(lost_tracks) == 0 or len(new_tracks) == 0:
            return np.array([[]]), lost_tracks, new_tracks
        
        n_lost = len(lost_tracks)
        n_new = len(new_tracks)
        
        # Initialize cost matrix with high cost (no match)
        cost_matrix = np.ones((n_lost, n_new)) * 1000.0
        
        for i, lost in enumerate(lost_tracks):
            lost_emb = lost.get('embedding')
            lost_cam = lost['camera_id']
            lost_time = lost.get('last_seen_time', 0)
            lost_tid = lost['track_id']
            
            if lost_emb is None:
                continue
            
            for j, new in enumerate(new_tracks):
                new_emb = new.get('embedding')
                new_cam = new['camera_id']
                new_time = new.get('detection_time', 0)
                new_bbox = new.get('bbox', (0, 0, 0, 0))
                
                if new_emb is None:
                    continue
                
                # Same camera - skip (handled by single-camera tracker)
                if lost_cam == new_cam:
                    continue
                
                # Time elapsed
                elapsed = new_time - lost_time
                if elapsed < 0:
                    continue
                
                # Check temporal constraint
                if camera_graph is not None:
                    if not camera_graph.is_valid_handoff(lost_cam, new_cam, elapsed):
                        continue
                    time_score = camera_graph.get_handoff_score(lost_cam, new_cam, elapsed)
                else:
                    # Default time score based on reasonable transfer time
                    time_score = max(0, 1 - elapsed / 120.0)  # Decay over 2 minutes
                
                # Check zone constraint
                if zone_gate is not None:
                    if not zone_gate.can_match(lost_cam, lost_tid, new_cam, new_bbox):
                        continue
                
                # Compute appearance similarity
                similarity = float(np.dot(lost_emb, new_emb))
                
                if similarity < self.similarity_threshold:
                    continue
                
                # Combined score (higher is better)
                combined = (
                    self.appearance_weight * similarity +
                    self.time_weight * time_score
                )
                
                # Convert to cost (lower is better)
                cost_matrix[i, j] = 1.0 - combined
        
        return cost_matrix, lost_tracks, new_tracks
    
    def match(
        self,
        lost_tracks: List[Dict],
        new_tracks: List[Dict],
        camera_graph: Any = None,
        zone_gate: Any = None
    ) -> List[Tuple[int, int, float]]:
        """
        Find optimal matches between lost and new tracks.
        
        Args:
            lost_tracks: Lost tracks with embeddings
            new_tracks: New/unmatched tracks with embeddings
            camera_graph: Optional CameraGraph for temporal constraints
            zone_gate: Optional ZoneGate for spatial constraints
            
        Returns:
            List of (lost_idx, new_idx, score) tuples
        """
        if len(lost_tracks) == 0 or len(new_tracks) == 0:
            return []
        
        cost_matrix, _, _ = self.compute_cost_matrix(
            lost_tracks, new_tracks, camera_graph, zone_gate
        )
        
        if cost_matrix.size == 0:
            return []
        
        # Run Hungarian algorithm
        matches = self._hungarian_match(cost_matrix)
        
        # Filter by threshold and compute final scores
        result = []
        for i, j in matches:
            cost = cost_matrix[i, j]
            if cost < 1000.0:  # Valid match
                score = 1.0 - cost
                result.append((i, j, score))
        
        return result
    
    def _hungarian_match(
        self,
        cost_matrix: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Solve assignment problem using Hungarian algorithm.
        
        Returns:
            List of (row, col) assignment pairs
        """
        try:
            import lap
            _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
            matches = []
            for i, j in enumerate(x):
                if j >= 0:
                    matches.append((i, j))
            return matches
        except ImportError:
            pass
        
        # Fallback to scipy
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            return list(zip(row_ind, col_ind))
        except ImportError:
            logger.warning("Neither lap nor scipy available for Hungarian matching")
            return self._greedy_match(cost_matrix)
    
    def _greedy_match(
        self,
        cost_matrix: np.ndarray
    ) -> List[Tuple[int, int]]:
        """Greedy matching fallback."""
        matches = []
        used_rows = set()
        used_cols = set()
        
        # Flatten and sort by cost
        indices = np.argsort(cost_matrix.flatten())
        
        for idx in indices:
            i = idx // cost_matrix.shape[1]
            j = idx % cost_matrix.shape[1]
            
            if i in used_rows or j in used_cols:
                continue
            
            if cost_matrix[i, j] < 1000.0:
                matches.append((i, j))
                used_rows.add(i)
                used_cols.add(j)
        
        return matches


class AppearanceMatcher:
    """
    Simple appearance-based matcher for ReID.
    Uses cosine similarity with gallery embeddings.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        top_k: int = 5
    ):
        """
        Initialize appearance matcher.
        
        Args:
            threshold: Minimum similarity threshold
            top_k: Maximum number of matches to return
        """
        self.threshold = threshold
        self.top_k = top_k
    
    def find_matches(
        self,
        query_embedding: np.ndarray,
        gallery: Dict[int, np.ndarray]
    ) -> List[Tuple[int, float]]:
        """
        Find matching identities in gallery.
        
        Args:
            query_embedding: Query embedding vector
            gallery: Dict mapping global_id to embedding
            
        Returns:
            List of (global_id, similarity) sorted by similarity
        """
        if len(gallery) == 0:
            return []
        
        # Compute similarities
        similarities = []
        for gid, gallery_emb in gallery.items():
            if gallery_emb is None:
                continue
            
            sim = float(np.dot(query_embedding, gallery_emb))
            if sim >= self.threshold:
                similarities.append((gid, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:self.top_k]
    
    def compute_distance_matrix(
        self,
        query_embeddings: np.ndarray,
        gallery_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise distance matrix.
        
        Args:
            query_embeddings: (N, D) query embeddings
            gallery_embeddings: (M, D) gallery embeddings
            
        Returns:
            (N, M) distance matrix (1 - cosine similarity)
        """
        # Normalize
        query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-6)
        gallery_norm = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-6)
        
        # Cosine similarity
        similarity = query_norm @ gallery_norm.T
        
        # Convert to distance
        return 1.0 - similarity

