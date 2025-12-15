"""
Camera topology graph for cross-camera association.
Defines spatial relationships and expected transfer times between cameras.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class CameraTransition:
    """Represents a possible transition between two cameras."""
    source_cam: str
    target_cam: str
    min_transfer_time: float  # seconds
    max_transfer_time: float  # seconds
    probability: float = 1.0  # Prior probability of this transition
    
    def is_valid_time(self, elapsed: float) -> bool:
        """Check if elapsed time is within valid transfer window."""
        return self.min_transfer_time <= elapsed <= self.max_transfer_time
    
    def time_score(self, elapsed: float) -> float:
        """
        Score based on how well elapsed time fits the expected window.
        Returns 0-1, with 1 being optimal.
        """
        if elapsed < self.min_transfer_time:
            return max(0, 1 - (self.min_transfer_time - elapsed) / self.min_transfer_time)
        elif elapsed > self.max_transfer_time:
            return max(0, 1 - (elapsed - self.max_transfer_time) / self.max_transfer_time)
        else:
            # Within window - score based on being close to midpoint
            mid = (self.min_transfer_time + self.max_transfer_time) / 2
            deviation = abs(elapsed - mid) / (self.max_transfer_time - self.min_transfer_time + 1e-6)
            return 1.0 - 0.5 * deviation


@dataclass
class CameraNode:
    """Represents a camera in the topology."""
    camera_id: str
    name: str = ""
    location: Optional[Tuple[float, float]] = None  # (x, y) physical coordinates
    entry_zones: List[str] = field(default_factory=list)  # Zone IDs where people enter
    exit_zones: List[str] = field(default_factory=list)   # Zone IDs where people exit
    
    def __post_init__(self):
        if not self.name:
            self.name = self.camera_id


class CameraGraph:
    """
    Graph representing camera topology and transition constraints.
    Used to filter candidate matches based on physical plausibility.
    """
    
    def __init__(self):
        """Initialize empty camera graph."""
        self.cameras: Dict[str, CameraNode] = {}
        self.transitions: Dict[Tuple[str, str], CameraTransition] = {}
        self._adjacency: Dict[str, Set[str]] = {}
    
    def add_camera(
        self,
        camera_id: str,
        name: str = "",
        location: Optional[Tuple[float, float]] = None,
        entry_zones: Optional[List[str]] = None,
        exit_zones: Optional[List[str]] = None
    ):
        """Add a camera to the graph."""
        self.cameras[camera_id] = CameraNode(
            camera_id=camera_id,
            name=name or camera_id,
            location=location,
            entry_zones=entry_zones or [],
            exit_zones=exit_zones or []
        )
        if camera_id not in self._adjacency:
            self._adjacency[camera_id] = set()
        
        logger.debug(f"Added camera: {camera_id}")
    
    def add_transition(
        self,
        source_cam: str,
        target_cam: str,
        min_time: float,
        max_time: float,
        probability: float = 1.0,
        bidirectional: bool = True
    ):
        """
        Add a transition edge between cameras.
        
        Args:
            source_cam: Source camera ID
            target_cam: Target camera ID
            min_time: Minimum expected transfer time (seconds)
            max_time: Maximum expected transfer time (seconds)
            probability: Prior probability of this transition
            bidirectional: If True, add reverse transition as well
        """
        # Ensure cameras exist
        if source_cam not in self.cameras:
            self.add_camera(source_cam)
        if target_cam not in self.cameras:
            self.add_camera(target_cam)
        
        # Add transition
        self.transitions[(source_cam, target_cam)] = CameraTransition(
            source_cam=source_cam,
            target_cam=target_cam,
            min_transfer_time=min_time,
            max_transfer_time=max_time,
            probability=probability
        )
        self._adjacency[source_cam].add(target_cam)
        
        # Add reverse transition if bidirectional
        if bidirectional:
            self.transitions[(target_cam, source_cam)] = CameraTransition(
                source_cam=target_cam,
                target_cam=source_cam,
                min_transfer_time=min_time,
                max_transfer_time=max_time,
                probability=probability
            )
            self._adjacency[target_cam].add(source_cam)
        
        logger.debug(f"Added transition: {source_cam} -> {target_cam} ({min_time}-{max_time}s)")
    
    def get_transition(self, source_cam: str, target_cam: str) -> Optional[CameraTransition]:
        """Get transition between two cameras."""
        return self.transitions.get((source_cam, target_cam))
    
    def can_transition(self, source_cam: str, target_cam: str) -> bool:
        """Check if transition between cameras is possible."""
        if source_cam == target_cam:
            return True  # Same camera is always valid
        return (source_cam, target_cam) in self.transitions
    
    def is_valid_handoff(
        self,
        source_cam: str,
        target_cam: str,
        elapsed_time: float
    ) -> bool:
        """
        Check if a handoff is temporally valid.
        
        Args:
            source_cam: Source camera ID
            target_cam: Target camera ID
            elapsed_time: Time since last seen in source camera
            
        Returns:
            True if handoff is plausible
        """
        if source_cam == target_cam:
            return True
        
        transition = self.get_transition(source_cam, target_cam)
        if transition is None:
            return False
        
        return transition.is_valid_time(elapsed_time)
    
    def get_handoff_score(
        self,
        source_cam: str,
        target_cam: str,
        elapsed_time: float
    ) -> float:
        """
        Get a score for how plausible a handoff is.
        
        Args:
            source_cam: Source camera ID
            target_cam: Target camera ID
            elapsed_time: Time since last seen in source camera
            
        Returns:
            Score between 0-1, higher is more plausible
        """
        if source_cam == target_cam:
            return 1.0
        
        transition = self.get_transition(source_cam, target_cam)
        if transition is None:
            return 0.0
        
        return transition.time_score(elapsed_time) * transition.probability
    
    def get_neighbors(self, camera_id: str) -> Set[str]:
        """Get cameras that can be reached from this camera."""
        return self._adjacency.get(camera_id, set())
    
    def get_candidate_sources(
        self,
        target_cam: str,
        max_elapsed_time: float
    ) -> List[str]:
        """
        Get cameras that could have handed off to target camera.
        
        Args:
            target_cam: Target camera ID
            max_elapsed_time: Maximum elapsed time to consider
            
        Returns:
            List of source camera IDs
        """
        sources = []
        for (src, tgt), transition in self.transitions.items():
            if tgt == target_cam and transition.min_transfer_time <= max_elapsed_time:
                sources.append(src)
        return sources
    
    @classmethod
    def from_config(cls, config: Dict) -> 'CameraGraph':
        """
        Create camera graph from configuration.
        
        Expected config format:
        {
            'cameras': [
                {'id': 'cam1', 'name': 'Entrance', 'entry_zones': ['z1'], 'exit_zones': ['z2']},
                ...
            ],
            'transitions': [
                {'from': 'cam1', 'to': 'cam2', 'min_time': 5, 'max_time': 30},
                ...
            ]
        }
        """
        graph = cls()
        
        # Add cameras
        for cam_cfg in config.get('cameras', []):
            graph.add_camera(
                camera_id=cam_cfg['id'],
                name=cam_cfg.get('name', ''),
                location=cam_cfg.get('location'),
                entry_zones=cam_cfg.get('entry_zones', []),
                exit_zones=cam_cfg.get('exit_zones', [])
            )
        
        # Add transitions
        for trans_cfg in config.get('transitions', []):
            graph.add_transition(
                source_cam=trans_cfg['from'],
                target_cam=trans_cfg['to'],
                min_time=trans_cfg.get('min_time', 0),
                max_time=trans_cfg.get('max_time', 60),
                probability=trans_cfg.get('probability', 1.0),
                bidirectional=trans_cfg.get('bidirectional', True)
            )
        
        return graph
    
    @classmethod
    def create_fully_connected(
        cls,
        camera_ids: List[str],
        default_min_time: float = 1.0,
        default_max_time: float = 120.0
    ) -> 'CameraGraph':
        """
        Create a fully connected camera graph.
        Useful when no spatial constraints are known.
        """
        graph = cls()
        
        for cam_id in camera_ids:
            graph.add_camera(cam_id)
        
        for i, src in enumerate(camera_ids):
            for tgt in camera_ids[i+1:]:
                graph.add_transition(
                    src, tgt,
                    min_time=default_min_time,
                    max_time=default_max_time,
                    bidirectional=True
                )
        
        return graph

