"""
Zone-based gating for cross-camera association.
Uses entry/exit zones to constrain matching candidates.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Zone:
    """Represents a polygonal zone in a camera view."""
    zone_id: str
    camera_id: str
    polygon: np.ndarray  # Shape: (N, 2) array of (x, y) vertices
    zone_type: str = "general"  # "entry", "exit", "general"
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = self.zone_id
        self.polygon = np.array(self.polygon, dtype=np.float32)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if a point is inside the polygon."""
        return self._point_in_polygon(x, y, self.polygon)
    
    def contains_bbox_center(self, bbox: Tuple[float, float, float, float]) -> bool:
        """Check if bbox center is inside the zone."""
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        return self.contains_point(cx, cy)
    
    def bbox_overlap(self, bbox: Tuple[float, float, float, float]) -> float:
        """
        Calculate approximate overlap between bbox and zone.
        Returns value between 0-1.
        """
        x, y, w, h = bbox
        
        # Sample points in bbox
        samples = [
            (x + w * 0.25, y + h * 0.25),
            (x + w * 0.75, y + h * 0.25),
            (x + w * 0.25, y + h * 0.75),
            (x + w * 0.75, y + h * 0.75),
            (x + w * 0.5, y + h * 0.5),
        ]
        
        inside = sum(1 for sx, sy in samples if self.contains_point(sx, sy))
        return inside / len(samples)
    
    @staticmethod
    def _point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
        """Ray casting algorithm for point-in-polygon test."""
        n = len(polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
                inside = not inside
            
            j = i
        
        return inside


class ZoneManager:
    """
    Manages zones across all cameras.
    Provides zone-based filtering for cross-camera matching.
    """
    
    def __init__(self):
        """Initialize zone manager."""
        self.zones: Dict[str, Zone] = {}  # zone_id -> Zone
        self.camera_zones: Dict[str, List[str]] = {}  # camera_id -> [zone_ids]
        self.entry_zones: Dict[str, List[str]] = {}  # camera_id -> [entry zone_ids]
        self.exit_zones: Dict[str, List[str]] = {}   # camera_id -> [exit zone_ids]
    
    def add_zone(
        self,
        zone_id: str,
        camera_id: str,
        polygon: List[Tuple[float, float]],
        zone_type: str = "general",
        name: str = ""
    ):
        """
        Add a zone.
        
        Args:
            zone_id: Unique zone identifier
            camera_id: Camera this zone belongs to
            polygon: List of (x, y) vertices
            zone_type: "entry", "exit", or "general"
            name: Human-readable name
        """
        zone = Zone(
            zone_id=zone_id,
            camera_id=camera_id,
            polygon=np.array(polygon),
            zone_type=zone_type,
            name=name or zone_id
        )
        
        self.zones[zone_id] = zone
        
        # Index by camera
        if camera_id not in self.camera_zones:
            self.camera_zones[camera_id] = []
            self.entry_zones[camera_id] = []
            self.exit_zones[camera_id] = []
        
        self.camera_zones[camera_id].append(zone_id)
        
        if zone_type == "entry":
            self.entry_zones[camera_id].append(zone_id)
        elif zone_type == "exit":
            self.exit_zones[camera_id].append(zone_id)
        
        logger.debug(f"Added zone: {zone_id} ({zone_type}) to camera {camera_id}")
    
    def get_zone(self, zone_id: str) -> Optional[Zone]:
        """Get a zone by ID."""
        return self.zones.get(zone_id)
    
    def get_zones_containing_point(
        self,
        camera_id: str,
        x: float,
        y: float
    ) -> List[Zone]:
        """Get all zones containing a point."""
        zones = []
        for zone_id in self.camera_zones.get(camera_id, []):
            zone = self.zones[zone_id]
            if zone.contains_point(x, y):
                zones.append(zone)
        return zones
    
    def get_zones_for_bbox(
        self,
        camera_id: str,
        bbox: Tuple[float, float, float, float],
        min_overlap: float = 0.5
    ) -> List[Zone]:
        """Get zones that overlap with a bounding box."""
        zones = []
        for zone_id in self.camera_zones.get(camera_id, []):
            zone = self.zones[zone_id]
            if zone.bbox_overlap(bbox) >= min_overlap:
                zones.append(zone)
        return zones
    
    def is_in_exit_zone(
        self,
        camera_id: str,
        bbox: Tuple[float, float, float, float]
    ) -> bool:
        """Check if bbox is in an exit zone."""
        for zone_id in self.exit_zones.get(camera_id, []):
            zone = self.zones[zone_id]
            if zone.contains_bbox_center(bbox):
                return True
        return False
    
    def is_in_entry_zone(
        self,
        camera_id: str,
        bbox: Tuple[float, float, float, float]
    ) -> bool:
        """Check if bbox is in an entry zone."""
        for zone_id in self.entry_zones.get(camera_id, []):
            zone = self.zones[zone_id]
            if zone.contains_bbox_center(bbox):
                return True
        return False
    
    @classmethod
    def from_config(cls, config: Dict) -> 'ZoneManager':
        """
        Create ZoneManager from configuration.
        
        Expected config format:
        {
            'zones': [
                {
                    'id': 'zone1',
                    'camera_id': 'cam1',
                    'polygon': [[x1, y1], [x2, y2], ...],
                    'type': 'entry',
                    'name': 'Main Entrance'
                },
                ...
            ]
        }
        """
        manager = cls()
        
        for zone_cfg in config.get('zones', []):
            manager.add_zone(
                zone_id=zone_cfg['id'],
                camera_id=zone_cfg['camera_id'],
                polygon=zone_cfg['polygon'],
                zone_type=zone_cfg.get('type', 'general'),
                name=zone_cfg.get('name', '')
            )
        
        return manager


class ZoneGate:
    """
    Uses zones to gate cross-camera matching candidates.
    Only allows matches where track exited via exit zone and
    entered via entry zone.
    """
    
    def __init__(
        self,
        zone_manager: ZoneManager,
        require_exit_zone: bool = True,
        require_entry_zone: bool = True
    ):
        """
        Initialize zone gate.
        
        Args:
            zone_manager: ZoneManager instance
            require_exit_zone: If True, require track to be in exit zone
            require_entry_zone: If True, require track to be in entry zone
        """
        self.zone_manager = zone_manager
        self.require_exit_zone = require_exit_zone
        self.require_entry_zone = require_entry_zone
        
        # Track last known positions (for exit zone check)
        self.last_positions: Dict[str, Dict[int, Tuple[float, float, float, float]]] = {}
    
    def update_position(
        self,
        camera_id: str,
        track_id: int,
        bbox: Tuple[float, float, float, float]
    ):
        """Update last known position for a track."""
        if camera_id not in self.last_positions:
            self.last_positions[camera_id] = {}
        self.last_positions[camera_id][track_id] = bbox
    
    def was_in_exit_zone(self, camera_id: str, track_id: int) -> bool:
        """Check if track was last seen in exit zone."""
        bbox = self.last_positions.get(camera_id, {}).get(track_id)
        if bbox is None:
            return not self.require_exit_zone
        return self.zone_manager.is_in_exit_zone(camera_id, bbox)
    
    def is_in_entry_zone(
        self,
        camera_id: str,
        bbox: Tuple[float, float, float, float]
    ) -> bool:
        """Check if bbox is in entry zone."""
        return self.zone_manager.is_in_entry_zone(camera_id, bbox)
    
    def can_match(
        self,
        source_cam: str,
        source_track_id: int,
        target_cam: str,
        target_bbox: Tuple[float, float, float, float]
    ) -> bool:
        """
        Check if a cross-camera match is valid based on zones.
        
        Args:
            source_cam: Camera where track was last seen
            source_track_id: Track ID in source camera
            target_cam: Camera where candidate is detected
            target_bbox: Bounding box of candidate
            
        Returns:
            True if match is valid based on zone constraints
        """
        # Same camera - always valid
        if source_cam == target_cam:
            return True
        
        # Check exit zone constraint
        if self.require_exit_zone:
            if not self.was_in_exit_zone(source_cam, source_track_id):
                return False
        
        # Check entry zone constraint
        if self.require_entry_zone:
            if not self.is_in_entry_zone(target_cam, target_bbox):
                return False
        
        return True
    
    def filter_candidates(
        self,
        source_cam: str,
        source_track_id: int,
        candidates: List[Dict]
    ) -> List[Dict]:
        """
        Filter candidates based on zone constraints.
        
        Args:
            source_cam: Camera where track was lost
            source_track_id: Track ID in source camera
            candidates: List of candidate detections with 'cam_id' and 'bbox'
            
        Returns:
            Filtered list of candidates
        """
        valid = []
        for cand in candidates:
            if self.can_match(
                source_cam,
                source_track_id,
                cand['cam_id'],
                cand['bbox']
            ):
                valid.append(cand)
        return valid

