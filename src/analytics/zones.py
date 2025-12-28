"""
Zone definitions and management for analytics.
Extends the association zone module with analytics-specific functionality.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnalyticsZone:
    """Zone with analytics capabilities."""
    zone_id: str
    camera_id: str
    name: str
    polygon: np.ndarray
    zone_type: str = "general"  # "roi", "entry", "exit", "dwell"
    
    # Analytics configuration
    track_dwell: bool = True
    dwell_threshold: float = 5.0  # Minimum dwell time to report (seconds)
    
    def __post_init__(self):
        self.polygon = np.array(self.polygon, dtype=np.float32)
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside polygon."""
        n = len(self.polygon)
        inside = False
        
        j = n - 1
        for i in range(n):
            xi, yi = self.polygon[i]
            xj, yj = self.polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-10) + xi):
                inside = not inside
            j = i
        
        return inside
    
    def contains_bbox(self, bbox: Tuple[float, float, float, float]) -> bool:
        """Check if bbox center is inside zone."""
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2
        return self.contains_point(cx, cy)


@dataclass
class CountingLine:
    """Line for counting people crossing."""
    line_id: str
    camera_id: str
    name: str
    start_point: Tuple[float, float]
    end_point: Tuple[float, float]
    
    # Direction reference: positive is left-to-right or top-to-bottom
    positive_direction: str = "in"  # "in" or "out"
    
    def __post_init__(self):
        self.start = np.array(self.start_point, dtype=np.float32)
        self.end = np.array(self.end_point, dtype=np.float32)
        
        # Compute line vector and normal
        self.direction = self.end - self.start
        self.length = np.linalg.norm(self.direction)
        if self.length > 0:
            self.direction = self.direction / self.length
        
        # Normal vector (perpendicular, points to "positive" side)
        self.normal = np.array([-self.direction[1], self.direction[0]])
    
    def get_side(self, point: Tuple[float, float]) -> int:
        """
        Get which side of the line a point is on.
        Returns: 1 for positive side, -1 for negative side, 0 if on line
        """
        p = np.array(point, dtype=np.float32)
        v = p - self.start
        cross = v[0] * self.normal[0] + v[1] * self.normal[1]
        
        if cross > 0.5:
            return 1
        elif cross < -0.5:
            return -1
        return 0
    
    def check_crossing(
        self,
        prev_point: Tuple[float, float],
        curr_point: Tuple[float, float]
    ) -> Optional[str]:
        """
        Check if movement crosses the line.
        
        Returns:
            "positive" if crossed from negative to positive side
            "negative" if crossed from positive to negative side
            None if no crossing
        """
        prev_side = self.get_side(prev_point)
        curr_side = self.get_side(curr_point)
        
        if prev_side == 0 or curr_side == 0:
            return None
        
        if prev_side != curr_side:
            if curr_side > 0:
                return "positive"
            else:
                return "negative"
        
        return None


class AnalyticsZoneManager:
    """
    Manages zones and counting lines for analytics.
    """
    
    def __init__(self):
        self.zones: Dict[str, AnalyticsZone] = {}
        self.lines: Dict[str, CountingLine] = {}
        self.camera_zones: Dict[str, List[str]] = {}
        self.camera_lines: Dict[str, List[str]] = {}
    
    def add_zone(
        self,
        zone_id: str,
        camera_id: str,
        name: str,
        polygon: List[Tuple[float, float]],
        zone_type: str = "general",
        track_dwell: bool = True,
        dwell_threshold: float = 5.0
    ):
        """Add an analytics zone."""
        zone = AnalyticsZone(
            zone_id=zone_id,
            camera_id=camera_id,
            name=name,
            polygon=np.array(polygon),
            zone_type=zone_type,
            track_dwell=track_dwell,
            dwell_threshold=dwell_threshold
        )
        
        self.zones[zone_id] = zone
        
        if camera_id not in self.camera_zones:
            self.camera_zones[camera_id] = []
        self.camera_zones[camera_id].append(zone_id)
        
        logger.debug(f"Added analytics zone: {zone_id} to camera {camera_id}")
    
    def add_line(
        self,
        line_id: str,
        camera_id: str,
        name: str,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        positive_direction: str = "in"
    ):
        """Add a counting line."""
        line = CountingLine(
            line_id=line_id,
            camera_id=camera_id,
            name=name,
            start_point=start_point,
            end_point=end_point,
            positive_direction=positive_direction
        )
        
        self.lines[line_id] = line
        
        if camera_id not in self.camera_lines:
            self.camera_lines[camera_id] = []
        self.camera_lines[camera_id].append(line_id)
        
        logger.debug(f"Added counting line: {line_id} to camera {camera_id}")
    
    def get_zones_for_camera(self, camera_id: str) -> List[AnalyticsZone]:
        """Get all zones for a camera."""
        zone_ids = self.camera_zones.get(camera_id, [])
        return [self.zones[zid] for zid in zone_ids]
    
    def get_lines_for_camera(self, camera_id: str) -> List[CountingLine]:
        """Get all counting lines for a camera."""
        line_ids = self.camera_lines.get(camera_id, [])
        return [self.lines[lid] for lid in line_ids]
    
    def get_zones_containing(
        self,
        camera_id: str,
        bbox: Tuple[float, float, float, float]
    ) -> List[AnalyticsZone]:
        """Get zones containing a bounding box."""
        zones = []
        for zone in self.get_zones_for_camera(camera_id):
            if zone.contains_bbox(bbox):
                zones.append(zone)
        return zones
    
    @classmethod
    def from_config(cls, config: Dict) -> 'AnalyticsZoneManager':
        """Create from configuration."""
        manager = cls()
        
        for zone_cfg in config.get('zones', []):
            manager.add_zone(
                zone_id=zone_cfg['id'],
                camera_id=zone_cfg['camera_id'],
                name=zone_cfg.get('name', zone_cfg['id']),
                polygon=zone_cfg['polygon'],
                zone_type=zone_cfg.get('type', 'general'),
                track_dwell=zone_cfg.get('track_dwell', True),
                dwell_threshold=zone_cfg.get('dwell_threshold', 5.0)
            )
        
        for line_cfg in config.get('lines', []):
            manager.add_line(
                line_id=line_cfg['id'],
                camera_id=line_cfg['camera_id'],
                name=line_cfg.get('name', line_cfg['id']),
                start_point=tuple(line_cfg['start']),
                end_point=tuple(line_cfg['end']),
                positive_direction=line_cfg.get('positive_direction', 'in')
            )
        
        return manager

