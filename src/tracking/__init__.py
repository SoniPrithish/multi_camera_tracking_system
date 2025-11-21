"""
Tracking module for multi-camera tracking system.
Provides single-camera object tracking with motion prediction.
"""

from .centroid import CentroidTracker
from .byte_tracker import ByteTracker, STrack, TrackState
from .kalman import KalmanFilter, bbox_to_xyah, xyah_to_bbox


def build_tracker(config):
    """
    Build a tracker based on configuration.
    
    Args:
        config: Either a string (tracker name) or dict with 'name' and params
        
    Returns:
        Tracker instance
    """
    if isinstance(config, str):
        name = config.lower()
        params = {}
    else:
        name = config.get('name', 'centroid').lower()
        params = {k: v for k, v in config.items() if k != 'name'}
    
    if name == 'centroid':
        return CentroidTracker(**params)
    elif name in ('byte', 'bytetrack'):
        return ByteTracker(**params)
    else:
        raise ValueError(f'Unknown tracker: {name}')


__all__ = [
    'CentroidTracker', 
    'ByteTracker', 
    'STrack', 
    'TrackState',
    'KalmanFilter',
    'bbox_to_xyah',
    'xyah_to_bbox',
    'build_tracker'
]
