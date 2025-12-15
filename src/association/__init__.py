"""
Cross-camera association module.
Handles identity matching across multiple camera views.
"""

from .camera_graph import CameraGraph, CameraTransition, CameraNode
from .zone_gate import Zone, ZoneManager, ZoneGate
from .matcher import CrossCameraMatcher, AppearanceMatcher, MatchCandidate
from .global_registry import GlobalIdentity, GlobalRegistry


__all__ = [
    # Camera graph
    'CameraGraph',
    'CameraTransition',
    'CameraNode',
    # Zones
    'Zone',
    'ZoneManager',
    'ZoneGate',
    # Matching
    'CrossCameraMatcher',
    'AppearanceMatcher',
    'MatchCandidate',
    # Registry
    'GlobalIdentity',
    'GlobalRegistry',
]

