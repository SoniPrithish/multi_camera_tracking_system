"""
Detection module for multi-camera tracking system.
Supports multiple detector backends: YOLOv8, ONNX, and a dummy detector for testing.
"""

from .detector import DummyDetector
from .yolov8 import YOLOv8Detector
from .onnx_detector import ONNXDetector


def build_detector(config):
    """
    Build a detector based on configuration.
    
    Args:
        config: Either a string (detector name) or dict with 'name' and params
        
    Returns:
        Detector instance
    """
    if isinstance(config, str):
        name = config.lower()
        params = {}
    else:
        name = config.get('name', 'dummy').lower()
        params = {k: v for k, v in config.items() if k != 'name'}
    
    if name == 'dummy':
        return DummyDetector()
    elif name == 'yolov8':
        return YOLOv8Detector(**params)
    elif name == 'onnx':
        if 'model_path' not in params:
            raise ValueError("ONNX detector requires 'model_path' parameter")
        return ONNXDetector(**params)
    else:
        raise ValueError(f'Unknown detector: {name}')


__all__ = ['DummyDetector', 'YOLOv8Detector', 'ONNXDetector', 'build_detector']
