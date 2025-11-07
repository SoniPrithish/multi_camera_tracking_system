
from .detector import DummyDetector
from .yolov8 import YOLOv8Detector

def build_detector(name: str):
    name = (name or 'dummy').lower()
    if name == 'dummy':
        return DummyDetector()
    elif name == 'yolov8':
        return YOLOv8Detector()  # stub; implement if you add ultralytics
    else:
        raise ValueError(f'Unknown detector: {name}')
