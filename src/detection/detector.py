
import numpy as np
import cv2

class DummyDetector:
    """A fast CPU-only fake detector.
    Produces a few moving boxes so the pipeline is visually testable without ML deps.
    """
    def __init__(self):
        self.frame_no = 0
        self.traj = [
            {'x': 100, 'y': 150, 'vx': 1.5, 'vy': 0.6},
            {'x': 300, 'y': 200, 'vx': -1.2, 'vy': 1.0},
        ]

    def detect(self, frame):
        self.frame_no += 1
        H, W = frame.shape[:2]
        dets = []
        for i, t in enumerate(self.traj):
            t['x'] = (t['x'] + t['vx']) % W
            t['y'] = (t['y'] + t['vy']) % H
            w, h = 60, 120
            x1 = max(0, int(t['x'] - w/2))
            y1 = max(0, int(t['y'] - h/2))
            dets.append({'bbox': (x1, y1, w, h), 'score': 0.9, 'cls': 0})
        return dets
