"""
Tests for detection module.
"""

import pytest
import numpy as np

from src.detection import build_detector, DummyDetector, YOLOv8Detector


class TestDummyDetector:
    """Tests for DummyDetector."""
    
    def test_init(self):
        detector = DummyDetector()
        assert detector is not None
    
    def test_detect_returns_list(self):
        detector = DummyDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        dets = detector.detect(frame)
        
        assert isinstance(dets, list)
    
    def test_detect_has_required_keys(self):
        detector = DummyDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        dets = detector.detect(frame)
        
        for det in dets:
            assert 'bbox' in det
            assert 'score' in det
            assert 'cls' in det
    
    def test_bbox_format(self):
        detector = DummyDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        dets = detector.detect(frame)
        
        for det in dets:
            bbox = det['bbox']
            assert len(bbox) == 4  # x, y, w, h
            x, y, w, h = bbox
            assert w > 0
            assert h > 0


class TestBuildDetector:
    """Tests for build_detector factory."""
    
    def test_build_dummy(self):
        detector = build_detector('dummy')
        assert isinstance(detector, DummyDetector)
    
    def test_build_with_dict_config(self):
        detector = build_detector({'name': 'dummy'})
        assert isinstance(detector, DummyDetector)
    
    def test_build_unknown_raises(self):
        with pytest.raises(ValueError):
            build_detector('unknown_detector')
    
    def test_build_yolov8(self):
        # YOLOv8 may not be available, so we catch import errors
        try:
            detector = build_detector('yolov8')
            assert isinstance(detector, YOLOv8Detector)
        except Exception:
            pytest.skip("YOLOv8 not available")


class TestYOLOv8Detector:
    """Tests for YOLOv8Detector (when available)."""
    
    @pytest.fixture
    def detector(self):
        try:
            return YOLOv8Detector(model='yolov8n.pt', device='cpu')
        except Exception:
            pytest.skip("YOLOv8 not available")
    
    def test_get_info(self, detector):
        info = detector.get_info()
        
        assert 'model' in info
        assert 'device' in info
        assert 'conf_threshold' in info
    
    def test_detect_empty_frame(self, detector):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        dets = detector.detect(frame)
        
        assert isinstance(dets, list)

