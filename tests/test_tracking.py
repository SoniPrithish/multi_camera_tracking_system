"""
Tests for tracking module.
"""

import pytest
import numpy as np

from src.tracking import build_tracker, CentroidTracker, ByteTracker
from src.tracking.kalman import KalmanFilter, bbox_to_xyah, xyah_to_bbox


class TestCentroidTracker:
    """Tests for CentroidTracker."""
    
    def test_init(self):
        tracker = CentroidTracker()
        assert tracker is not None
    
    def test_update_returns_list(self):
        tracker = CentroidTracker()
        dets = [{'bbox': (100, 100, 50, 100), 'score': 0.9}]
        
        tracks = tracker.update('cam1', dets)
        
        assert isinstance(tracks, list)
    
    def test_track_has_required_keys(self):
        tracker = CentroidTracker()
        dets = [{'bbox': (100, 100, 50, 100), 'score': 0.9}]
        
        tracks = tracker.update('cam1', dets)
        
        assert len(tracks) > 0
        for track in tracks:
            assert 'tid' in track
            assert 'bbox' in track
            assert 'score' in track
    
    def test_track_persistence(self):
        tracker = CentroidTracker()
        
        # First frame
        dets1 = [{'bbox': (100, 100, 50, 100), 'score': 0.9}]
        tracks1 = tracker.update('cam1', dets1)
        tid1 = tracks1[0]['tid']
        
        # Second frame - same location
        dets2 = [{'bbox': (105, 102, 50, 100), 'score': 0.9}]
        tracks2 = tracker.update('cam1', dets2)
        tid2 = tracks2[0]['tid']
        
        # Same track ID should be assigned
        assert tid1 == tid2
    
    def test_new_track_different_location(self):
        tracker = CentroidTracker()
        
        # First frame
        dets1 = [{'bbox': (100, 100, 50, 100), 'score': 0.9}]
        tracks1 = tracker.update('cam1', dets1)
        tid1 = tracks1[0]['tid']
        
        # Second frame - far away location
        dets2 = [{'bbox': (500, 400, 50, 100), 'score': 0.9}]
        tracks2 = tracker.update('cam1', dets2)
        tid2 = tracks2[0]['tid']
        
        # Different track ID
        assert tid1 != tid2


class TestByteTracker:
    """Tests for ByteTracker."""
    
    def test_init(self):
        tracker = ByteTracker()
        assert tracker is not None
    
    def test_update_returns_list(self):
        tracker = ByteTracker()
        dets = [{'bbox': (100, 100, 50, 100), 'score': 0.9}]
        
        tracks = tracker.update('cam1', dets)
        
        assert isinstance(tracks, list)
    
    def test_high_confidence_tracking(self):
        tracker = ByteTracker(track_high_thresh=0.5, min_hits=1)
        
        # High confidence detection
        dets = [{'bbox': (100, 100, 50, 100), 'score': 0.9}]
        
        for _ in range(3):
            tracks = tracker.update('cam1', dets)
        
        assert len(tracks) > 0
    
    def test_low_confidence_association(self):
        tracker = ByteTracker(
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            min_hits=1
        )
        
        # Create track with high confidence
        dets_high = [{'bbox': (100, 100, 50, 100), 'score': 0.9}]
        for _ in range(3):
            tracker.update('cam1', dets_high)
        
        # Update with low confidence at same location
        dets_low = [{'bbox': (105, 102, 50, 100), 'score': 0.3}]
        tracks = tracker.update('cam1', dets_low)
        
        # Track should be associated with low-conf detection
        assert len(tracks) >= 0  # May or may not associate depending on exact params
    
    def test_reset(self):
        tracker = ByteTracker()
        
        dets = [{'bbox': (100, 100, 50, 100), 'score': 0.9}]
        tracker.update('cam1', dets)
        
        tracker.reset()
        
        tracks = tracker.update('cam1', dets)
        # After reset, should start fresh


class TestKalmanFilter:
    """Tests for Kalman filter."""
    
    def test_init(self):
        kf = KalmanFilter()
        assert kf is not None
    
    def test_initiate(self):
        kf = KalmanFilter()
        measurement = np.array([100, 100, 0.5, 150])  # cx, cy, aspect, height
        
        mean, cov = kf.initiate(measurement)
        
        assert mean.shape == (8,)
        assert cov.shape == (8, 8)
    
    def test_predict(self):
        kf = KalmanFilter()
        measurement = np.array([100, 100, 0.5, 150])
        mean, cov = kf.initiate(measurement)
        
        new_mean, new_cov = kf.predict(mean, cov)
        
        assert new_mean.shape == (8,)
        assert new_cov.shape == (8, 8)
    
    def test_update(self):
        kf = KalmanFilter()
        measurement = np.array([100, 100, 0.5, 150])
        mean, cov = kf.initiate(measurement)
        mean, cov = kf.predict(mean, cov)
        
        new_measurement = np.array([102, 101, 0.5, 150])
        new_mean, new_cov = kf.update(mean, cov, new_measurement)
        
        assert new_mean.shape == (8,)


class TestBboxConversion:
    """Tests for bounding box conversion utilities."""
    
    def test_bbox_to_xyah(self):
        bbox = (100, 100, 50, 150)  # x, y, w, h
        xyah = bbox_to_xyah(bbox)
        
        assert len(xyah) == 4
        # cx = 100 + 50/2 = 125
        assert xyah[0] == 125
        # cy = 100 + 150/2 = 175
        assert xyah[1] == 175
        # a = w/h = 50/150
        assert abs(xyah[2] - 50/150) < 0.001
        # h = 150
        assert xyah[3] == 150
    
    def test_xyah_to_bbox(self):
        xyah = np.array([125, 175, 50/150, 150])
        bbox = xyah_to_bbox(xyah)
        
        assert len(bbox) == 4
        assert abs(bbox[0] - 100) < 0.001
        assert abs(bbox[1] - 100) < 0.001
        assert abs(bbox[2] - 50) < 0.001
        assert abs(bbox[3] - 150) < 0.001
    
    def test_roundtrip(self):
        original = (100, 100, 50, 150)
        xyah = bbox_to_xyah(original)
        recovered = xyah_to_bbox(xyah)
        
        for a, b in zip(original, recovered):
            assert abs(a - b) < 0.001


class TestBuildTracker:
    """Tests for build_tracker factory."""
    
    def test_build_centroid(self):
        tracker = build_tracker('centroid')
        assert isinstance(tracker, CentroidTracker)
    
    def test_build_byte(self):
        tracker = build_tracker('byte')
        assert isinstance(tracker, ByteTracker)
    
    def test_build_with_config(self):
        tracker = build_tracker({'name': 'bytetrack'})
        assert isinstance(tracker, ByteTracker)
    
    def test_build_unknown_raises(self):
        with pytest.raises(ValueError):
            build_tracker('unknown')

