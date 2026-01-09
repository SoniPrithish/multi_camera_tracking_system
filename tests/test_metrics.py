"""
Tests for evaluation metrics.
"""

import pytest
import numpy as np

from src.evaluation.metrics import (
    compute_iou,
    match_detections,
    MOTACalculator,
    IDF1Calculator,
    HOTACalculator,
    MultiCameraMetrics
)


class TestComputeIoU:
    """Tests for IoU computation."""
    
    def test_identical_boxes(self):
        box = (0, 0, 100, 100)
        iou = compute_iou(box, box)
        assert abs(iou - 1.0) < 0.001
    
    def test_no_overlap(self):
        box1 = (0, 0, 50, 50)
        box2 = (100, 100, 50, 50)
        iou = compute_iou(box1, box2)
        assert iou == 0.0
    
    def test_partial_overlap(self):
        box1 = (0, 0, 100, 100)
        box2 = (50, 50, 100, 100)
        iou = compute_iou(box1, box2)
        assert 0 < iou < 1
    
    def test_contained_box(self):
        box1 = (0, 0, 100, 100)
        box2 = (25, 25, 50, 50)
        iou = compute_iou(box1, box2)
        # Smaller box is 25% of larger box
        # IoU = 2500 / (10000 + 2500 - 2500) = 0.25
        assert abs(iou - 0.25) < 0.001


class TestMatchDetections:
    """Tests for detection matching."""
    
    def test_empty_inputs(self):
        matches, unmatched_gt, unmatched_pred = match_detections([], [])
        assert matches == []
        assert unmatched_gt == []
        assert unmatched_pred == []
    
    def test_empty_gt(self):
        pred = [(0, 0, 50, 50)]
        matches, unmatched_gt, unmatched_pred = match_detections([], pred)
        assert matches == []
        assert unmatched_pred == [0]
    
    def test_empty_pred(self):
        gt = [(0, 0, 50, 50)]
        matches, unmatched_gt, unmatched_pred = match_detections(gt, [])
        assert matches == []
        assert unmatched_gt == [0]
    
    def test_perfect_match(self):
        gt = [(0, 0, 50, 50)]
        pred = [(0, 0, 50, 50)]
        matches, unmatched_gt, unmatched_pred = match_detections(gt, pred)
        
        assert len(matches) == 1
        assert matches[0] == (0, 0)
        assert unmatched_gt == []
        assert unmatched_pred == []
    
    def test_threshold(self):
        gt = [(0, 0, 50, 50)]
        pred = [(100, 100, 50, 50)]  # Far away
        
        matches, _, _ = match_detections(gt, pred, iou_threshold=0.5)
        assert len(matches) == 0


class TestMOTACalculator:
    """Tests for MOTA metric."""
    
    def test_init(self):
        calc = MOTACalculator()
        assert calc is not None
    
    def test_perfect_tracking(self):
        calc = MOTACalculator()
        
        gt_boxes = [(0, 0, 50, 50)]
        gt_ids = [1]
        pred_boxes = [(0, 0, 50, 50)]
        pred_ids = [1]
        
        calc.update(gt_boxes, gt_ids, pred_boxes, pred_ids)
        
        result = calc.compute()
        assert result['mota'] == 1.0
        assert result['precision'] == 1.0
        assert result['recall'] == 1.0
    
    def test_missed_detection(self):
        calc = MOTACalculator()
        
        gt_boxes = [(0, 0, 50, 50)]
        gt_ids = [1]
        pred_boxes = []
        pred_ids = []
        
        calc.update(gt_boxes, gt_ids, pred_boxes, pred_ids)
        
        result = calc.compute()
        assert result['fn'] == 1
        assert result['recall'] == 0.0
    
    def test_false_positive(self):
        calc = MOTACalculator()
        
        gt_boxes = []
        gt_ids = []
        pred_boxes = [(0, 0, 50, 50)]
        pred_ids = [1]
        
        calc.update(gt_boxes, gt_ids, pred_boxes, pred_ids)
        
        result = calc.compute()
        assert result['fp'] == 1
    
    def test_id_switch(self):
        calc = MOTACalculator()
        
        gt_boxes = [(0, 0, 50, 50)]
        gt_ids = [1]
        
        # First frame - track with ID 1
        calc.update(gt_boxes, gt_ids, gt_boxes, [1])
        
        # Second frame - same GT, but prediction has different ID
        calc.update(gt_boxes, gt_ids, gt_boxes, [2])
        
        result = calc.compute()
        assert result['idsw'] == 1


class TestIDF1Calculator:
    """Tests for IDF1 metric."""
    
    def test_init(self):
        calc = IDF1Calculator()
        assert calc is not None
    
    def test_perfect_tracking(self):
        calc = IDF1Calculator()
        
        gt_boxes = [(0, 0, 50, 50)]
        gt_ids = [1]
        pred_boxes = [(0, 0, 50, 50)]
        pred_ids = [1]
        
        # Multiple frames
        for _ in range(5):
            calc.update(gt_boxes, gt_ids, pred_boxes, pred_ids)
        
        result = calc.compute()
        assert result['idf1'] == 1.0
    
    def test_no_matches(self):
        calc = IDF1Calculator()
        
        gt_boxes = [(0, 0, 50, 50)]
        gt_ids = [1]
        pred_boxes = [(200, 200, 50, 50)]  # No overlap
        pred_ids = [1]
        
        calc.update(gt_boxes, gt_ids, pred_boxes, pred_ids)
        
        result = calc.compute()
        assert result['idf1'] == 0.0


class TestHOTACalculator:
    """Tests for HOTA metric."""
    
    def test_init(self):
        calc = HOTACalculator()
        assert calc is not None
    
    def test_compute(self):
        calc = HOTACalculator()
        
        gt_boxes = [(0, 0, 50, 50)]
        gt_ids = [1]
        pred_boxes = [(0, 0, 50, 50)]
        pred_ids = [1]
        
        calc.update(gt_boxes, gt_ids, pred_boxes, pred_ids)
        
        result = calc.compute()
        assert 'hota' in result
        assert 'deta' in result
        assert 'assa' in result


class TestMultiCameraMetrics:
    """Tests for multi-camera metrics."""
    
    def test_init(self):
        metrics = MultiCameraMetrics()
        assert metrics is not None
    
    def test_update_camera(self):
        metrics = MultiCameraMetrics()
        
        metrics.update_camera(
            'cam1',
            [(0, 0, 50, 50)],
            [1],
            [(0, 0, 50, 50)],
            [1]
        )
        
        result = metrics.compute()
        assert 'per_camera' in result
        assert 'cam1' in result['per_camera']
    
    def test_aggregate(self):
        metrics = MultiCameraMetrics()
        
        # Perfect tracking on two cameras
        for cam in ['cam1', 'cam2']:
            metrics.update_camera(
                cam,
                [(0, 0, 50, 50)],
                [1],
                [(0, 0, 50, 50)],
                [1]
            )
        
        result = metrics.compute()
        assert 'aggregate' in result
        assert result['aggregate']['mota'] == 1.0

