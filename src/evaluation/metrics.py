"""
Multi-object tracking metrics: MOTA, IDF1, HOTA.
Computes standard MOT metrics for evaluation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def compute_iou(box1: Tuple, box2: Tuple) -> float:
    """
    Compute IoU between two bounding boxes.
    
    Args:
        box1: (x, y, w, h)
        box2: (x, y, w, h)
        
    Returns:
        IoU value between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to xyxy
    xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
    xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
    
    # Intersection
    xi1 = max(xa1, xb1)
    yi1 = max(ya1, yb1)
    xi2 = min(xa2, xb2)
    yi2 = min(ya2, yb2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter = (xi2 - xi1) * (yi2 - yi1)
    
    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


def match_detections(
    gt_boxes: List[Tuple],
    pred_boxes: List[Tuple],
    iou_threshold: float = 0.5
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match ground truth and predicted boxes using Hungarian algorithm.
    
    Returns:
        matches: List of (gt_idx, pred_idx) pairs
        unmatched_gt: Indices of unmatched ground truth
        unmatched_pred: Indices of unmatched predictions
    """
    if len(gt_boxes) == 0:
        return [], [], list(range(len(pred_boxes)))
    if len(pred_boxes) == 0:
        return [], list(range(len(gt_boxes))), []
    
    # Compute IoU matrix
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i, gt in enumerate(gt_boxes):
        for j, pred in enumerate(pred_boxes):
            iou_matrix[i, j] = compute_iou(gt, pred)
    
    # Hungarian matching
    try:
        from scipy.optimize import linear_sum_assignment
        # Convert to cost matrix (1 - IoU)
        cost_matrix = 1 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_gt = set(range(len(gt_boxes)))
        unmatched_pred = set(range(len(pred_boxes)))
        
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= iou_threshold:
                matches.append((r, c))
                unmatched_gt.discard(r)
                unmatched_pred.discard(c)
        
        return matches, list(unmatched_gt), list(unmatched_pred)
        
    except ImportError:
        # Greedy matching fallback
        matches = []
        used_gt = set()
        used_pred = set()
        
        # Sort by IoU descending
        pairs = []
        for i in range(len(gt_boxes)):
            for j in range(len(pred_boxes)):
                if iou_matrix[i, j] >= iou_threshold:
                    pairs.append((iou_matrix[i, j], i, j))
        
        pairs.sort(reverse=True)
        
        for iou, i, j in pairs:
            if i not in used_gt and j not in used_pred:
                matches.append((i, j))
                used_gt.add(i)
                used_pred.add(j)
        
        unmatched_gt = [i for i in range(len(gt_boxes)) if i not in used_gt]
        unmatched_pred = [j for j in range(len(pred_boxes)) if j not in used_pred]
        
        return matches, unmatched_gt, unmatched_pred


class MOTACalculator:
    """
    Calculates MOTA (Multiple Object Tracking Accuracy).
    
    MOTA = 1 - (FN + FP + IDSW) / GT
    
    Where:
    - FN: False Negatives (missed detections)
    - FP: False Positives (false alarms)
    - IDSW: ID Switches
    - GT: Total ground truth objects
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        self.total_gt = 0
        self.total_fn = 0
        self.total_fp = 0
        self.total_idsw = 0
        self.total_tp = 0
        
        # Track ID mapping for ID switch detection
        self.gt_to_pred_id: Dict[int, int] = {}
    
    def update(
        self,
        gt_boxes: List[Tuple],
        gt_ids: List[int],
        pred_boxes: List[Tuple],
        pred_ids: List[int]
    ):
        """
        Update metrics with one frame.
        
        Args:
            gt_boxes: List of (x, y, w, h) ground truth boxes
            gt_ids: List of ground truth track IDs
            pred_boxes: List of (x, y, w, h) predicted boxes
            pred_ids: List of predicted track IDs
        """
        # Match
        matches, unmatched_gt, unmatched_pred = match_detections(
            gt_boxes, pred_boxes, self.iou_threshold
        )
        
        # Count TP, FN, FP
        self.total_gt += len(gt_boxes)
        self.total_fn += len(unmatched_gt)
        self.total_fp += len(unmatched_pred)
        self.total_tp += len(matches)
        
        # Count ID switches
        for gt_idx, pred_idx in matches:
            gt_id = gt_ids[gt_idx]
            pred_id = pred_ids[pred_idx]
            
            if gt_id in self.gt_to_pred_id:
                if self.gt_to_pred_id[gt_id] != pred_id:
                    self.total_idsw += 1
            
            self.gt_to_pred_id[gt_id] = pred_id
    
    def compute(self) -> Dict[str, float]:
        """Compute MOTA and related metrics."""
        if self.total_gt == 0:
            return {'mota': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        mota = 1 - (self.total_fn + self.total_fp + self.total_idsw) / self.total_gt
        
        precision = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else 0
        recall = self.total_tp / self.total_gt if self.total_gt > 0 else 0
        
        return {
            'mota': mota,
            'precision': precision,
            'recall': recall,
            'fn': self.total_fn,
            'fp': self.total_fp,
            'idsw': self.total_idsw,
            'tp': self.total_tp,
            'gt': self.total_gt
        }


class IDF1Calculator:
    """
    Calculates IDF1 (ID F1 Score).
    
    IDF1 = 2 * IDTP / (2 * IDTP + IDFN + IDFP)
    
    Measures the ratio of correctly identified detections.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        # Per-ID counts
        self.gt_id_frames: Dict[int, int] = defaultdict(int)  # GT ID -> frame count
        self.pred_id_frames: Dict[int, int] = defaultdict(int)  # Pred ID -> frame count
        self.match_counts: Dict[Tuple[int, int], int] = defaultdict(int)  # (GT ID, Pred ID) -> count
    
    def update(
        self,
        gt_boxes: List[Tuple],
        gt_ids: List[int],
        pred_boxes: List[Tuple],
        pred_ids: List[int]
    ):
        """Update with one frame."""
        # Count frames per ID
        for gt_id in gt_ids:
            self.gt_id_frames[gt_id] += 1
        for pred_id in pred_ids:
            self.pred_id_frames[pred_id] += 1
        
        # Match and count
        matches, _, _ = match_detections(gt_boxes, pred_boxes, self.iou_threshold)
        
        for gt_idx, pred_idx in matches:
            gt_id = gt_ids[gt_idx]
            pred_id = pred_ids[pred_idx]
            self.match_counts[(gt_id, pred_id)] += 1
    
    def compute(self) -> Dict[str, float]:
        """Compute IDF1 and related metrics."""
        if len(self.gt_id_frames) == 0:
            return {'idf1': 0.0, 'idp': 0.0, 'idr': 0.0}
        
        # Find best matching between GT and Pred IDs
        # Use Hungarian matching on ID overlap
        gt_ids = list(self.gt_id_frames.keys())
        pred_ids = list(self.pred_id_frames.keys())
        
        if len(pred_ids) == 0:
            return {'idf1': 0.0, 'idp': 0.0, 'idr': 0.0}
        
        # Build cost matrix based on match counts
        cost_matrix = np.zeros((len(gt_ids), len(pred_ids)))
        for i, gt_id in enumerate(gt_ids):
            for j, pred_id in enumerate(pred_ids):
                cost_matrix[i, j] = -self.match_counts.get((gt_id, pred_id), 0)
        
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            
            idtp = 0
            for r, c in zip(row_ind, col_ind):
                gt_id = gt_ids[r]
                pred_id = pred_ids[c]
                idtp += self.match_counts.get((gt_id, pred_id), 0)
        except ImportError:
            # Simple greedy matching
            idtp = 0
            used = set()
            for gt_id in gt_ids:
                best_match = 0
                best_pred = None
                for pred_id in pred_ids:
                    if pred_id not in used:
                        count = self.match_counts.get((gt_id, pred_id), 0)
                        if count > best_match:
                            best_match = count
                            best_pred = pred_id
                if best_pred is not None:
                    idtp += best_match
                    used.add(best_pred)
        
        idfn = sum(self.gt_id_frames.values()) - idtp
        idfp = sum(self.pred_id_frames.values()) - idtp
        
        idf1 = 2 * idtp / (2 * idtp + idfn + idfp) if (2 * idtp + idfn + idfp) > 0 else 0
        idp = idtp / (idtp + idfp) if (idtp + idfp) > 0 else 0
        idr = idtp / (idtp + idfn) if (idtp + idfn) > 0 else 0
        
        return {
            'idf1': idf1,
            'idp': idp,
            'idr': idr,
            'idtp': idtp,
            'idfn': idfn,
            'idfp': idfp
        }


class HOTACalculator:
    """
    Calculates HOTA (Higher Order Tracking Accuracy).
    
    HOTA = sqrt(DetA * AssA)
    
    Where:
    - DetA: Detection Accuracy
    - AssA: Association Accuracy
    """
    
    def __init__(self, iou_thresholds: Optional[List[float]] = None):
        self.iou_thresholds = iou_thresholds or [0.5]
        self.reset()
    
    def reset(self):
        self.frames_data = []
    
    def update(
        self,
        gt_boxes: List[Tuple],
        gt_ids: List[int],
        pred_boxes: List[Tuple],
        pred_ids: List[int]
    ):
        """Store frame data for later computation."""
        self.frames_data.append({
            'gt_boxes': gt_boxes,
            'gt_ids': gt_ids,
            'pred_boxes': pred_boxes,
            'pred_ids': pred_ids
        })
    
    def compute(self) -> Dict[str, float]:
        """Compute HOTA and related metrics."""
        if len(self.frames_data) == 0:
            return {'hota': 0.0, 'deta': 0.0, 'assa': 0.0}
        
        hota_scores = []
        deta_scores = []
        assa_scores = []
        
        for iou_thresh in self.iou_thresholds:
            deta, assa = self._compute_at_threshold(iou_thresh)
            hota = np.sqrt(deta * assa) if deta > 0 and assa > 0 else 0
            
            hota_scores.append(hota)
            deta_scores.append(deta)
            assa_scores.append(assa)
        
        return {
            'hota': np.mean(hota_scores),
            'deta': np.mean(deta_scores),
            'assa': np.mean(assa_scores),
            'hota_per_thresh': dict(zip(self.iou_thresholds, hota_scores))
        }
    
    def _compute_at_threshold(self, iou_thresh: float) -> Tuple[float, float]:
        """Compute DetA and AssA at a specific IoU threshold."""
        total_tp = 0
        total_fn = 0
        total_fp = 0
        
        # For association accuracy
        gt_id_matches: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        pred_id_matches: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        
        for frame in self.frames_data:
            matches, unmatched_gt, unmatched_pred = match_detections(
                frame['gt_boxes'],
                frame['pred_boxes'],
                iou_thresh
            )
            
            total_tp += len(matches)
            total_fn += len(unmatched_gt)
            total_fp += len(unmatched_pred)
            
            for gt_idx, pred_idx in matches:
                gt_id = frame['gt_ids'][gt_idx]
                pred_id = frame['pred_ids'][pred_idx]
                gt_id_matches[gt_id][pred_id] += 1
                pred_id_matches[pred_id][gt_id] += 1
        
        # Detection Accuracy
        deta = total_tp / (total_tp + total_fn + total_fp) if (total_tp + total_fn + total_fp) > 0 else 0
        
        # Association Accuracy
        if total_tp == 0:
            assa = 0
        else:
            tpa = 0
            fna = 0
            fpa = 0
            
            for gt_id, pred_counts in gt_id_matches.items():
                if len(pred_counts) == 0:
                    continue
                
                # Best matching pred ID
                best_pred_id = max(pred_counts.keys(), key=lambda x: pred_counts[x])
                match_count = pred_counts[best_pred_id]
                
                gt_total = sum(pred_counts.values())
                pred_total = sum(pred_id_matches[best_pred_id].values())
                
                tpa += match_count
                fna += gt_total - match_count
                fpa += pred_total - match_count
            
            assa = tpa / (tpa + fna + fpa) if (tpa + fna + fpa) > 0 else 0
        
        return deta, assa


class MultiCameraMetrics:
    """
    Computes metrics for multi-camera tracking.
    Includes cross-camera handoff accuracy.
    """
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.reset()
    
    def reset(self):
        self.per_camera_mota = {}
        self.per_camera_idf1 = {}
        
        self.total_handoffs_gt = 0
        self.correct_handoffs = 0
        self.missed_handoffs = 0
        self.false_handoffs = 0
    
    def update_camera(
        self,
        camera_id: str,
        gt_boxes: List[Tuple],
        gt_ids: List[int],
        pred_boxes: List[Tuple],
        pred_ids: List[int]
    ):
        """Update metrics for a single camera."""
        if camera_id not in self.per_camera_mota:
            self.per_camera_mota[camera_id] = MOTACalculator(self.iou_threshold)
            self.per_camera_idf1[camera_id] = IDF1Calculator(self.iou_threshold)
        
        self.per_camera_mota[camera_id].update(gt_boxes, gt_ids, pred_boxes, pred_ids)
        self.per_camera_idf1[camera_id].update(gt_boxes, gt_ids, pred_boxes, pred_ids)
    
    def update_handoff(
        self,
        gt_source_cam: str,
        gt_target_cam: str,
        gt_global_id: int,
        pred_source_cam: Optional[str],
        pred_target_cam: Optional[str],
        pred_global_id: Optional[int]
    ):
        """Update handoff metrics."""
        self.total_handoffs_gt += 1
        
        if pred_global_id is not None and pred_source_cam == gt_source_cam and pred_target_cam == gt_target_cam:
            if pred_global_id == gt_global_id:
                self.correct_handoffs += 1
            else:
                self.false_handoffs += 1
        else:
            self.missed_handoffs += 1
    
    def compute(self) -> Dict[str, Any]:
        """Compute all metrics."""
        results = {
            'per_camera': {},
            'aggregate': {},
            'cross_camera': {}
        }
        
        # Per-camera metrics
        for cam_id in self.per_camera_mota.keys():
            mota_metrics = self.per_camera_mota[cam_id].compute()
            idf1_metrics = self.per_camera_idf1[cam_id].compute()
            
            results['per_camera'][cam_id] = {
                **mota_metrics,
                **idf1_metrics
            }
        
        # Aggregate metrics
        if len(results['per_camera']) > 0:
            results['aggregate']['mota'] = np.mean([
                m.get('mota', 0) for m in results['per_camera'].values()
            ])
            results['aggregate']['idf1'] = np.mean([
                m.get('idf1', 0) for m in results['per_camera'].values()
            ])
        
        # Cross-camera handoff metrics
        if self.total_handoffs_gt > 0:
            results['cross_camera'] = {
                'handoff_precision': self.correct_handoffs / (self.correct_handoffs + self.false_handoffs) if (self.correct_handoffs + self.false_handoffs) > 0 else 0,
                'handoff_recall': self.correct_handoffs / self.total_handoffs_gt,
                'total_gt_handoffs': self.total_handoffs_gt,
                'correct_handoffs': self.correct_handoffs,
                'missed_handoffs': self.missed_handoffs,
                'false_handoffs': self.false_handoffs
            }
        
        return results

