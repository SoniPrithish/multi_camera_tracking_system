"""
ONNX-based detector for CPU inference acceleration.
Provides a lightweight alternative when GPU is not available.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ONNXDetector:
    """
    ONNX Runtime-based detector for efficient CPU inference.
    Compatible with exported YOLOv8 ONNX models.
    """
    
    PERSON_CLASS_ID = 0
    
    def __init__(
        self,
        model_path: str,
        conf: float = 0.25,
        iou: float = 0.45,
        img_size: int = 640,
        classes: Optional[List[int]] = None,
        use_gpu: bool = False
    ):
        """
        Initialize ONNX detector.
        
        Args:
            model_path: Path to ONNX model file
            conf: Confidence threshold
            iou: IoU threshold for NMS
            img_size: Input image size
            classes: Class IDs to detect (None = person only)
            use_gpu: Whether to use GPU (if available)
        """
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.classes = classes if classes is not None else [self.PERSON_CLASS_ID]
        
        self._load_model(use_gpu)
        
    def _load_model(self, use_gpu: bool):
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            
            # Set up execution providers
            providers = []
            if use_gpu:
                providers.append('CUDAExecutionProvider')
            providers.append('CPUExecutionProvider')
            
            # Session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 4
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [o.name for o in self.session.get_outputs()]
            
            logger.info(f"ONNX detector loaded: {self.model_path}")
            logger.info(f"Input: {self.input_name}, shape: {self.input_shape}")
            
        except ImportError:
            raise ImportError("onnxruntime is required for ONNX inference")
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model: {e}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in frame.
        
        Args:
            frame: BGR image (H, W, 3)
            
        Returns:
            List of detections with bbox, score, cls
        """
        # Preprocess
        img, ratio, pad = self._preprocess(frame)
        
        # Inference
        outputs = self.session.run(self.output_names, {self.input_name: img})
        
        # Postprocess
        detections = self._postprocess(outputs[0], ratio, pad, frame.shape[:2])
        
        return detections
    
    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox resize and normalize."""
        h, w = frame.shape[:2]
        
        # Calculate scale
        scale = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size (center padding)
        pad_h = (self.img_size - new_h) // 2
        pad_w = (self.img_size - new_w) // 2
        
        padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        # Normalize and transpose
        img = padded.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[np.newaxis, ...]  # BHWC -> BCHW
        
        return img, scale, (pad_w, pad_h)
    
    def _postprocess(
        self,
        output: np.ndarray,
        scale: float,
        pad: Tuple[int, int],
        orig_hw: Tuple[int, int]
    ) -> List[Dict]:
        """Process YOLO output to get detections."""
        # Handle different output formats
        if output.ndim == 3:
            # Shape: (1, 84, N) -> (N, 84)
            output = output[0].T
        
        boxes = output[:, :4]  # cx, cy, w, h
        class_scores = output[:, 4:]
        
        detections = []
        
        for i in range(len(boxes)):
            scores = class_scores[i]
            max_score = np.max(scores)
            cls_id = np.argmax(scores)
            
            if max_score < self.conf or cls_id not in self.classes:
                continue
            
            cx, cy, w, h = boxes[i]
            
            # Convert to corner format and unpad/unscale
            x1 = (cx - w / 2 - pad[0]) / scale
            y1 = (cy - h / 2 - pad[1]) / scale
            w = w / scale
            h = h / scale
            
            # Clip to image bounds
            x1 = np.clip(x1, 0, orig_hw[1])
            y1 = np.clip(y1, 0, orig_hw[0])
            w = np.clip(w, 0, orig_hw[1] - x1)
            h = np.clip(h, 0, orig_hw[0] - y1)
            
            detections.append({
                'bbox': (float(x1), float(y1), float(w), float(h)),
                'score': float(max_score),
                'cls': int(cls_id)
            })
        
        # Apply NMS
        return self._nms(detections)
    
    def _nms(self, detections: List[Dict]) -> List[Dict]:
        """Non-maximum suppression."""
        if len(detections) == 0:
            return []
        
        boxes = np.array([[d['bbox'][0], d['bbox'][1], 
                          d['bbox'][0] + d['bbox'][2], 
                          d['bbox'][1] + d['bbox'][3]] for d in detections])
        scores = np.array([d['score'] for d in detections])
        
        x1, y1, x2, y2 = boxes.T
        areas = (x2 - x1) * (y2 - y1)
        
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            mask = iou <= self.iou
            order = order[1:][mask]
        
        return [detections[i] for i in keep]
    
    def get_info(self) -> Dict:
        """Get detector info."""
        return {
            'model': self.model_path,
            'backend': 'onnxruntime',
            'conf': self.conf,
            'iou': self.iou,
            'img_size': self.img_size,
            'classes': self.classes
        }

