"""
YOLOv8 Person Detector with ONNX export support.
Supports GPU acceleration with CPU fallback.
"""

import os
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class YOLOv8Detector:
    """
    YOLOv8-based person detector using Ultralytics.
    Supports multiple model sizes and ONNX export for CPU acceleration.
    """
    
    PERSON_CLASS_ID = 0  # COCO class ID for person
    
    def __init__(
        self,
        model: str = 'yolov8n.pt',
        device: str = 'auto',
        conf: float = 0.25,
        iou: float = 0.45,
        img_size: int = 640,
        half: bool = False,
        classes: Optional[List[int]] = None,
    ):
        """
        Initialize YOLOv8 detector.
        
        Args:
            model: Model name or path (yolov8n/s/m/l/x.pt or .onnx)
            device: Device to run on ('auto', 'cpu', 'cuda', '0', '1', etc.)
            conf: Confidence threshold for detections
            iou: IoU threshold for NMS
            img_size: Input image size
            half: Use FP16 inference (GPU only)
            classes: List of class IDs to detect (None = person only)
        """
        self.model_name = model
        self.conf = conf
        self.iou = iou
        self.img_size = img_size
        self.half = half
        self.classes = classes if classes is not None else [self.PERSON_CLASS_ID]
        
        # Determine device
        self.device = self._resolve_device(device)
        
        # Load model
        self.model = None
        self.onnx_session = None
        self._load_model()
        
        logger.info(f"YOLOv8 detector initialized: model={model}, device={self.device}")
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string to actual device."""
        if device == 'auto':
            try:
                import torch
                if torch.cuda.is_available():
                    return 'cuda'
            except ImportError:
                pass
            return 'cpu'
        return device
    
    def _load_model(self):
        """Load the YOLO model (Ultralytics or ONNX)."""
        if self.model_name.endswith('.onnx'):
            self._load_onnx_model()
        else:
            self._load_ultralytics_model()
    
    def _load_ultralytics_model(self):
        """Load model using Ultralytics library."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            
            # Warmup
            dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            _ = self.model.predict(dummy, verbose=False)
            
            logger.info(f"Loaded Ultralytics model: {self.model_name}")
        except ImportError:
            logger.warning("Ultralytics not installed. Falling back to ONNX or dummy mode.")
            self._try_onnx_fallback()
        except Exception as e:
            logger.error(f"Failed to load Ultralytics model: {e}")
            self._try_onnx_fallback()
    
    def _try_onnx_fallback(self):
        """Try to load ONNX version of the model."""
        onnx_path = self.model_name.replace('.pt', '.onnx')
        if os.path.exists(onnx_path):
            self.model_name = onnx_path
            self._load_onnx_model()
        else:
            logger.warning(f"No ONNX model at {onnx_path}. Detector will return empty results.")
    
    def _load_onnx_model(self):
        """Load ONNX model using onnxruntime."""
        try:
            import onnxruntime as ort
            
            providers = ['CPUExecutionProvider']
            if self.device != 'cpu':
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            self.onnx_session = ort.InferenceSession(
                self.model_name,
                providers=providers
            )
            
            # Get input details
            self.onnx_input_name = self.onnx_session.get_inputs()[0].name
            self.onnx_input_shape = self.onnx_session.get_inputs()[0].shape
            
            logger.info(f"Loaded ONNX model: {self.model_name}")
        except ImportError:
            logger.error("onnxruntime not installed. Cannot load ONNX model.")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
    
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect persons in frame.
        
        Args:
            frame: BGR image as numpy array (H, W, 3)
            
        Returns:
            List of detections, each with keys:
                - bbox: (x, y, w, h) in pixel coordinates
                - score: confidence score
                - cls: class ID (0 for person)
        """
        if self.model is not None:
            return self._detect_ultralytics(frame)
        elif self.onnx_session is not None:
            return self._detect_onnx(frame)
        else:
            return []
    
    def _detect_ultralytics(self, frame: np.ndarray) -> List[Dict]:
        """Run detection using Ultralytics YOLO."""
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.img_size,
            classes=self.classes,
            device=self.device,
            half=self.half,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())
                
                x1, y1, x2, y2 = xyxy
                w, h = x2 - x1, y2 - y1
                
                detections.append({
                    'bbox': (float(x1), float(y1), float(w), float(h)),
                    'score': conf,
                    'cls': cls
                })
        
        return detections
    
    def _detect_onnx(self, frame: np.ndarray) -> List[Dict]:
        """Run detection using ONNX Runtime."""
        # Preprocess
        img, ratio, (pad_w, pad_h) = self._preprocess_onnx(frame)
        
        # Run inference
        outputs = self.onnx_session.run(None, {self.onnx_input_name: img})
        
        # Postprocess
        detections = self._postprocess_onnx(outputs[0], ratio, pad_w, pad_h, frame.shape)
        
        return detections
    
    def _preprocess_onnx(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess frame for ONNX inference."""
        h, w = frame.shape[:2]
        
        # Calculate resize ratio
        ratio = min(self.img_size / h, self.img_size / w)
        new_h, new_w = int(h * ratio), int(w * ratio)
        
        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to square
        pad_h = (self.img_size - new_h) // 2
        pad_w = (self.img_size - new_w) // 2
        
        padded = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        # Convert to float and normalize
        img = padded.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, 0)  # Add batch dimension
        
        return img, ratio, (pad_w, pad_h)
    
    def _postprocess_onnx(
        self,
        output: np.ndarray,
        ratio: float,
        pad_w: int,
        pad_h: int,
        orig_shape: Tuple[int, ...]
    ) -> List[Dict]:
        """Postprocess ONNX output to get detections."""
        # YOLOv8 output shape: (1, 84, 8400) for 80 classes
        # Transpose to (8400, 84)
        predictions = output[0].T
        
        # Get boxes and scores
        boxes = predictions[:, :4]  # x_center, y_center, width, height
        scores = predictions[:, 4:]  # class scores
        
        detections = []
        
        for i, box in enumerate(boxes):
            # Get max class score
            class_scores = scores[i]
            max_score = np.max(class_scores)
            cls_id = np.argmax(class_scores)
            
            # Filter by confidence and class
            if max_score < self.conf or cls_id not in self.classes:
                continue
            
            # Convert from center to corner format
            x_center, y_center, w, h = box
            x1 = x_center - w / 2
            y1 = y_center - h / 2
            
            # Remove padding and scale back
            x1 = (x1 - pad_w) / ratio
            y1 = (y1 - pad_h) / ratio
            w = w / ratio
            h = h / ratio
            
            # Clip to image bounds
            x1 = max(0, min(x1, orig_shape[1]))
            y1 = max(0, min(y1, orig_shape[0]))
            w = min(w, orig_shape[1] - x1)
            h = min(h, orig_shape[0] - y1)
            
            detections.append({
                'bbox': (float(x1), float(y1), float(w), float(h)),
                'score': float(max_score),
                'cls': int(cls_id)
            })
        
        # Apply NMS
        detections = self._nms(detections)
        
        return detections
    
    def _nms(self, detections: List[Dict]) -> List[Dict]:
        """Apply non-maximum suppression."""
        if len(detections) == 0:
            return []
        
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['score'] for d in detections])
        
        # Convert to xyxy format for NMS
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= self.iou)[0]
            order = order[inds + 1]
        
        return [detections[i] for i in keep]
    
    def export_onnx(self, output_path: Optional[str] = None) -> str:
        """
        Export model to ONNX format for CPU inference.
        
        Args:
            output_path: Path to save ONNX model (default: same as model with .onnx ext)
            
        Returns:
            Path to exported ONNX model
        """
        if self.model is None:
            raise RuntimeError("Cannot export: Ultralytics model not loaded")
        
        if output_path is None:
            output_path = self.model_name.replace('.pt', '.onnx')
        
        self.model.export(
            format='onnx',
            imgsz=self.img_size,
            half=False,
            dynamic=False,
            simplify=True
        )
        
        logger.info(f"Exported ONNX model to: {output_path}")
        return output_path
    
    def get_info(self) -> Dict:
        """Get detector information."""
        return {
            'model': self.model_name,
            'device': self.device,
            'conf_threshold': self.conf,
            'iou_threshold': self.iou,
            'img_size': self.img_size,
            'classes': self.classes,
            'backend': 'ultralytics' if self.model else ('onnx' if self.onnx_session else 'none')
        }
