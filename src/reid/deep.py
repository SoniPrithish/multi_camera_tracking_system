"""
Deep ReID module using OSNet for person re-identification.
Supports PyTorch, TorchScript, and ONNX backends.
"""

import os
import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class DeepReID:
    """
    Deep ReID using OSNet or similar models.
    Extracts 512-dim embeddings for cross-camera matching.
    """
    
    # Standard ReID input size
    INPUT_SIZE = (256, 128)  # (height, width)
    EMBEDDING_DIM = 512
    
    # ImageNet normalization
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __init__(
        self,
        model: str = 'osnet_x0_25',
        device: str = 'auto',
        ema_alpha: float = 0.9,
        gallery_size: int = 100,
        sim_thresh: float = 0.6,
        model_path: Optional[str] = None,
    ):
        """
        Initialize DeepReID.
        
        Args:
            model: Model name (osnet_x0_25, osnet_x1_0, osnet_ain_x1_0)
            device: Device ('auto', 'cpu', 'cuda')
            ema_alpha: EMA smoothing factor for embeddings
            gallery_size: Maximum gallery size per global ID
            sim_thresh: Similarity threshold for matching
            model_path: Optional path to pre-trained weights
        """
        self.model_name = model
        self.ema_alpha = ema_alpha
        self.gallery_size = gallery_size
        self.sim_thresh = sim_thresh
        self.model_path = model_path
        
        # Resolve device
        self.device = self._resolve_device(device)
        
        # Model backends
        self.torch_model = None
        self.onnx_session = None
        
        # Global ID management
        self.next_gid = 1
        self.gallery: Dict[int, np.ndarray] = {}  # gid -> EMA embedding
        self.gallery_history: Dict[int, List[np.ndarray]] = defaultdict(list)
        
        # Track-to-global ID mapping per camera
        self.track_to_global: Dict[str, Dict[int, int]] = defaultdict(dict)
        
        # Load model
        self._load_model()
        
        logger.info(f"DeepReID initialized: model={model}, device={self.device}")
    
    def _resolve_device(self, device: str) -> str:
        """Resolve device string."""
        if device == 'auto':
            try:
                import torch
                return 'cuda' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                return 'cpu'
        return device
    
    def _load_model(self):
        """Load the ReID model."""
        # Try PyTorch first
        if self._load_torch_model():
            return
        
        # Try ONNX fallback
        if self._load_onnx_model():
            return
        
        logger.warning("No ReID model loaded. Using fallback color histogram features.")
    
    def _load_torch_model(self) -> bool:
        """Load model using torchreid."""
        try:
            import torch
            
            # Try torchreid
            try:
                from torchreid.utils import FeatureExtractor
                
                self.torch_model = FeatureExtractor(
                    model_name=self.model_name,
                    model_path=self.model_path,
                    device=self.device
                )
                logger.info(f"Loaded torchreid model: {self.model_name}")
                return True
            except ImportError:
                pass
            
            # Try loading from torchvision or custom
            if self.model_path and os.path.exists(self.model_path):
                # Load TorchScript model
                if self.model_path.endswith('.pt') or self.model_path.endswith('.pth'):
                    self.torch_model = torch.jit.load(self.model_path)
                    self.torch_model.to(self.device)
                    self.torch_model.eval()
                    logger.info(f"Loaded TorchScript model: {self.model_path}")
                    return True
            
            return False
            
        except ImportError:
            logger.warning("PyTorch not installed")
            return False
        except Exception as e:
            logger.warning(f"Failed to load torch model: {e}")
            return False
    
    def _load_onnx_model(self) -> bool:
        """Load ONNX model."""
        try:
            import onnxruntime as ort
            
            onnx_path = self.model_path
            if onnx_path is None:
                # Look for default ONNX model
                onnx_path = f"models/{self.model_name}.onnx"
            
            if not os.path.exists(onnx_path):
                return False
            
            providers = ['CPUExecutionProvider']
            if self.device != 'cpu':
                providers = ['CUDAExecutionProvider'] + providers
            
            self.onnx_session = ort.InferenceSession(onnx_path, providers=providers)
            self.onnx_input_name = self.onnx_session.get_inputs()[0].name
            
            logger.info(f"Loaded ONNX model: {onnx_path}")
            return True
            
        except ImportError:
            return False
        except Exception as e:
            logger.warning(f"Failed to load ONNX model: {e}")
            return False
    
    def encode(self, frame: np.ndarray, tracks: List[Dict]) -> Dict[int, np.ndarray]:
        """
        Extract embeddings for tracked persons.
        
        Args:
            frame: BGR image
            tracks: List of tracks with 'tid' and 'bbox' keys
            
        Returns:
            Dict mapping track_id to embedding vector
        """
        if len(tracks) == 0:
            return {}
        
        # Crop and preprocess
        crops = []
        valid_tids = []
        
        for t in tracks:
            x, y, w, h = map(int, t['bbox'])
            
            # Ensure valid crop
            x = max(0, x)
            y = max(0, y)
            x2 = min(frame.shape[1], x + w)
            y2 = min(frame.shape[0], y + h)
            
            if x2 <= x or y2 <= y:
                continue
            
            crop = frame[y:y2, x:x2]
            if crop.size == 0:
                continue
            
            # Preprocess
            crop = self._preprocess(crop)
            crops.append(crop)
            valid_tids.append(t['tid'])
        
        if len(crops) == 0:
            return {}
        
        # Batch inference
        batch = np.stack(crops, axis=0)
        embeddings = self._extract_features(batch)
        
        # Map to track IDs
        result = {}
        for tid, emb in zip(valid_tids, embeddings):
            result[tid] = emb
        
        return result
    
    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Preprocess crop for ReID model."""
        # Resize to standard size
        resized = cv2.resize(crop, (self.INPUT_SIZE[1], self.INPUT_SIZE[0]))
        
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - self.MEAN) / self.STD
        
        # HWC to CHW
        transposed = normalized.transpose(2, 0, 1)
        
        return transposed
    
    def _extract_features(self, batch: np.ndarray) -> np.ndarray:
        """Extract features from preprocessed batch."""
        if self.torch_model is not None:
            return self._extract_torch(batch)
        elif self.onnx_session is not None:
            return self._extract_onnx(batch)
        else:
            return self._extract_fallback(batch)
    
    def _extract_torch(self, batch: np.ndarray) -> np.ndarray:
        """Extract features using PyTorch model."""
        import torch
        
        with torch.no_grad():
            tensor = torch.from_numpy(batch).to(self.device)
            
            if hasattr(self.torch_model, '__call__'):
                # torchreid FeatureExtractor expects image paths or numpy arrays
                # We need to handle this differently
                features = self.torch_model(tensor)
            else:
                features = self.torch_model(tensor)
            
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()
        
        # Normalize
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-6
        return features / norms
    
    def _extract_onnx(self, batch: np.ndarray) -> np.ndarray:
        """Extract features using ONNX Runtime."""
        outputs = self.onnx_session.run(None, {self.onnx_input_name: batch.astype(np.float32)})
        features = outputs[0]
        
        # Normalize
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-6
        return features / norms
    
    def _extract_fallback(self, batch: np.ndarray) -> np.ndarray:
        """Fallback feature extraction using color histograms."""
        features = []
        
        for img in batch:
            # CHW to HWC
            img = img.transpose(1, 2, 0)
            
            # Denormalize
            img = img * self.STD + self.MEAN
            img = (img * 255).clip(0, 255).astype(np.uint8)
            
            # Color histogram
            hist = []
            for ch in range(3):
                h = cv2.calcHist([img], [ch], None, [32], [0, 256]).flatten()
                hist.append(h)
            
            # Spatial pyramid
            h, w = img.shape[:2]
            for r in range(2):
                for c in range(2):
                    region = img[r*h//2:(r+1)*h//2, c*w//2:(c+1)*w//2]
                    for ch in range(3):
                        h_r = cv2.calcHist([region], [ch], None, [16], [0, 256]).flatten()
                        hist.append(h_r)
            
            feat = np.concatenate(hist)
            features.append(feat)
        
        features = np.array(features, dtype=np.float32)
        
        # Normalize
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-6
        return features / norms
    
    def assign_global_ids(
        self,
        cam_id: str,
        tracks: List[Dict],
        embeddings: Dict[int, np.ndarray]
    ) -> Dict[int, int]:
        """
        Assign global IDs to tracks using appearance matching.
        
        Args:
            cam_id: Camera identifier
            tracks: List of tracks
            embeddings: Embeddings for each track
            
        Returns:
            Dict mapping track_id to global_id
        """
        result = {}
        cam_mapping = self.track_to_global[cam_id]
        
        for t in tracks:
            tid = t['tid']
            emb = embeddings.get(tid)
            
            # Check if track already has global ID
            if tid in cam_mapping:
                gid = cam_mapping[tid]
                
                # Update gallery with EMA
                if emb is not None and gid in self.gallery:
                    self._update_gallery(gid, emb)
                
                result[tid] = gid
                continue
            
            # New track - try to match with gallery
            if emb is not None:
                gid, similarity = self._match_gallery(emb)
                
                if gid is not None and similarity >= self.sim_thresh:
                    # Matched existing identity
                    self._update_gallery(gid, emb)
                else:
                    # New identity
                    gid = self._create_new_identity(emb)
            else:
                # No embedding - create new identity
                gid = self._create_new_identity(None)
            
            cam_mapping[tid] = gid
            result[tid] = gid
        
        return result
    
    def _match_gallery(self, embedding: np.ndarray) -> Tuple[Optional[int], float]:
        """Find best match in gallery."""
        if len(self.gallery) == 0:
            return None, 0.0
        
        best_gid = None
        best_sim = -1.0
        
        for gid, gallery_emb in self.gallery.items():
            if gallery_emb is None:
                continue
            
            # Cosine similarity
            sim = float(np.dot(embedding, gallery_emb))
            
            if sim > best_sim:
                best_sim = sim
                best_gid = gid
        
        return best_gid, best_sim
    
    def _update_gallery(self, gid: int, embedding: np.ndarray):
        """Update gallery with EMA."""
        if gid not in self.gallery or self.gallery[gid] is None:
            self.gallery[gid] = embedding
        else:
            # EMA update
            self.gallery[gid] = (
                self.ema_alpha * self.gallery[gid] + 
                (1 - self.ema_alpha) * embedding
            )
            # Re-normalize
            norm = np.linalg.norm(self.gallery[gid]) + 1e-6
            self.gallery[gid] = self.gallery[gid] / norm
        
        # Store in history
        self.gallery_history[gid].append(embedding)
        if len(self.gallery_history[gid]) > self.gallery_size:
            self.gallery_history[gid].pop(0)
    
    def _create_new_identity(self, embedding: Optional[np.ndarray]) -> int:
        """Create new global identity."""
        gid = self.next_gid
        self.next_gid += 1
        
        if embedding is not None:
            self.gallery[gid] = embedding.copy()
            self.gallery_history[gid] = [embedding]
        else:
            self.gallery[gid] = None
        
        return gid
    
    def get_gallery_embedding(self, gid: int) -> Optional[np.ndarray]:
        """Get gallery embedding for a global ID."""
        return self.gallery.get(gid)
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2))
    
    def reset(self):
        """Reset all state."""
        self.next_gid = 1
        self.gallery.clear()
        self.gallery_history.clear()
        self.track_to_global.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ReID statistics."""
        return {
            'total_identities': len(self.gallery),
            'gallery_sizes': {gid: len(hist) for gid, hist in self.gallery_history.items()},
            'model': self.model_name,
            'device': self.device,
            'backend': 'torch' if self.torch_model else ('onnx' if self.onnx_session else 'fallback')
        }
