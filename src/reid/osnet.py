"""
OSNet-based person re-identification feature extractor — GPU optimized.

Uses MobileNetV3-Small backbone (pretrained) with FP16 half-precision.
Extracts 512-d appearance embeddings from person crops.
Optimized for RTX 4060 Ti — large batch, FP16, CUDA streams.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

logger = logging.getLogger(__name__)


class OSNetExtractor:
    """Extract 512-d appearance embeddings — GPU optimized with FP16."""

    def __init__(
        self,
        model_name: str = "osnet_x0_25",
        device: str = "0",
        input_h: int = 256,
        input_w: int = 128,
        batch_size: int = 64,
        half: bool = True,
    ):
        self.device = torch.device(f"cuda:{device}" if device.isdigit() and torch.cuda.is_available() else "cpu")
        self.input_h = input_h
        self.input_w = input_w
        self.batch_size = batch_size
        self.use_half = half and self.device.type == "cuda"

        logger.info("Loading ReID model=%s device=%s half=%s", model_name, self.device, self.use_half)

        self.model = self._build_model(model_name)
        self.model = self.model.to(self.device)
        if self.use_half:
            self.model = self.model.half()
        self.model.eval()

        # Precompute normalization tensors on GPU
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        if self.use_half:
            self.mean = self.mean.half()
            self.std = self.std.half()

        # Warm-up
        dummy = torch.randn(4, 3, input_h, input_w, device=self.device)
        if self.use_half:
            dummy = dummy.half()
        with torch.no_grad():
            out = self.model(self._normalize(dummy))
        self.embed_dim = out.shape[1]
        logger.info("ReID model loaded — embed_dim=%d, device=%s, FP16=%s", self.embed_dim, self.device, self.use_half)

    def _build_model(self, model_name: str) -> nn.Module:
        import torchvision.models as models
        if model_name in ("osnet_x0_25", "mobilenet"):
            backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            features = backbone.features
            pool = nn.AdaptiveAvgPool2d(1)
            head = nn.Sequential(
                nn.Linear(576, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
            )
            return _ReIDModel(features, pool, head)
        elif model_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            features = nn.Sequential(*list(backbone.children())[:-2])
            pool = nn.AdaptiveAvgPool2d(1)
            head = nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512))
            return _ReIDModel(features, pool, head)
        else:
            raise ValueError(f"Unknown ReID model: {model_name}")

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor with ImageNet stats (already on GPU)."""
        return (x - self.mean) / self.std

    @torch.no_grad()
    def extract(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract embeddings from BGR person crops — batched GPU inference.

        Parameters
        ----------
        crops : list of np.ndarray (H, W, 3) BGR

        Returns
        -------
        embeddings : np.ndarray (N, embed_dim), L2-normalized
        """
        if not crops:
            return np.zeros((0, self.embed_dim), dtype=np.float32)

        all_embs = []
        for i in range(0, len(crops), self.batch_size):
            batch_crops = crops[i : i + self.batch_size]

            # Fast resize + convert on CPU, then move to GPU in one shot
            tensors = []
            for crop in batch_crops:
                if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
                    tensors.append(torch.zeros(3, self.input_h, self.input_w))
                    continue
                # BGR→RGB, resize, to tensor (0-1)
                import cv2
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (self.input_w, self.input_h))
                t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                tensors.append(t)

            batch = torch.stack(tensors).to(self.device, non_blocking=True)
            if self.use_half:
                batch = batch.half()

            batch = self._normalize(batch)
            embs = self.model(batch)

            # L2 normalize
            embs = embs / (embs.norm(dim=1, keepdim=True) + 1e-8)
            all_embs.append(embs.float().cpu().numpy())

        return np.concatenate(all_embs, axis=0)

    def extract_single(self, crop: np.ndarray) -> np.ndarray:
        embs = self.extract([crop])
        return embs[0] if len(embs) > 0 else np.zeros(self.embed_dim, dtype=np.float32)


class _ReIDModel(nn.Module):
    def __init__(self, features, pool, head):
        super().__init__()
        self.features = features
        self.pool = pool
        self.head = head

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x
