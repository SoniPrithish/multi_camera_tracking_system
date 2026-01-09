"""
Data loaders for multi-camera tracking datasets.
Supports PETS, EPFL, WILDTRACK, and MOT formats.
"""

import os
import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterator, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FrameAnnotation:
    """Annotation for a single frame."""
    frame_id: int
    camera_id: str
    boxes: List[Tuple[float, float, float, float]]  # (x, y, w, h)
    track_ids: List[int]
    global_ids: Optional[List[int]] = None  # For multi-camera


@dataclass
class DatasetInfo:
    """Dataset metadata."""
    name: str
    num_cameras: int
    camera_ids: List[str]
    num_frames: int
    fps: float
    resolution: Tuple[int, int]  # (width, height)


class BaseDataLoader:
    """Base class for dataset loaders."""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        if not self.root_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {root_path}")
    
    def get_info(self) -> DatasetInfo:
        """Get dataset metadata."""
        raise NotImplementedError
    
    def iter_frames(self, camera_id: Optional[str] = None) -> Iterator[FrameAnnotation]:
        """Iterate over frame annotations."""
        raise NotImplementedError
    
    def get_frame(self, frame_id: int, camera_id: str) -> Optional[FrameAnnotation]:
        """Get annotation for a specific frame."""
        raise NotImplementedError
    
    def get_image_path(self, frame_id: int, camera_id: str) -> Optional[Path]:
        """Get path to frame image."""
        raise NotImplementedError


class MOTLoader(BaseDataLoader):
    """
    Loader for MOT Challenge format.
    
    Expected structure:
    root/
      gt/gt.txt
      det/det.txt
      img1/
        000001.jpg
        ...
    """
    
    def __init__(self, root_path: str, camera_id: str = "cam1"):
        super().__init__(root_path)
        self.camera_id = camera_id
        self._load_annotations()
    
    def _load_annotations(self):
        """Load ground truth annotations."""
        gt_file = self.root_path / "gt" / "gt.txt"
        
        self.annotations: Dict[int, FrameAnnotation] = {}
        
        if not gt_file.exists():
            logger.warning(f"Ground truth file not found: {gt_file}")
            return
        
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 6:
                    continue
                
                frame_id = int(parts[0])
                track_id = int(parts[1])
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                
                # Filter by confidence if available
                if len(parts) >= 7:
                    conf = float(parts[6])
                    if conf == 0:  # Ignored region
                        continue
                
                if frame_id not in self.annotations:
                    self.annotations[frame_id] = FrameAnnotation(
                        frame_id=frame_id,
                        camera_id=self.camera_id,
                        boxes=[],
                        track_ids=[]
                    )
                
                self.annotations[frame_id].boxes.append((x, y, w, h))
                self.annotations[frame_id].track_ids.append(track_id)
    
    def get_info(self) -> DatasetInfo:
        img_dir = self.root_path / "img1"
        num_frames = len(list(img_dir.glob("*.jpg"))) if img_dir.exists() else 0
        
        return DatasetInfo(
            name=self.root_path.name,
            num_cameras=1,
            camera_ids=[self.camera_id],
            num_frames=num_frames,
            fps=30.0,
            resolution=(1920, 1080)
        )
    
    def iter_frames(self, camera_id: Optional[str] = None) -> Iterator[FrameAnnotation]:
        for frame_id in sorted(self.annotations.keys()):
            yield self.annotations[frame_id]
    
    def get_frame(self, frame_id: int, camera_id: str) -> Optional[FrameAnnotation]:
        return self.annotations.get(frame_id)
    
    def get_image_path(self, frame_id: int, camera_id: str) -> Optional[Path]:
        img_path = self.root_path / "img1" / f"{frame_id:06d}.jpg"
        return img_path if img_path.exists() else None


class WILDTRACKLoader(BaseDataLoader):
    """
    Loader for WILDTRACK dataset.
    
    Expected structure:
    root/
      annotations_positions/
        00000000.json
        ...
      Image_subsets/
        C1/
          00000000.png
          ...
        C2/
        ...
    """
    
    CAMERA_IDS = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
    
    def __init__(self, root_path: str):
        super().__init__(root_path)
        self._load_annotations()
    
    def _load_annotations(self):
        """Load all annotations."""
        ann_dir = self.root_path / "annotations_positions"
        
        self.annotations: Dict[int, Dict[str, FrameAnnotation]] = {}
        
        if not ann_dir.exists():
            logger.warning(f"Annotations directory not found: {ann_dir}")
            return
        
        for ann_file in sorted(ann_dir.glob("*.json")):
            frame_id = int(ann_file.stem)
            
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            self.annotations[frame_id] = {}
            
            for cam_id in self.CAMERA_IDS:
                self.annotations[frame_id][cam_id] = FrameAnnotation(
                    frame_id=frame_id,
                    camera_id=cam_id,
                    boxes=[],
                    track_ids=[],
                    global_ids=[]
                )
            
            for person in data:
                person_id = person.get('personID', 0)
                
                for view in person.get('views', []):
                    cam_idx = view.get('viewNum', 0)
                    if cam_idx < 0 or cam_idx >= len(self.CAMERA_IDS):
                        continue
                    
                    cam_id = self.CAMERA_IDS[cam_idx]
                    
                    x_min = view.get('xmin', 0)
                    y_min = view.get('ymin', 0)
                    x_max = view.get('xmax', 0)
                    y_max = view.get('ymax', 0)
                    
                    w = x_max - x_min
                    h = y_max - y_min
                    
                    if w > 0 and h > 0:
                        self.annotations[frame_id][cam_id].boxes.append((x_min, y_min, w, h))
                        self.annotations[frame_id][cam_id].track_ids.append(person_id)
                        self.annotations[frame_id][cam_id].global_ids.append(person_id)
    
    def get_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="WILDTRACK",
            num_cameras=7,
            camera_ids=self.CAMERA_IDS,
            num_frames=len(self.annotations),
            fps=2.0,
            resolution=(1920, 1080)
        )
    
    def iter_frames(self, camera_id: Optional[str] = None) -> Iterator[FrameAnnotation]:
        for frame_id in sorted(self.annotations.keys()):
            if camera_id:
                if camera_id in self.annotations[frame_id]:
                    yield self.annotations[frame_id][camera_id]
            else:
                for cam_id in self.CAMERA_IDS:
                    if cam_id in self.annotations[frame_id]:
                        yield self.annotations[frame_id][cam_id]
    
    def get_frame(self, frame_id: int, camera_id: str) -> Optional[FrameAnnotation]:
        if frame_id in self.annotations and camera_id in self.annotations[frame_id]:
            return self.annotations[frame_id][camera_id]
        return None
    
    def get_image_path(self, frame_id: int, camera_id: str) -> Optional[Path]:
        img_path = self.root_path / "Image_subsets" / camera_id / f"{frame_id:08d}.png"
        return img_path if img_path.exists() else None


class EPFLLoader(BaseDataLoader):
    """
    Loader for EPFL multi-camera dataset.
    
    Expected structure:
    root/
      calibration/
      dataset_parameters/
      groundtruth.txt
      images/
        terrace1/
          0001.jpeg
          ...
    """
    
    def __init__(self, root_path: str):
        super().__init__(root_path)
        self._detect_cameras()
        self._load_annotations()
    
    def _detect_cameras(self):
        """Detect available camera views."""
        images_dir = self.root_path / "images"
        self.camera_ids = []
        
        if images_dir.exists():
            for d in sorted(images_dir.iterdir()):
                if d.is_dir():
                    self.camera_ids.append(d.name)
        
        if not self.camera_ids:
            # Fallback
            self.camera_ids = ['terrace1', 'terrace2', 'terrace3', 'terrace4']
    
    def _load_annotations(self):
        """Load ground truth annotations."""
        gt_file = self.root_path / "groundtruth.txt"
        
        self.annotations: Dict[int, Dict[str, FrameAnnotation]] = {}
        
        if not gt_file.exists():
            logger.warning(f"Ground truth not found: {gt_file}")
            return
        
        # Parse EPFL format (varies by sequence)
        # This is a simplified parser
        with open(gt_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                
                try:
                    frame_id = int(parts[0])
                    person_id = int(parts[1])
                    x = float(parts[2])
                    y = float(parts[3])
                    
                    # Approximate bounding box
                    w, h = 50, 150
                    
                    if frame_id not in self.annotations:
                        self.annotations[frame_id] = {}
                        for cam_id in self.camera_ids:
                            self.annotations[frame_id][cam_id] = FrameAnnotation(
                                frame_id=frame_id,
                                camera_id=cam_id,
                                boxes=[],
                                track_ids=[],
                                global_ids=[]
                            )
                    
                    # Add to first camera (simplified)
                    cam_id = self.camera_ids[0]
                    self.annotations[frame_id][cam_id].boxes.append((x, y, w, h))
                    self.annotations[frame_id][cam_id].track_ids.append(person_id)
                    self.annotations[frame_id][cam_id].global_ids.append(person_id)
                    
                except (ValueError, IndexError):
                    continue
    
    def get_info(self) -> DatasetInfo:
        return DatasetInfo(
            name="EPFL",
            num_cameras=len(self.camera_ids),
            camera_ids=self.camera_ids,
            num_frames=len(self.annotations),
            fps=25.0,
            resolution=(360, 288)
        )
    
    def iter_frames(self, camera_id: Optional[str] = None) -> Iterator[FrameAnnotation]:
        for frame_id in sorted(self.annotations.keys()):
            if camera_id:
                if camera_id in self.annotations[frame_id]:
                    yield self.annotations[frame_id][camera_id]
            else:
                for cam_id in self.camera_ids:
                    if cam_id in self.annotations[frame_id]:
                        yield self.annotations[frame_id][cam_id]
    
    def get_frame(self, frame_id: int, camera_id: str) -> Optional[FrameAnnotation]:
        if frame_id in self.annotations and camera_id in self.annotations[frame_id]:
            return self.annotations[frame_id][camera_id]
        return None
    
    def get_image_path(self, frame_id: int, camera_id: str) -> Optional[Path]:
        img_path = self.root_path / "images" / camera_id / f"{frame_id:04d}.jpeg"
        if not img_path.exists():
            img_path = img_path.with_suffix('.jpg')
        return img_path if img_path.exists() else None


def load_dataset(path: str, dataset_type: Optional[str] = None) -> BaseDataLoader:
    """
    Load a dataset based on path or type.
    
    Args:
        path: Path to dataset root
        dataset_type: Dataset type ('mot', 'wildtrack', 'epfl', or None for auto-detect)
        
    Returns:
        Appropriate DataLoader instance
    """
    path = Path(path)
    
    if dataset_type:
        dataset_type = dataset_type.lower()
    else:
        # Auto-detect based on structure
        if (path / "annotations_positions").exists():
            dataset_type = 'wildtrack'
        elif (path / "gt" / "gt.txt").exists():
            dataset_type = 'mot'
        elif (path / "groundtruth.txt").exists():
            dataset_type = 'epfl'
        else:
            raise ValueError(f"Could not auto-detect dataset type for: {path}")
    
    if dataset_type == 'mot':
        return MOTLoader(str(path))
    elif dataset_type == 'wildtrack':
        return WILDTRACKLoader(str(path))
    elif dataset_type == 'epfl':
        return EPFLLoader(str(path))
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

