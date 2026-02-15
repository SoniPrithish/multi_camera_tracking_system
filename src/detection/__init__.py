from .detector import DummyDetector
from .yolov8 import YOLOv8Detector


def build_detector(modules_cfg):
    """Build a detector from the full modules config dict (or a plain name string)."""
    if isinstance(modules_cfg, str):
        name = modules_cfg
        modules_cfg = {}
    else:
        name = modules_cfg.get("detector", "dummy")

    name = (name or "dummy").lower()

    if name == "dummy":
        return DummyDetector()
    elif name == "yolov8":
        return YOLOv8Detector(
            model=modules_cfg.get("detector_model", "yolov8n.pt"),
            device=str(modules_cfg.get("detector_device", "cpu")),
            conf=float(modules_cfg.get("detector_conf", 0.35)),
            iou=float(modules_cfg.get("detector_iou", 0.45)),
            img_size=int(modules_cfg.get("detector_img_size", 640)),
        )
    else:
        raise ValueError(f"Unknown detector: {name}")
