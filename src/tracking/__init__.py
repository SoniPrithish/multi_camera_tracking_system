from .centroid import CentroidTracker
from .byte_tracker import ByteTracker


def build_tracker(modules_cfg):
    """Build a tracker from modules config dict or plain name string."""
    if isinstance(modules_cfg, str):
        name = modules_cfg
        modules_cfg = {}
    else:
        name = modules_cfg.get("tracker", "centroid")

    name = (name or "centroid").lower()

    if name == "centroid":
        return CentroidTracker()
    elif name == "byte":
        return ByteTracker(
            max_age=int(modules_cfg.get("tracker_max_age", 30)),
            min_hits=int(modules_cfg.get("tracker_min_hits", 3)),
            iou_thresh=float(modules_cfg.get("tracker_iou_thresh", 0.2)),
            high_thresh=float(modules_cfg.get("tracker_high_thresh", 0.5)),
            low_thresh=float(modules_cfg.get("tracker_low_thresh", 0.1)),
            match_thresh=float(modules_cfg.get("tracker_match_thresh", 0.8)),
        )
    else:
        raise ValueError(f"Unknown tracker: {name}")
