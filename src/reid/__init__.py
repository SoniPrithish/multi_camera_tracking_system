from .cosine import CosineReID
from .deep import DeepReID


def build_reid(modules_cfg):
    """Build a ReID module from modules config dict or plain name string."""
    if isinstance(modules_cfg, str):
        name = modules_cfg
        modules_cfg = {}
    else:
        name = modules_cfg.get("reid", "cosine")

    name = (name or "cosine").lower()

    if name == "cosine":
        return CosineReID(
            sim_thresh=float(modules_cfg.get("reid_sim_thresh", 0.9)),
        )
    elif name == "deep":
        return DeepReID(
            model=modules_cfg.get("reid_model", "osnet_x0_25"),
            device=str(modules_cfg.get("reid_device", modules_cfg.get("detector_device", "0"))),
            sim_thresh=float(modules_cfg.get("reid_sim_thresh", 0.45)),
            ema_alpha=float(modules_cfg.get("reid_ema_alpha", 0.7)),
            gallery_ttl=float(modules_cfg.get("reid_gallery_ttl", 120)),
            max_gallery_size=int(modules_cfg.get("reid_max_gallery", 500)),
        )
    else:
        raise ValueError(f"Unknown reid: {name}")
