from .cross_camera import CrossCameraAssociator


def build_associator(cfg: dict) -> CrossCameraAssociator:
    """Build a cross-camera association module from config."""
    return CrossCameraAssociator(cfg)
