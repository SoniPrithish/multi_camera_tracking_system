from .event_detector import EventDetector


def build_event_detector(cfg: dict) -> EventDetector:
    """Build an event detector from the 'events' config block."""
    return EventDetector(cfg)
