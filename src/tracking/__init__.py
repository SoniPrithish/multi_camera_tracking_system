
from .centroid import CentroidTracker
from .byte_tracker import ByteTracker

def build_tracker(name: str):
    name = (name or 'centroid').lower()
    if name == 'centroid':
        return CentroidTracker()
    elif name == 'byte':
        return ByteTracker()  # stub
    else:
        raise ValueError(f'Unknown tracker: {name}')
