
from .cosine import CosineReID
from .deep import DeepReID

def build_reid(name: str):
    name = (name or 'cosine').lower()
    if name == 'cosine':
        return CosineReID()
    elif name == 'deep':
        return DeepReID()  # stub
    else:
        raise ValueError(f'Unknown reid: {name}')
