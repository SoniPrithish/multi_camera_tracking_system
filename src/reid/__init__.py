"""
Re-Identification module for cross-camera person matching.
Provides appearance-based feature extraction and matching.
"""

from .cosine import CosineReID
from .deep import DeepReID


def build_reid(config):
    """
    Build a ReID module based on configuration.
    
    Args:
        config: Either a string (reid name) or dict with 'name' and params
        
    Returns:
        ReID instance
    """
    if isinstance(config, str):
        name = config.lower()
        params = {}
    else:
        name = config.get('name', 'cosine').lower()
        params = {k: v for k, v in config.items() if k != 'name'}
    
    if name == 'cosine':
        return CosineReID(**params)
    elif name in ('deep', 'osnet'):
        return DeepReID(**params)
    else:
        raise ValueError(f'Unknown reid: {name}')


__all__ = ['CosineReID', 'DeepReID', 'build_reid']
