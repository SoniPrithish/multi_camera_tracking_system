"""
Tests for ReID module.
"""

import pytest
import numpy as np

from src.reid import build_reid, CosineReID, DeepReID


class TestCosineReID:
    """Tests for CosineReID (histogram-based)."""
    
    def test_init(self):
        reid = CosineReID()
        assert reid is not None
    
    def test_encode_returns_dict(self):
        reid = CosineReID()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tracks = [{'tid': 1, 'bbox': (100, 100, 50, 100)}]
        
        embs = reid.encode(frame, tracks)
        
        assert isinstance(embs, dict)
        assert 1 in embs
    
    def test_encode_returns_vectors(self):
        reid = CosineReID()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tracks = [{'tid': 1, 'bbox': (100, 100, 50, 100)}]
        
        embs = reid.encode(frame, tracks)
        
        assert isinstance(embs[1], np.ndarray)
        assert embs[1].ndim == 1
    
    def test_assign_global_ids(self):
        reid = CosineReID()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tracks = [{'tid': 1, 'bbox': (100, 100, 50, 100)}]
        
        embs = reid.encode(frame, tracks)
        gids = reid.assign_global_ids('cam1', tracks, embs)
        
        assert isinstance(gids, dict)
        assert 1 in gids
        assert isinstance(gids[1], int)
    
    def test_same_person_same_id(self):
        reid = CosineReID(sim_thresh=0.5)
        
        # Create a consistent-looking person crop
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        frame[100:200, 100:150] = [255, 0, 0]  # Red region
        
        tracks = [{'tid': 1, 'bbox': (100, 100, 50, 100)}]
        
        embs = reid.encode(frame, tracks)
        gids1 = reid.assign_global_ids('cam1', tracks, embs)
        
        # Same frame, different track ID - should match
        tracks2 = [{'tid': 2, 'bbox': (100, 100, 50, 100)}]
        embs2 = reid.encode(frame, tracks2)
        gids2 = reid.assign_global_ids('cam2', tracks2, embs2)
        
        # Should get same global ID (high similarity)
        assert gids1[1] == gids2[2]


class TestDeepReID:
    """Tests for DeepReID."""
    
    def test_init(self):
        reid = DeepReID()
        assert reid is not None
    
    def test_encode_returns_dict(self):
        reid = DeepReID()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tracks = [{'tid': 1, 'bbox': (100, 100, 50, 100)}]
        
        embs = reid.encode(frame, tracks)
        
        assert isinstance(embs, dict)
    
    def test_assign_global_ids(self):
        reid = DeepReID()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tracks = [{'tid': 1, 'bbox': (100, 100, 50, 100)}]
        
        embs = reid.encode(frame, tracks)
        gids = reid.assign_global_ids('cam1', tracks, embs)
        
        assert isinstance(gids, dict)
        assert 1 in gids
    
    def test_get_stats(self):
        reid = DeepReID()
        stats = reid.get_stats()
        
        assert 'total_identities' in stats
        assert 'model' in stats
    
    def test_reset(self):
        reid = DeepReID()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        tracks = [{'tid': 1, 'bbox': (100, 100, 50, 100)}]
        
        embs = reid.encode(frame, tracks)
        reid.assign_global_ids('cam1', tracks, embs)
        
        reid.reset()
        
        assert reid.next_gid == 1
        assert len(reid.gallery) == 0


class TestBuildReID:
    """Tests for build_reid factory."""
    
    def test_build_cosine(self):
        reid = build_reid('cosine')
        assert isinstance(reid, CosineReID)
    
    def test_build_deep(self):
        reid = build_reid('deep')
        assert isinstance(reid, DeepReID)
    
    def test_build_with_config(self):
        reid = build_reid({'name': 'cosine', 'sim_thresh': 0.8})
        assert isinstance(reid, CosineReID)
        assert reid.sim_thresh == 0.8
    
    def test_build_unknown_raises(self):
        with pytest.raises(ValueError):
            build_reid('unknown')

