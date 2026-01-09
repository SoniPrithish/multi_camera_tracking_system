"""
Pytest configuration and fixtures.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def sample_frame():
    """Generate a sample video frame."""
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections():
    """Generate sample detections."""
    return [
        {'bbox': (100, 100, 50, 100), 'score': 0.9, 'cls': 0},
        {'bbox': (300, 200, 60, 120), 'score': 0.85, 'cls': 0},
    ]


@pytest.fixture
def sample_tracks():
    """Generate sample tracks."""
    return [
        {'tid': 1, 'bbox': (100, 100, 50, 100), 'score': 0.9},
        {'tid': 2, 'bbox': (300, 200, 60, 120), 'score': 0.85},
    ]


@pytest.fixture
def sample_config():
    """Generate sample pipeline configuration."""
    return {
        'inputs': [
            {'path': 'test.mp4', 'camera_id': 'cam1'}
        ],
        'output': {
            'video_dir': 'outputs/test',
            'tracks_path': 'outputs/test/tracks.jsonl'
        },
        'modules': {
            'detector': 'dummy',
            'tracker': 'centroid',
            'reid': 'cosine'
        },
        'runtime': {
            'max_frames': 10
        }
    }


@pytest.fixture
def zone_config():
    """Generate sample zone configuration."""
    return {
        'zones': [
            {
                'id': 'z1',
                'camera_id': 'cam1',
                'polygon': [[0, 0], [100, 0], [100, 100], [0, 100]],
                'type': 'entry',
                'name': 'Entrance'
            },
            {
                'id': 'z2',
                'camera_id': 'cam1',
                'polygon': [[200, 0], [300, 0], [300, 100], [200, 100]],
                'type': 'exit',
                'name': 'Exit'
            }
        ],
        'lines': [
            {
                'id': 'l1',
                'camera_id': 'cam1',
                'name': 'Counter',
                'start': [0, 50],
                'end': [100, 50]
            }
        ]
    }


@pytest.fixture
def camera_graph_config():
    """Generate sample camera graph configuration."""
    return {
        'cameras': [
            {'id': 'cam1', 'name': 'Entrance'},
            {'id': 'cam2', 'name': 'Hallway'},
            {'id': 'cam3', 'name': 'Exit'}
        ],
        'transitions': [
            {'from': 'cam1', 'to': 'cam2', 'min_time': 5, 'max_time': 30},
            {'from': 'cam2', 'to': 'cam3', 'min_time': 10, 'max_time': 60}
        ]
    }

