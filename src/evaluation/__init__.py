"""
Evaluation module for multi-camera tracking.
Provides dataset loaders, metrics, and benchmark runner.
"""

from .data_loaders import (
    load_dataset,
    BaseDataLoader,
    MOTLoader,
    WILDTRACKLoader,
    EPFLLoader,
    FrameAnnotation,
    DatasetInfo
)
from .metrics import (
    MOTACalculator,
    IDF1Calculator,
    HOTACalculator,
    MultiCameraMetrics,
    compute_iou,
    match_detections
)
from .benchmark import (
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkResult,
    LatencyProfiler,
    run_benchmark
)


__all__ = [
    # Data loaders
    'load_dataset',
    'BaseDataLoader',
    'MOTLoader',
    'WILDTRACKLoader',
    'EPFLLoader',
    'FrameAnnotation',
    'DatasetInfo',
    # Metrics
    'MOTACalculator',
    'IDF1Calculator',
    'HOTACalculator',
    'MultiCameraMetrics',
    'compute_iou',
    'match_detections',
    # Benchmark
    'BenchmarkRunner',
    'BenchmarkConfig',
    'BenchmarkResult',
    'LatencyProfiler',
    'run_benchmark',
]

