"""
Benchmark runner for multi-camera tracking evaluation.
Runs tracking pipeline on datasets and computes metrics.
"""

import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np

from .data_loaders import load_dataset, BaseDataLoader, FrameAnnotation
from .metrics import MOTACalculator, IDF1Calculator, HOTACalculator, MultiCameraMetrics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    dataset_path: str
    dataset_type: Optional[str] = None
    output_dir: str = "outputs/benchmark"
    
    # Detection settings
    detector_name: str = "yolov8"
    detector_conf: float = 0.25
    
    # Tracker settings
    tracker_name: str = "bytetrack"
    
    # ReID settings
    reid_name: str = "deep"
    
    # Evaluation settings
    iou_threshold: float = 0.5
    max_frames: Optional[int] = None
    
    # Profiling
    profile_latency: bool = True


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    dataset_name: str
    num_frames: int
    num_cameras: int
    
    # Metrics
    mota: float
    idf1: float
    hota: float
    precision: float
    recall: float
    
    # Per-camera metrics
    per_camera: Dict[str, Dict[str, float]]
    
    # Cross-camera metrics
    handoff_precision: float = 0.0
    handoff_recall: float = 0.0
    
    # Profiling
    avg_fps: float = 0.0
    avg_latency_ms: float = 0.0
    latency_breakdown: Dict[str, float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class LatencyProfiler:
    """Profile latency of pipeline stages."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self._start_times: Dict[str, float] = {}
    
    def start(self, stage: str):
        self._start_times[stage] = time.perf_counter()
    
    def stop(self, stage: str):
        if stage in self._start_times:
            elapsed = (time.perf_counter() - self._start_times[stage]) * 1000  # ms
            if stage not in self.timings:
                self.timings[stage] = []
            self.timings[stage].append(elapsed)
    
    def get_summary(self) -> Dict[str, float]:
        return {
            stage: np.mean(times) for stage, times in self.timings.items()
        }
    
    def get_total_avg(self) -> float:
        if 'total' in self.timings:
            return np.mean(self.timings['total'])
        return sum(np.mean(t) for t in self.timings.values())


class BenchmarkRunner:
    """
    Runs tracking pipeline on evaluation datasets.
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.profiler = LatencyProfiler() if config.profile_latency else None
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._init_pipeline()
    
    def _init_pipeline(self):
        """Initialize tracking pipeline components."""
        # Import here to avoid circular imports
        from ..detection import build_detector
        from ..tracking import build_tracker
        from ..reid import build_reid
        
        self.detector = build_detector({
            'name': self.config.detector_name,
            'conf': self.config.detector_conf
        })
        
        self.tracker = build_tracker({
            'name': self.config.tracker_name
        })
        
        self.reid = build_reid({
            'name': self.config.reid_name
        })
        
        logger.info(f"Pipeline initialized: detector={self.config.detector_name}, "
                   f"tracker={self.config.tracker_name}, reid={self.config.reid_name}")
    
    def run(self) -> BenchmarkResult:
        """
        Run benchmark evaluation.
        
        Returns:
            BenchmarkResult with all metrics
        """
        logger.info(f"Loading dataset: {self.config.dataset_path}")
        
        # Load dataset
        loader = load_dataset(self.config.dataset_path, self.config.dataset_type)
        info = loader.get_info()
        
        logger.info(f"Dataset: {info.name}, cameras={info.num_cameras}, frames={info.num_frames}")
        
        # Initialize metrics
        metrics = MultiCameraMetrics(self.config.iou_threshold)
        hota_calc = HOTACalculator([0.5])
        
        # Process frames
        frame_count = 0
        start_time = time.time()
        
        import cv2
        
        for camera_id in info.camera_ids:
            logger.info(f"Processing camera: {camera_id}")
            
            for annotation in loader.iter_frames(camera_id):
                if self.config.max_frames and frame_count >= self.config.max_frames:
                    break
                
                # Load image
                img_path = loader.get_image_path(annotation.frame_id, camera_id)
                if img_path is None or not img_path.exists():
                    continue
                
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue
                
                # Profile total time
                if self.profiler:
                    self.profiler.start('total')
                
                # Detection
                if self.profiler:
                    self.profiler.start('detection')
                detections = self.detector.detect(frame)
                if self.profiler:
                    self.profiler.stop('detection')
                
                # Tracking
                if self.profiler:
                    self.profiler.start('tracking')
                tracks = self.tracker.update(camera_id, detections)
                if self.profiler:
                    self.profiler.stop('tracking')
                
                # ReID
                if self.profiler:
                    self.profiler.start('reid')
                embeddings = self.reid.encode(frame, tracks)
                global_ids = self.reid.assign_global_ids(camera_id, tracks, embeddings)
                if self.profiler:
                    self.profiler.stop('reid')
                
                if self.profiler:
                    self.profiler.stop('total')
                
                # Extract predictions
                pred_boxes = [t['bbox'] for t in tracks]
                pred_ids = [t['tid'] for t in tracks]
                pred_global_ids = [global_ids.get(t['tid'], t['tid']) for t in tracks]
                
                # Update metrics
                metrics.update_camera(
                    camera_id,
                    annotation.boxes,
                    annotation.track_ids,
                    pred_boxes,
                    pred_ids
                )
                
                hota_calc.update(
                    annotation.boxes,
                    annotation.track_ids,
                    pred_boxes,
                    pred_ids
                )
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames...")
        
        # Compute final metrics
        elapsed = time.time() - start_time
        
        metrics_result = metrics.compute()
        hota_result = hota_calc.compute()
        
        # Aggregate per-camera metrics
        per_camera = metrics_result.get('per_camera', {})
        
        avg_mota = np.mean([m.get('mota', 0) for m in per_camera.values()]) if per_camera else 0
        avg_idf1 = np.mean([m.get('idf1', 0) for m in per_camera.values()]) if per_camera else 0
        avg_precision = np.mean([m.get('precision', 0) for m in per_camera.values()]) if per_camera else 0
        avg_recall = np.mean([m.get('recall', 0) for m in per_camera.values()]) if per_camera else 0
        
        # Cross-camera metrics
        cross_cam = metrics_result.get('cross_camera', {})
        
        # Latency
        latency_breakdown = self.profiler.get_summary() if self.profiler else {}
        avg_latency = self.profiler.get_total_avg() if self.profiler else 0
        avg_fps = frame_count / elapsed if elapsed > 0 else 0
        
        result = BenchmarkResult(
            dataset_name=info.name,
            num_frames=frame_count,
            num_cameras=info.num_cameras,
            mota=avg_mota,
            idf1=avg_idf1,
            hota=hota_result.get('hota', 0),
            precision=avg_precision,
            recall=avg_recall,
            per_camera=per_camera,
            handoff_precision=cross_cam.get('handoff_precision', 0),
            handoff_recall=cross_cam.get('handoff_recall', 0),
            avg_fps=avg_fps,
            avg_latency_ms=avg_latency,
            latency_breakdown=latency_breakdown
        )
        
        # Save results
        result_path = self.output_dir / f"benchmark_{info.name}_{int(time.time())}.json"
        result.to_json(str(result_path))
        logger.info(f"Results saved to: {result_path}")
        
        # Print summary
        self._print_summary(result)
        
        return result
    
    def _print_summary(self, result: BenchmarkResult):
        """Print benchmark summary."""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Dataset: {result.dataset_name}")
        print(f"Frames: {result.num_frames}, Cameras: {result.num_cameras}")
        print("-" * 60)
        print(f"MOTA:      {result.mota:.4f}")
        print(f"IDF1:      {result.idf1:.4f}")
        print(f"HOTA:      {result.hota:.4f}")
        print(f"Precision: {result.precision:.4f}")
        print(f"Recall:    {result.recall:.4f}")
        print("-" * 60)
        print(f"Avg FPS:     {result.avg_fps:.2f}")
        print(f"Avg Latency: {result.avg_latency_ms:.2f} ms")
        
        if result.latency_breakdown:
            print("\nLatency Breakdown:")
            for stage, latency in result.latency_breakdown.items():
                print(f"  {stage}: {latency:.2f} ms")
        
        print("=" * 60 + "\n")


def run_benchmark(
    dataset_path: str,
    dataset_type: Optional[str] = None,
    output_dir: str = "outputs/benchmark",
    max_frames: Optional[int] = None
) -> BenchmarkResult:
    """
    Convenience function to run benchmark.
    
    Args:
        dataset_path: Path to dataset
        dataset_type: Dataset type (auto-detected if None)
        output_dir: Output directory for results
        max_frames: Maximum frames to process
        
    Returns:
        BenchmarkResult
    """
    config = BenchmarkConfig(
        dataset_path=dataset_path,
        dataset_type=dataset_type,
        output_dir=output_dir,
        max_frames=max_frames
    )
    
    runner = BenchmarkRunner(config)
    return runner.run()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run tracking benchmark')
    parser.add_argument('--dataset', '-d', required=True, help='Path to dataset')
    parser.add_argument('--type', '-t', help='Dataset type')
    parser.add_argument('--output', '-o', default='outputs/benchmark', help='Output directory')
    parser.add_argument('--max-frames', '-m', type=int, help='Max frames to process')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    run_benchmark(
        dataset_path=args.dataset,
        dataset_type=args.type,
        output_dir=args.output,
        max_frames=args.max_frames
    )

