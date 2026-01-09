"""
Multi-camera tracking pipeline.
Orchestrates detection, tracking, ReID, and analytics.
"""

import os
import time
import json
import logging
from typing import Dict, List, Optional, Any

import cv2
import numpy as np

from .io.streams import MultiVideoReader
from .io.sink import VideoWriterMux, TracksWriter
from .io.health import HealthMonitor, HealthThresholds
from .utils.viz import draw_detections
from .detection import build_detector
from .tracking import build_tracker
from .reid import build_reid
from .association import GlobalRegistry, CameraGraph, ZoneManager, ZoneGate, CrossCameraMatcher
from .analytics import AnalyticsZoneManager, TrackAnalytics, EventAggregator

logger = logging.getLogger(__name__)


class MultiCamPipeline:
    """
    Complete multi-camera tracking pipeline.
    
    Integrates:
    - Multi-stream video input with health monitoring
    - Person detection (YOLOv8/ONNX)
    - Single-camera tracking (ByteTrack)
    - Cross-camera ReID (OSNet)
    - Global identity registry
    - Zone-based analytics
    """
    
    def __init__(self, cfg: dict):
        """
        Initialize pipeline from configuration.
        
        Args:
            cfg: Configuration dictionary
        """
        self.cfg = cfg
        self.running = False
        
        # Video I/O
        self.reader = MultiVideoReader(cfg['inputs'])
        os.makedirs(cfg['output']['video_dir'], exist_ok=True)
        self.writer = VideoWriterMux(cfg['output']['video_dir'])
        self.tracks_writer = TracksWriter(cfg['output']['tracks_path'])
        
        # Health monitoring
        self.health_monitor = HealthMonitor(
            thresholds=HealthThresholds(
                min_fps=cfg.get('runtime', {}).get('min_fps', 10.0),
                max_latency_ms=cfg.get('runtime', {}).get('max_latency_ms', 200.0)
            )
        )
        
        # Detection
        detector_cfg = cfg.get('modules', {}).get('detector', 'dummy')
        self.detector = build_detector(detector_cfg)
        
        # Tracking
        tracker_cfg = cfg.get('modules', {}).get('tracker', 'centroid')
        self.tracker = build_tracker(tracker_cfg)
        
        # ReID
        reid_cfg = cfg.get('modules', {}).get('reid', 'cosine')
        self.reid = build_reid(reid_cfg)
        
        # Global registry
        self.registry = GlobalRegistry(
            ema_alpha=cfg.get('reid', {}).get('ema_alpha', 0.9),
            lost_timeout=cfg.get('reid', {}).get('lost_timeout', 300.0)
        )
        
        # Camera graph (if configured)
        self.camera_graph = None
        if 'camera_graph' in cfg:
            self.camera_graph = CameraGraph.from_config(cfg['camera_graph'])
        
        # Zone gate (if configured)
        self.zone_gate = None
        if 'zones' in cfg:
            zone_manager = ZoneManager.from_config(cfg['zones'])
            self.zone_gate = ZoneGate(zone_manager)
        
        # Cross-camera matcher
        self.matcher = CrossCameraMatcher(
            similarity_threshold=cfg.get('reid', {}).get('sim_thresh', 0.5)
        )
        
        # Analytics
        analytics_cfg = cfg.get('analytics', {})
        self.analytics_zone_manager = None
        self.track_analytics = None
        self.aggregator = None
        
        if 'zones' in analytics_cfg:
            self.analytics_zone_manager = AnalyticsZoneManager.from_config(analytics_cfg)
            self.track_analytics = TrackAnalytics(
                self.analytics_zone_manager,
                dwell_update_interval=analytics_cfg.get('dwell_update_interval', 10.0)
            )
            self.aggregator = EventAggregator()
            
            # Connect analytics to aggregator
            self.track_analytics.add_event_callback(self.aggregator.process_event)
        
        # Runtime settings
        self.runtime = cfg.get('runtime', {})
        self.draw_cfg = self.runtime.get('draw', {})
        
        # Statistics
        self.stats = {
            'frame_count': 0,
            'start_time': None,
            'fps': 0.0
        }
        
        logger.info("MultiCamPipeline initialized")
    
    def run(self):
        """Run the tracking pipeline."""
        self.running = True
        self.stats['start_time'] = time.time()
        
        max_frames = self.runtime.get('max_frames')
        skip_frames = int(self.runtime.get('skip_frames', 1))
        show = bool(self.cfg['output'].get('show', False))
        
        # Start health monitoring
        self.health_monitor.start(lambda: self.reader.get_health())
        
        frame_idx = 0
        fps_counter = 0
        fps_start = time.time()
        
        logger.info("Pipeline started")
        
        try:
            while self.running:
                batch = self.reader.read()
                if batch is None:
                    break
                
                frame_idx += 1
                
                if skip_frames > 1 and frame_idx % skip_frames != 0:
                    continue
                
                for cam_id, frame in batch.items():
                    self._process_frame(cam_id, frame, frame_idx)
                
                # FPS calculation
                fps_counter += 1
                if time.time() - fps_start >= 1.0:
                    self.stats['fps'] = fps_counter
                    fps_counter = 0
                    fps_start = time.time()
                
                self.stats['frame_count'] = frame_idx
                
                # Visualization
                if show:
                    tiled = self.writer.latest_tiled()
                    if tiled is not None:
                        cv2.imshow('MULTI_CAMERA_TRACKING', tiled)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break
                
                if max_frames is not None and frame_idx >= int(max_frames):
                    break
        
        except KeyboardInterrupt:
            logger.info("Pipeline interrupted")
        
        finally:
            self.stop()
    
    def _process_frame(self, cam_id: str, frame: np.ndarray, frame_idx: int):
        """Process a single frame from one camera."""
        now = time.time()
        
        # Detection
        dets = self.detector.detect(frame)
        
        # Per-camera tracking
        cam_tracks = self.tracker.update(cam_id, dets)
        
        # ReID embeddings
        embs = self.reid.encode(frame, cam_tracks)
        
        # Global ID assignment
        global_ids = self.reid.assign_global_ids(cam_id, cam_tracks, embs)
        
        # Update registry
        for track in cam_tracks:
            tid = track['tid']
            gid = global_ids.get(tid)
            emb = embs.get(tid)
            
            if gid is not None:
                self.registry.assign_global_id(
                    cam_id, tid, gid, emb, now
                )
        
        # Update zone gate positions
        if self.zone_gate:
            for track in cam_tracks:
                self.zone_gate.update_position(cam_id, track['tid'], track['bbox'])
        
        # Analytics
        if self.track_analytics:
            events = self.track_analytics.update(cam_id, cam_tracks, global_ids)
        
        # Visualization
        vis = frame.copy()
        vis = draw_detections(
            vis, cam_tracks,
            cfg=self.draw_cfg,
            cam_id=cam_id,
            global_ids=global_ids
        )
        
        # Write video
        self.writer.write(
            cam_id, vis,
            fps=self.reader.fps(cam_id),
            size=(vis.shape[1], vis.shape[0])
        )
        
        # Write tracks
        for t in cam_tracks:
            gid = global_ids.get(t['tid'], None)
            rec = {
                'ts': now,
                'frame_idx': frame_idx,
                'camera_id': cam_id,
                'track_id': t['tid'],
                'global_id': gid,
                'bbox_xywh': list(map(float, t['bbox'])),
                'score': float(t.get('score', 1.0)),
            }
            self.tracks_writer.write(rec)
    
    def stop(self):
        """Stop the pipeline and cleanup."""
        self.running = False
        
        self.health_monitor.stop()
        self.reader.close()
        self.writer.close()
        self.tracks_writer.close()
        cv2.destroyAllWindows()
        
        # Save registry manifest
        manifest_path = self.cfg['output'].get('manifest_path')
        if manifest_path:
            self.registry.save_manifest(manifest_path)
        
        # Save aggregator stats
        if self.aggregator:
            stats_path = self.cfg['output'].get('stats_path')
            if stats_path:
                self.aggregator.save(stats_path)
        
        elapsed = time.time() - self.stats['start_time'] if self.stats['start_time'] else 0
        logger.info(f"Pipeline stopped. Processed {self.stats['frame_count']} frames in {elapsed:.1f}s")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        stats = {
            **self.stats,
            'health': self.health_monitor.get_summary(),
            'registry': self.registry.get_stats(),
        }
        
        if self.aggregator:
            stats['analytics'] = self.aggregator.get_summary()
        
        return stats
    
    # API integration methods
    def get_analytics(self):
        return self.track_analytics
    
    def get_aggregator(self):
        return self.aggregator
    
    def get_zone_manager(self):
        return self.analytics_zone_manager
    
    def get_registry(self):
        return self.registry
    
    def get_health_monitor(self):
        return self.health_monitor
