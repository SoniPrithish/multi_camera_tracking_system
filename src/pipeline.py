
import os, json, time
import cv2
import numpy as np
from .io.streams import MultiVideoReader
from .io.sink import VideoWriterMux, TracksWriter
from .utils.viz import draw_detections
from .detection import build_detector
from .tracking import build_tracker
from .reid import build_reid

class MultiCamPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.reader = MultiVideoReader(cfg['inputs'])
        os.makedirs(cfg['output']['video_dir'], exist_ok=True)
        self.writer = VideoWriterMux(cfg['output']['video_dir'])
        self.tracks_writer = TracksWriter(cfg['output']['tracks_path'])
        self.detector = build_detector(cfg['modules']['detector'])
        self.tracker = build_tracker(cfg['modules']['tracker'])
        self.reid = build_reid(cfg['modules']['reid'])
        self.runtime = cfg.get('runtime', {})
        self.draw_cfg = self.runtime.get('draw', {})

    def run(self):
        max_frames = self.runtime.get('max_frames')
        skip_frames = int(self.runtime.get('skip_frames', 1))
        show = bool(self.cfg['output'].get('show', False))
        frame_idx = 0

        while True:
            batch = self.reader.read()
            if batch is None:
                break
            frame_idx += 1
            if skip_frames > 1 and frame_idx % skip_frames != 0:
                continue

            for cam_id, frame in batch.items():
                dets = self.detector.detect(frame)
                # Per-camera tracking
                cam_tracks = self.tracker.update(cam_id, dets)
                # ReID embeddings (one per track bbox)
                embs = self.reid.encode(frame, cam_tracks)
                # Cross-camera association
                global_ids = self.reid.assign_global_ids(cam_id, cam_tracks, embs)
                # Draw + write
                vis = frame.copy()
                vis = draw_detections(vis, cam_tracks, cfg=self.draw_cfg, cam_id=cam_id, global_ids=global_ids)
                self.writer.write(cam_id, vis, fps=self.reader.fps(cam_id), size=(vis.shape[1], vis.shape[0]))

                # Persist track outputs
                now = time.time()
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

            if show:
                tiled = self.writer.latest_tiled()
                if tiled is not None:
                    cv2.imshow('MULTI_CAMERA_TRACKING', tiled)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            if max_frames is not None and frame_idx >= int(max_frames):
                break

        self.reader.close()
        self.writer.close()
        self.tracks_writer.close()
        cv2.destroyAllWindows()
