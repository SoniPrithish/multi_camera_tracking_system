
import numpy as np

def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax+aw, bx+bw)
    inter_y2 = min(ay+ah, by+bh)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    union = aw*ah + bw*bh - inter
    return inter / union

class CentroidTracker:
    """Very simple per-camera tracker via greedy IoU matching."""
    def __init__(self, iou_thresh=0.1, max_age=30):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_tid = 1
        self.tracks = {}  # cam_id -> {tid: {'bbox':..., 'age':...}}

    def update(self, cam_id, detections):
        tracks = self.tracks.setdefault(cam_id, {})
        used = set()
        outputs = []

        # Try to match existing tracks
        for tid, st in list(tracks.items()):
            st['age'] += 1
            best = (-1, None)
            for j, det in enumerate(detections):
                if j in used: 
                    continue
                iou = iou_xywh(st['bbox'], det['bbox'])
                if iou > best[0]:
                    best = (iou, j)
            if best[1] is not None and best[0] >= self.iou_thresh:
                j = best[1]
                st['bbox'] = detections[j]['bbox']
                st['age'] = 0
                used.add(j)
            if st['age'] > self.max_age:
                tracks.pop(tid, None)
            else:
                outputs.append({'tid': tid, 'bbox': st['bbox'], 'score': 1.0})

        # Initialize new tracks for unmatched detections
        for j, det in enumerate(detections):
            if j in used:
                continue
            tid = self.next_tid
            self.next_tid += 1
            tracks[tid] = {'bbox': det['bbox'], 'age': 0}
            outputs.append({'tid': tid, 'bbox': det['bbox'], 'score': 1.0})

        return outputs
