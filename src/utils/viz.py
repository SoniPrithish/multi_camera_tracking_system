
import cv2

def draw_detections(frame, tracks, cfg=None, cam_id=None, global_ids=None):
    cfg = cfg or {}
    for t in tracks:
        x, y, w, h = map(int, t['bbox'])
        gid = global_ids.get(t['tid']) if global_ids else None
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        label_parts = []
        if cfg.get('show_cam', True) and cam_id:
            label_parts.append(str(cam_id))
        if cfg.get('show_id', True):
            label_parts.append(f"ID {t['tid']}")
        if gid is not None:
            label_parts.append(f"G{gid}")
        label = " ".join(label_parts)
        if label:
            cv2.putText(frame, label, (x, max(0, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return frame
