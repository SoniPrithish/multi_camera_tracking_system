
import cv2

class MultiVideoReader:
    def __init__(self, inputs):
        self.caps = {}
        for item in inputs:
            cam_id = item.get('camera_id')
            path = str(item.get('path'))
            cap = cv2.VideoCapture(path if not path.isdigit() else int(path))
            if not cap.isOpened():
                print(f'[WARN] Could not open {path}')
            self.caps[cam_id] = cap

    def read(self):
        frames = {}
        any_ok = False
        for cam_id, cap in list(self.caps.items()):
            ok, frame = cap.read()
            if ok:
                frames[cam_id] = frame
                any_ok = True
        if not any_ok:
            return None
        return frames

    def fps(self, cam_id):
        cap = self.caps.get(cam_id)
        if cap:
            fps = cap.get(cv2.CAP_PROP_FPS)
            return fps if fps and fps > 0 else 25
        return 25

    def close(self):
        for cap in self.caps.values():
            cap.release()
