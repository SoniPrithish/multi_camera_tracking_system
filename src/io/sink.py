
import os, cv2, numpy as np

class VideoWriterMux:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.writers = {}
        self.last_frames = {}

    def write(self, cam_id, frame, fps=25, size=None):
        self.last_frames[cam_id] = frame.copy()
        if cam_id not in self.writers:
            h, w = frame.shape[:2]
            if size is None:
                size = (w, h)
            path = os.path.join(self.out_dir, f'{cam_id}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writers[cam_id] = cv2.VideoWriter(path, fourcc, fps, size)
        self.writers[cam_id].write(frame)

    def latest_tiled(self, cols=2):
        if not self.last_frames:
            return None
        frames = list(self.last_frames.values())
        # Resize to smallest h
        h = min(f.shape[0] for f in frames)
        rsz = [cv2.resize(f, (int(f.shape[1]*h/f.shape[0]), h)) for f in frames]
        # Pad to equal width
        w = max(f.shape[1] for f in rsz)
        pad = [cv2.copyMakeBorder(f,0,0,0,w-f.shape[1],cv2.BORDER_CONSTANT,value=(20,20,20)) for f in rsz]
        # Tile
        rows = (len(pad)+cols-1)//cols
        tiles = []
        for r in range(rows):
            row = pad[r*cols:(r+1)*cols]
            if len(row) < cols:
                row += [np.zeros_like(pad[0])] * (cols - len(row))
            tiles.append(cv2.hconcat(row))
        return cv2.vconcat(tiles)

    def close(self):
        for w in self.writers.values():
            w.release()

class TracksWriter:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, 'w', encoding='utf-8')

    def write(self, rec: dict):
        import json
        self.f.write(json.dumps(rec) + '\n')

    def close(self):
        self.f.close()
