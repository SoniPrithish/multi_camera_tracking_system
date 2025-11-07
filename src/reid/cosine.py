
import numpy as np
import cv2

class CosineReID:
    """Toy ReID that uses color histograms as embeddings + cosine similarity across cameras."""
    def __init__(self, sim_thresh=0.9):
        self.sim_thresh = sim_thresh
        self.gallery = {}  # global_id -> embedding
        self.next_gid = 1

    def encode(self, frame, tracks):
        embs = {}
        for t in tracks:
            x,y,w,h = map(int, t['bbox'])
            crop = frame[max(0,y):y+h, max(0,x):x+w]
            if crop.size == 0:
                vec = np.zeros(64, dtype=np.float32)
            else:
                hist = []
                for ch in range(3):
                    hch = cv2.calcHist([crop],[ch],None,[8],[0,256]).flatten()
                    hist.append(hch)
                vec = np.concatenate(hist).astype(np.float32)
                n = np.linalg.norm(vec)
                if n > 0:
                    vec = vec / (n + 1e-6)
            embs[t['tid']] = vec
        return embs

    def assign_global_ids(self, cam_id, tracks, embs):
        # Greedy match to existing gallery by cosine similarity
        gids = {}
        for t in tracks:
            e = embs.get(t['tid'])
            if e is None or e.sum() == 0:
                gid = self.next_gid; self.next_gid += 1
                self.gallery[gid] = e
                gids[t['tid']] = gid
                continue
            # find best
            best_gid, best_sim = None, -1.0
            for gid, gemb in self.gallery.items():
                if gemb is None or (hasattr(gemb, 'sum') and gemb.sum() == 0):
                    continue
                denom = (np.linalg.norm(e)*np.linalg.norm(gemb) + 1e-6)
                sim = float(np.dot(e, gemb) / denom)
                if sim > best_sim:
                    best_sim = sim; best_gid = gid
            if best_sim >= self.sim_thresh and best_gid is not None:
                gids[t['tid']] = best_gid
                # update gallery with moving average
                self.gallery[best_gid] = 0.5*self.gallery[best_gid] + 0.5*e
            else:
                gid = self.next_gid; self.next_gid += 1
                self.gallery[gid] = e
                gids[t['tid']] = gid
        return gids
