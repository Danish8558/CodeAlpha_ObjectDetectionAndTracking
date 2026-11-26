"""SORT: Simple Online and Realtime Tracking (Bewley et al., 2016).

Kalman filter per track (constant velocity on box center, area and aspect ratio)
plus Hungarian assignment on IoU (scipy.optimize.linear_sum_assignment).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N,4) and b (M,4), boxes as x1,y1,x2,y2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    a = a[:, None, :]
    b = b[None, :, :]
    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return inter / np.maximum(union, 1e-9)


def box_to_z(box):
    """Convert bounding box [x1, y1, x2, y2] to Kalman state measurement [cx, cy, s, r]."""
    w = max(float(box[2] - box[0]), 1e-6)
    h = max(float(box[3] - box[1]), 1e-6)
    return np.array([box[0] + w / 2.0, box[1] + h / 2.0, w * h, w / h], dtype=float).reshape(4, 1)


def x_to_box(x):
    """Convert Kalman state vector [cx, cy, s, r] back to [x1, y1, x2, y2]."""
    x = np.asarray(x, dtype=float).ravel()
    s = max(float(x[2]), 1e-6)
    r = max(float(x[3]), 1e-6)
    w = np.sqrt(max(s * r, 1e-6))
    h = s / w
    cx, cy = float(x[0]), float(x[1])
    return np.array([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dtype=float)


class KalmanBoxTracker:
    """State: [cx, cy, s, r, vcx, vcy, vs]."""
    count = 0

    def __init__(self, box, cls=-1, score=0.0):
        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((4, 7))
        self.H[:4, :4] = np.eye(4)
        self.R = np.eye(4)
        self.R[2:, 2:] *= 10.0
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 1000.0
        self.Q = np.eye(7)
        self.Q[-1, -1] *= 0.01
        self.Q[4:, 4:] *= 0.01
        self.x = np.zeros((7, 1))
        self.x[:4] = box_to_z(box)

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0
        self.cls = cls
        self.score = score

    def predict(self):
        """Advance the state vector and return the predicted bounding box estimate."""
        if self.x[2, 0] + self.x[6, 0] <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return x_to_box(self.x)

    def update(self, box, cls=-1, score=0.0):
        """Update the state vector with observed bounding box."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.cls = cls
        self.score = score
        z = box_to_z(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        try:
            K = self.P @ self.H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ self.H.T @ np.linalg.pinv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def state(self):
        """Return the current bounding box estimate."""
        return x_to_box(self.x)


class Sort:
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0

    def reset(self):
        """Clear all active tracks and reset frame counter."""
        self.trackers = []
        self.frame_count = 0

    def update(self, dets: np.ndarray) -> np.ndarray:
        """Update tracks from detections.

        dets: (N, 4..6) as [x1, y1, x2, y2, (score), (cls)].
        Returns array of shape (M, 7): [x1, y1, x2, y2, track_id, cls, score].
        """
        self.frame_count += 1
        if dets is None or len(dets) == 0:
            dets = np.empty((0, 6), dtype=float)
        else:
            dets = np.asarray(dets, dtype=float)
            if dets.ndim == 1:
                dets = dets.reshape(1, -1)
            if dets.shape[1] == 4:
                dets = np.hstack([dets, np.ones((len(dets), 1)), -np.ones((len(dets), 1))])
            elif dets.shape[1] == 5:
                dets = np.hstack([dets, -np.ones((len(dets), 1))])

        preds = []
        for t in self.trackers:
            p = t.predict()
            preds.append(p)
        preds = np.array(preds).reshape(-1, 4) if preds else np.empty((0, 4))
        valid = ~np.any(np.isnan(preds), axis=1) if len(preds) else np.array([], dtype=bool)
        self.trackers = [t for t, v in zip(self.trackers, valid) if v]
        preds = preds[valid] if len(preds) else np.empty((0, 4))

        matched = []
        unmatched_dets = list(range(len(dets)))
        if len(preds) > 0 and len(dets) > 0:
            iou = iou_batch(dets[:, :4], preds)
            rows, cols = linear_sum_assignment(-iou)
            unmatched_dets = [d for d in range(len(dets)) if d not in rows]
            for r, c in zip(rows, cols):
                if iou[r, c] < self.iou_threshold:
                    unmatched_dets.append(r)
                else:
                    matched.append((r, c))

        for d, t in matched:
            self.trackers[t].update(dets[d, :4], int(dets[d, 5]), float(dets[d, 4]))
        for d in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(dets[d, :4], int(dets[d, 5]), float(dets[d, 4])))

        out = []
        alive = []
        for t in self.trackers:
            if t.time_since_update <= self.max_age:
                alive.append(t)
            if t.time_since_update < 1 and (t.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                out.append([*t.state(), t.id, t.cls, t.score])
        self.trackers = alive
        return np.array(out, dtype=float) if out else np.empty((0, 7), dtype=float)


if __name__ == "__main__":
    tracker = Sort(min_hits=1)
    for f in range(10):
        dets = np.array([[10 + 5 * f, 10, 60 + 5 * f, 60, 0.9, 0],
                         [200, 100 + 3 * f, 260, 180 + 3 * f, 0.8, 2]])
        tracks = tracker.update(dets)
    print("Self-test succeeded. Active tracks:", tracks[:, 4:6])
