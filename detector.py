"""YOLOv8 detection + tracking engine shared by the CLI and the Streamlit app.

Supported Trackers:
  - "bytetrack"  : Ultralytics built-in ByteTrack
  - "botsort"    : Ultralytics built-in BoT-SORT (appearance + motion aware)
  - "sort"       : Standalone SORT in sort.py (Kalman filter + Hungarian assignment)
"""
import colorsys
import time
from collections import Counter, defaultdict, deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from sort import Sort

TRACKERS = ["bytetrack", "botsort", "sort"]


def color_for(idx: int) -> Tuple[int, int, int]:
    """Generate a distinct, consistent BGR color for each track ID."""
    h = (int(idx) * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


class DetectorTracker:
    def __init__(
        self,
        weights: str = "yolov8n.pt",
        tracker: str = "bytetrack",
        conf: float = 0.35,
        iou: float = 0.5,
        classes: Optional[List[int]] = None,
        trail: int = 30,
    ):
        from ultralytics import YOLO  # lazy import for fast startup

        self.weights = weights
        self.tracker = tracker if tracker in TRACKERS else "bytetrack"
        self.conf = float(conf)
        self.iou = float(iou)
        self.classes = classes
        self.trail_len = max(1, int(trail))

        self.model = YOLO(self.weights)
        self.names = self.model.names if hasattr(self.model, "names") else {}
        self.sort: Optional[Sort] = (
            Sort(max_age=30, min_hits=3, iou_threshold=0.3) if self.tracker == "sort" else None
        )
        self.trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.trail_len))
        self.unique_ids: Dict[str, set] = defaultdict(set)
        self._t = time.time()
        self.fps = 0.0

    def reset(self):
        """Reset internal trackers, object trails, unique counts, and FPS statistics."""
        if self.tracker == "sort":
            self.sort = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
        else:
            self.sort = None
        self.trails.clear()
        self.unique_ids.clear()
        self.fps = 0.0
        self._t = time.time()

    def get_label(self, cls: int) -> str:
        """Resolve class index to name, handling both dict and list name formats."""
        if isinstance(self.names, dict):
            return str(self.names.get(cls, cls))
        if isinstance(self.names, (list, tuple)) and 0 <= cls < len(self.names):
            return str(self.names[cls])
        return str(cls)

    def _run(self, frame: np.ndarray) -> np.ndarray:
        """Run detection and tracking on a single frame.

        Returns:
            np.ndarray: Array of shape (N, 7) with columns [x1, y1, x2, y2, track_id, cls, score].
        """
        if self.tracker == "sort":
            if self.sort is None:
                self.sort = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
            res = self.model.predict(
                frame, conf=self.conf, iou=self.iou, classes=self.classes, verbose=False
            )[0]
            if res.boxes is None or len(res.boxes) == 0:
                return self.sort.update(np.empty((0, 6), dtype=float))

            b = res.boxes
            xyxy = b.xyxy.cpu().numpy()
            conf = b.conf.cpu().numpy()[:, None]
            cls = b.cls.cpu().numpy()[:, None]
            dets = np.hstack([xyxy, conf, cls])
            return self.sort.update(dets)

        # ByteTrack or BoT-SORT
        res = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            iou=self.iou,
            classes=self.classes,
            tracker=f"{self.tracker}.yaml",
            verbose=False,
        )[0]

        b = res.boxes
        if b is None or len(b) == 0 or b.id is None:
            return np.empty((0, 7), dtype=float)

        xyxy = b.xyxy.cpu().numpy()
        track_id = b.id.cpu().numpy()[:, None]
        cls = b.cls.cpu().numpy()[:, None]
        conf = b.conf.cpu().numpy()[:, None]

        # Return standardized format [x1, y1, x2, y2, track_id, cls, score]
        return np.hstack([xyxy, track_id, cls, conf])

    def process(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int]]:
        """Detect and track objects in one BGR frame.

        Returns:
            Tuple: (annotated_frame, current_frame_counts, unique_counts_so_far)
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        tracks = self._run(frame)
        counts = Counter()

        if len(tracks) > 0:
            for item in tracks:
                if len(item) < 7:
                    continue
                x1, y1, x2, y2, tid, cls, score = item[:7]
                tid = int(tid)
                cls = int(cls)
                score = float(score)

                label_name = self.get_label(cls)
                counts[label_name] += 1
                self.unique_ids[label_name].add(tid)

                col = color_for(tid)
                p1 = (max(0, min(w - 1, int(x1))), max(0, min(h - 1, int(y1))))
                p2 = (max(0, min(w - 1, int(x2))), max(0, min(h - 1, int(y2))))

                # Draw bounding box
                cv2.rectangle(annotated, p1, p2, col, 2)

                # Draw label banner safely within image boundaries
                label = f"{label_name} #{tid} {score:.2f}"
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)

                if p1[1] - th - 8 >= 0:
                    y1_rect = p1[1] - th - 8
                    y2_rect = p1[1]
                    y_text = p1[1] - 5
                else:
                    y1_rect = p1[1]
                    y2_rect = min(h - 1, p1[1] + th + 8)
                    y_text = min(h - 1, p1[1] + th + 4)

                x2_rect = min(w - 1, p1[0] + tw + 6)
                cv2.rectangle(annotated, (p1[0], y1_rect), (x2_rect, y2_rect), col, -1)
                cv2.putText(
                    annotated,
                    label,
                    (p1[0] + 3, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

                # Update trajectory trail
                cx = int((p1[0] + p2[0]) / 2)
                cy = int((p1[1] + p2[1]) / 2)
                self.trails[tid].append((cx, cy))
                pts = list(self.trails[tid])
                for i in range(1, len(pts)):
                    cv2.line(annotated, pts[i - 1], pts[i], col, 2)

        # Update FPS with Exponential Moving Average
        now = time.time()
        dt = now - self._t
        self._t = now
        if dt > 0:
            current_inst_fps = 1.0 / dt
            self.fps = (
                0.9 * self.fps + 0.1 * current_inst_fps
                if self.fps > 0
                else current_inst_fps
            )

        # Draw HUD bar
        total_in_frame = int(sum(counts.values()))
        hud = f"FPS: {self.fps:.1f} | Tracker: {self.tracker} | In Frame: {total_in_frame}"
        (hw, hh), _ = cv2.getTextSize(hud, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
        cv2.rectangle(annotated, (0, 0), (min(w, hw + 16), min(h, hh + 16)), (20, 20, 20), -1)
        cv2.putText(
            annotated,
            hud,
            (8, hh + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 120),
            1,
            cv2.LINE_AA,
        )

        unique_summary = {k: len(v) for k, v in self.unique_ids.items()}
        return annotated, dict(counts), unique_summary
