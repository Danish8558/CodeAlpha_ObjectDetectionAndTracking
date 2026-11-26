"""Real-time object detection and tracking from a webcam or video file (OpenCV CLI).

Usage Examples:
  # Run live webcam tracking with ByteTrack
  python detect_track.py --source 0

  # Run video file with standalone SORT tracker
  python detect_track.py --source traffic.mp4 --tracker sort

  # Filter specific classes and record output to video
  python detect_track.py --source traffic.mp4 --classes person car --save output.mp4

Controls:
  Press 'q' or 'ESC' to exit.
"""
import argparse
import os
import sys

import cv2
import numpy as np

from detector import TRACKERS, DetectorTracker


def parse_args():
    parser = argparse.ArgumentParser(
        description="CodeAlpha: Real-time Object Detection and Tracking with YOLOv8"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (e.g., 0) or path to an input video file",
    )
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="YOLOv8 weights file (e.g., yolov8n.pt, yolov8s.pt)",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack",
        choices=TRACKERS,
        help="Tracking algorithm: bytetrack, botsort, or sort",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold (0.05 to 0.95)",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="NMS / tracking IoU threshold",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Specific class names to detect/track, e.g. person car dog",
    )
    parser.add_argument(
        "--trail",
        type=int,
        default=30,
        help="Length of historical motion trajectory trails",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional path to save annotated output video (e.g. output.mp4)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the OpenCV display window (useful for headless / batch processing)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine input source
    if args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source
        if not os.path.exists(source):
            print(f"Error: Video file not found: {source}", file=sys.stderr)
            sys.exit(1)

    print(f"Initializing detector with model '{args.weights}' and tracker '{args.tracker}'...")
    engine = DetectorTracker(
        weights=args.weights,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        trail=args.trail,
    )

    # Class filter mapping
    if args.classes:
        names_dict = (
            engine.names
            if isinstance(engine.names, dict)
            else {i: n for i, n in enumerate(engine.names)}
        )
        lookup = {str(v).lower(): k for k, v in names_dict.items()}
        selected_ids = []
        for c in args.classes:
            c_low = c.lower()
            if c_low in lookup:
                selected_ids.append(lookup[c_low])
            else:
                print(f"Warning: Class '{c}' not found in model classes.")
        if selected_ids:
            engine.classes = selected_ids
            print(f"Filtering to classes: {[names_dict[i] for i in selected_ids]}")

    print(f"Opening video source: {args.source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source: {args.source}", file=sys.stderr)
        sys.exit(1)

    writer = None
    frames_processed = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frames_processed += 1
            out, counts, unique = engine.process(frame)

            # Setup video writer if saving
            if args.save and writer is None:
                h, w = out.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps is None or fps <= 0 or np.isnan(fps):
                    fps = 25.0
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
                print(f"Recording output to '{args.save}' ({w}x{h} @ {fps:.1f} FPS)...")

            if writer:
                writer.write(out)

            if not args.no_show:
                window_title = "CodeAlpha Object Detection & Tracking (Press Q to quit)"
                cv2.imshow(window_title, out)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("Quit requested by user.")
                    break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"Saved annotated video to: {args.save}")
        if not args.no_show:
            cv2.destroyAllWindows()

    print(f"\nProcessing complete. Total frames processed: {frames_processed}")
    print("Unique objects tracked across session:")
    for cls_name, total in unique.items():
        print(f"  - {cls_name}: {total}")


if __name__ == "__main__":
    main()
