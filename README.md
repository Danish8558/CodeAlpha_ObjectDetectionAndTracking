# CodeAlpha AI Task: Real-Time Object Detection and Tracking

An end-to-end computer vision solution delivering real-time object detection and multi-object tracking (MOT) using **YOLOv8** alongside **ByteTrack**, **BoT-SORT**, and a standalone **SORT** tracker.

Featuring an interactive **Streamlit Web UI** with telemetry and a high-performance **OpenCV CLI** viewer with video recording and class filtering.

---

## 🌟 Key Features

- **State-of-the-Art Detection**: Powered by Ultralytics **YOLOv8** (`yolov8n`, `yolov8s`, `yolov8m`) pre-trained on the COCO dataset (80 object classes).
- **Multiple Multi-Object Trackers**:
  - **ByteTrack**: High-speed, high-accuracy two-stage association retaining low-confidence detections.
  - **BoT-SORT**: Advanced motion compensation and appearance-aware tracking.
  - **SORT (Custom Built)**: Standalone implementation from scratch using a Kalman Filter (constant velocity motion model) and the Hungarian algorithm for IoU association via `scipy.optimize.linear_sum_assignment`.
- **Persistent Trajectories**: Renders real-time movement trails showing the historical motion path of each tracked identity.
- **Live Telemetry & Analytics**:
  - Real-time Exponential Moving Average (EMA) FPS overlay.
  - Per-frame active object count by class.
  - Cumulative unique object IDs tracked across the entire session.
- **Multiple Input Sources**:
  - Video files (`.mp4`, `.avi`, `.mov`, `.mkv`).
  - Local OpenCV webcams.
  - In-browser camera streaming via **Streamlit WebRTC**.
- **Class Filtering & Recording**: Isolate specific classes (e.g., only `person` and `car`) and export annotated videos with synchronized audio/video framerates.

---

## 📁 Repository Structure

```
CodeAlpha_ObjectDetectionTracking/
├── app.py              # Streamlit Web Application with interactive dashboard & WebRTC
├── detector.py         # Unified YOLOv8 detection + tracking engine with HUD and trails
├── sort.py             # Standalone SORT tracker (Kalman Filter + Hungarian matching)
├── detect_track.py     # Command-line interface (CLI) for real-time OpenCV playback/recording
├── requirements.txt    # Project dependencies
├── .gitignore          # Git exclusion rules (ignores models, temp files, caches)
└── README.md           # Documentation and setup instructions
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Danish8558/CodeAlpha_ObjectDetectionAndTracking.git
cd CodeAlpha_ObjectDetectionAndTracking
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🖥️ Usage

### A. Streamlit Web Dashboard
Launch the interactive web application:
```bash
streamlit run app.py
```
- Select your model weights and tracking algorithm in the sidebar.
- Adjust confidence, IoU, and trajectory trail length.
- Upload any video or select local webcam / browser camera streaming.
- View live telemetry tables showing objects in frame and unique objects tracked.

### B. Command-Line Interface (OpenCV Window)

#### Run live webcam tracking:
```bash
python detect_track.py --source 0
```

#### Run video file with standalone SORT tracker:
```bash
python detect_track.py --source traffic.mp4 --tracker sort
```

#### Filter specific classes (e.g. only person and car):
```bash
python detect_track.py --source traffic.mp4 --classes person car
```

#### Record annotated output video:
```bash
python detect_track.py --source traffic.mp4 --save output.mp4
```

#### CLI Options Reference:
| Option | Default | Description |
|---|---|---|
| `--source` | `0` | Camera index (`0`) or video file path |
| `--weights` | `yolov8n.pt` | YOLOv8 model weights file |
| `--tracker` | `bytetrack` | Tracker choice: `bytetrack`, `botsort`, `sort` |
| `--conf` | `0.35` | Confidence score threshold |
| `--iou` | `0.5` | IoU association threshold |
| `--classes` | `None` | Class names to keep (e.g., `--classes person car`) |
| `--trail` | `30` | Length of motion trajectory trail |
| `--save` | `None` | Output file path to save annotated video |
| `--no-show` | `False` | Run in headless mode without opening display window |

---

## 🧠 Tracker Comparison

| Tracker | Association Method | Motion Model | Appearance Features | Best For |
|---|---|---|---|---|
| **ByteTrack** | Two-stage IoU association | Kalman Filter | No | Dense crowds, fast-moving scenes |
| **BoT-SORT** | IoU + ReID cosine distance | Camera Motion + KF | Yes (optional) | Complex occlusions, erratic camera |
| **SORT** | Hungarian IoU matching | 7-State Kalman Filter | No | Lightweight, transparent, educational |

---

## 🧪 Self-Test & Verification

Verify the standalone SORT tracker:
```bash
python sort.py
```

Run CLI self-test with dummy input:
```bash
python -c "from detector import DetectorTracker; dt = DetectorTracker(); print('Detector loaded successfully!')"
```

---

## 📜 License & Acknowledgments
Developed for the **CodeAlpha AI Internship**.
- Detection: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- SORT algorithm based on Bewley et al., *Simple Online and Realtime Tracking* (2016).
