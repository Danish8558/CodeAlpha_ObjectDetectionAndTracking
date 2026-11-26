"""Streamlit Web UI for YOLOv8 Object Detection and Multi-Object Tracking.

Run:
  streamlit run app.py
"""
import os
import tempfile
import time
from typing import Dict

import cv2
import streamlit as st

from detector import TRACKERS, DetectorTracker

st.set_page_config(
    page_title="Object Detection & Tracking | CodeAlpha",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom header styling
st.markdown(
    """
    <style>
    .hero-container {
        padding: 1.5rem 1.8rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #0f172a 0%, #0369a1 50%, #059669 100%);
        color: #ffffff;
        margin-bottom: 1.5rem;
    }
    .hero-container h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
    }
    .hero-container p {
        margin: 0.4rem 0 0;
        font-size: 1.05rem;
        opacity: 0.95;
    }
    .stMetric {
        background-color: rgba(255, 255, 255, 0.04);
        padding: 10px;
        border-radius: 8px;
    }
    </style>
    <div class="hero-container">
        <h1>🎯 Real-Time Object Detection & Tracking</h1>
        <p>YOLOv8 deep learning detection coupled with ByteTrack, BoT-SORT, and classic SORT tracking.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading YOLOv8 weights and tracker...")
def load_engine(weights: str, tracker: str) -> DetectorTracker:
    return DetectorTracker(weights=weights, tracker=tracker)


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    weights = st.selectbox(
        "YOLOv8 Model Weights",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        index=0,
        help="yolov8n (nano) is fastest; yolov8m is most accurate.",
    )

    tracker = st.selectbox(
        "Tracking Algorithm",
        TRACKERS,
        index=0,
        format_func=lambda t: {
            "bytetrack": "ByteTrack (Fast & Accurate)",
            "botsort": "BoT-SORT (Motion + Appearance)",
            "sort": "SORT (Kalman Filter + Hungarian)",
        }[t],
    )

    conf = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.35,
        step=0.05,
        help="Minimum detection confidence to consider a bounding box.",
    )

    iou = st.slider(
        "IoU Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Intersection-over-Union threshold for non-max suppression and tracking overlap.",
    )

    trail_len = st.slider(
        "Trajectory Trail Length",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
        help="Number of historical centroid points retained for object movement paths.",
    )

    engine = load_engine(weights, tracker)
    engine.conf = conf
    engine.iou = iou
    engine.trail_len = trail_len

    # Resolving class list
    names_dict = (
        engine.names
        if isinstance(engine.names, dict)
        else {i: n for i, n in enumerate(engine.names)}
    )
    all_names = list(names_dict.values())
    picked = st.multiselect(
        "Filter Specific Classes (Empty = All)",
        options=all_names,
        default=[],
        help="Select one or more classes to detect and track exclusively.",
    )

    lookup = {str(v).lower(): k for k, v in names_dict.items()}
    engine.classes = [lookup[str(n).lower()] for n in picked if str(n).lower() in lookup] if picked else None

    frame_skip = st.slider(
        "Process Every N-th Frame",
        min_value=1,
        max_value=5,
        value=1,
        help="Skipping frames can boost performance on CPU.",
    )

    st.markdown("---")
    source = st.radio(
        "Select Input Source",
        ["Upload Video", "Webcam (Local OpenCV)", "Webcam (Browser WebRTC)"],
    )

# Main UI layout
left_col, right_col = st.columns([3, 1])

with left_col:
    frame_placeholder = st.empty()

with right_col:
    st.subheader("📊 Live Telemetry")
    fps_metric = st.empty()
    st.markdown("##### Current Frame Counts")
    current_counts_box = st.empty()
    st.markdown("##### Unique Objects Tracked")
    unique_counts_box = st.empty()


def update_telemetry(counts: Dict[str, int], unique: Dict[str, int], current_fps: float):
    """Update Streamlit dashboard telemetry tables and metrics."""
    fps_metric.metric("Processing Speed", f"{current_fps:.1f} FPS")

    if counts:
        current_counts_box.dataframe(
            {"Class": list(counts.keys()), "In Frame": list(counts.values())},
            hide_index=True,
            use_container_width=True,
        )
    else:
        current_counts_box.info("No objects detected in current frame.")

    if unique:
        unique_counts_box.dataframe(
            {"Class": list(unique.keys()), "Total Unique IDs": list(unique.values())},
            hide_index=True,
            use_container_width=True,
        )
    else:
        unique_counts_box.info("No unique objects registered yet.")


def run_video_stream(cap: cv2.VideoCapture, stop_button_key: str):
    """Process video capture loop and display live annotated frames."""
    engine.reset()
    stop_button = st.button("⏹️ Stop Processing", key=stop_button_key, type="secondary")
    frame_count = 0

    try:
        while cap.isOpened() and not stop_button:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            annotated_frame, counts, unique = engine.process(frame)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            update_telemetry(counts, unique, engine.fps)
    finally:
        cap.release()


# 1. Video file upload mode
if source == "Upload Video":
    uploaded_file = st.file_uploader(
        "Choose a video file to detect and track",
        type=["mp4", "avi", "mov", "mkv"],
    )

    if uploaded_file is not None:
        st.info(f"Video uploaded: **{uploaded_file.name}** ({uploaded_file.size / (1024*1024):.2f} MB)")
        start_btn = st.button("▶️ Start Detection & Tracking", type="primary")

        if start_btn:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                temp_path = tmp.name

            try:
                cap = cv2.VideoCapture(temp_path)
                if not cap.isOpened():
                    st.error("Failed to read video file.")
                else:
                    run_video_stream(cap, "stop_uploaded_video")
                    st.success("🎉 Video processing completed successfully!")
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

# 2. Local OpenCV webcam
elif source == "Webcam (Local OpenCV)":
    st.info("Direct camera access via OpenCV (ideal when running locally on Windows/Linux/macOS).")
    cam_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
    if st.button("▶️ Start Local Webcam", type="primary"):
        cap = cv2.VideoCapture(int(cam_index))
        if not cap.isOpened():
            st.error(f"Cannot open webcam with index {cam_index}. Please verify device connection.")
        else:
            run_video_stream(cap, "stop_local_webcam")

# 3. Browser WebRTC webcam
else:
    st.info("Browser webcam streaming via WebRTC (compatible with cloud hosting & remote browsers).")
    try:
        import av
        from streamlit_webrtc import webrtc_streamer

        class WebRTCState:
            counts: Dict[str, int] = {}
            unique: Dict[str, int] = {}

        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            annotated_img, WebRTCState.counts, WebRTCState.unique = engine.process(img)
            return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

        webrtc_ctx = webrtc_streamer(
            key="codealpha-yolo-tracker",
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
        )

        if webrtc_ctx and webrtc_ctx.state and webrtc_ctx.state.playing:
            while webrtc_ctx.state.playing:
                update_telemetry(WebRTCState.counts, WebRTCState.unique, engine.fps)
                time.sleep(0.4)
    except ImportError:
        st.warning(
            "The `streamlit-webrtc` and `av` packages are required for in-browser streaming.\n\n"
            "To enable browser webcam, run: `pip install streamlit-webrtc av`"
        )
