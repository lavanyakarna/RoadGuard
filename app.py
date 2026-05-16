import cv2
import streamlit as st
from ultralytics import YOLO
import numpy as np
import time
import threading
import os
from datetime import datetime
import pygame

# Initialize pygame for sound
pygame.mixer.init()

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Classes we care about
HAZARD_CLASSES = {
    0: "Person",
    1: "Bicycle",
    2: "Car",
    15: "Cat",
    16: "Dog",
    19: "Cattle"
}

# Alert colors (BGR)
ALERT_COLORS = {
    "GREEN": (0, 255, 0),
    "YELLOW": (0, 255, 255),
    "RED": (0, 0, 255)
}

# Create clips folder if not exists
os.makedirs("clips", exist_ok=True)

# ── CLAHE low-light enhancement ──
def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    return enhanced

# ── Proximity based alert level ──
def get_alert_level(box_area, frame_area):
    ratio = box_area / frame_area
    if ratio > 0.01:
        return "RED"
    elif ratio > 0.005:
        return "YELLOW"
    else:
        return "YELLOW"

# ── Play alert sound ──
def play_alert_sound():
    try:
        pygame.mixer.music.load("alert.wav")
        pygame.mixer.music.play()
    except:
        pass

# ── Save clip when RED triggers ──
def save_clip(frames, fps=20):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"clips/alert_{timestamp}.avi"
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    return filename

# ── Process each frame ──
def process_frame(frame):
    enhanced = enhance_frame(frame)
    results = model(enhanced, verbose=False)[0]
    alert_level = "GREEN"
    detections = []
    frame_area = frame.shape[0] * frame.shape[1]

    for box in results.boxes:
        cls = int(box.cls[0])
        if cls in HAZARD_CLASSES:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            box_area = (x2 - x1) * (y2 - y1)
            level = get_alert_level(box_area, frame_area)

            if level == "RED":
                alert_level = "RED"
            elif alert_level != "RED":
                alert_level = "YELLOW"

            color = ALERT_COLORS[level]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{HAZARD_CLASSES[cls]} {conf:.2f} [{level}]"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            detections.append({
                "class": HAZARD_CLASSES[cls],
                "confidence": round(conf, 2),
                "alert": level
            })

    if alert_level == "GREEN":
        banner = "ROAD CLEAR"
    elif alert_level == "YELLOW":
        banner = "CAUTION - HAZARD DETECTED"
    else:
        banner = "DANGER - BRAKE NOW!"

    color = ALERT_COLORS[alert_level]
    cv2.putText(frame, banner, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

    return frame, alert_level, detections


# ────────────────────────────────────────
#  STREAMLIT UI
# ────────────────────────────────────────
st.set_page_config(page_title="RoadGuard", layout="wide", page_icon="🚗")
st.title("🚗 RoadGuard — Road Hazard Detection System")
st.caption("Real-time multi-class road hazard detection with proximity alerts")

col1, col2 = st.columns([3, 1])

with col2:
    st.subheader("📊 Live Dashboard")
    alert_box = st.empty()
    st.divider()
    st.markdown("**Session Stats**")
    total_box = st.empty()
    red_box = st.empty()
    clips_box = st.empty()
    st.divider()
    st.markdown("**Recent Detections**")
    log_box = st.empty()

with col1:
    source = st.radio("Input Source", ["Webcam", "Upload Video"], horizontal=True)

    if source == "Webcam":
        run = st.toggle("▶ Start Detection")
        frame_window = st.empty()

        if run:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("Cannot access webcam")
            else:
                detection_log = []
                total_detections = 0
                red_alerts = 0
                clips_saved = 0
                frame_buffer = []
                last_sound_time = 0
                last_clip_time = 0

                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Lost webcam feed")
                        break

                    frame, alert_level, detections = process_frame(frame)

                    frame_buffer.append(frame.copy())
                    if len(frame_buffer) > 60:
                        frame_buffer.pop(0)

                    total_detections += len(detections)

                    if alert_level == "RED":
                        red_alerts += 1
                        if time.time() - last_sound_time > 2:
                            threading.Thread(target=play_alert_sound).start()
                            last_sound_time = time.time()
                        if len(frame_buffer) >= 20 and time.time() - last_clip_time > 30:
                            threading.Thread(
                                target=save_clip,
                                args=(frame_buffer.copy(),)
                            ).start()
                            clips_saved += 1
                            last_clip_time = time.time()

                    if detections:
                        detection_log = detections + detection_log
                        detection_log = detection_log[:5]

                    if alert_level == "RED":
                        alert_box.error("🔴 DANGER — BRAKE NOW!")
                    elif alert_level == "YELLOW":
                        alert_box.warning("🟡 CAUTION — Hazard Detected")
                    else:
                        alert_box.success("🟢 Road Clear")

                    total_box.metric("Total Detections", total_detections)
                    red_box.metric("RED Alerts", red_alerts)
                    clips_box.metric("Clips Saved", clips_saved)

                    if detection_log:
                        log_text = "\n".join(
                            [f"- {d['class']} ({d['confidence']}) [{d['alert']}]"
                             for d in detection_log]
                        )
                        log_box.markdown(log_text)

                    frame_window.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True
                    )

            cap.release()

    elif source == "Upload Video":
        uploaded = st.file_uploader("Upload a road video", type=["mp4", "avi", "mov"])

        if uploaded:
            with open("temp_video.mp4", "wb") as f:
                f.write(uploaded.read())

            st.success("Video uploaded! Click Process to start.")
            if st.button("▶ Process Video"):
                cap = cv2.VideoCapture("temp_video.mp4")
                frame_window = st.empty()
                alert_box2 = st.empty()
                detection_log = []
                total_detections = 0
                red_alerts = 0
                clips_saved = 0
                frame_buffer = []
                last_sound_time = 0
                last_clip_time = 0

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame, alert_level, detections = process_frame(frame)
                    total_detections += len(detections)

                    frame_buffer.append(frame.copy())
                    if len(frame_buffer) > 60:
                        frame_buffer.pop(0)

                    if alert_level == "RED":
                        red_alerts += 1
                        if time.time() - last_sound_time > 2:
                            threading.Thread(target=play_alert_sound).start()
                            last_sound_time = time.time()
                        if len(frame_buffer) >= 20 and time.time() - last_clip_time > 30:
                            threading.Thread(
                                target=save_clip,
                                args=(frame_buffer.copy(),)
                            ).start()
                            clips_saved += 1
                            last_clip_time = time.time()

                    if detections:
                        detection_log = detections + detection_log
                        detection_log = detection_log[:5]

                    if alert_level == "RED":
                        alert_box2.error("🔴 DANGER — BRAKE NOW!")
                    elif alert_level == "YELLOW":
                        alert_box2.warning("🟡 CAUTION — Hazard Detected")
                    else:
                        alert_box2.success("🟢 Road Clear")

                    if detection_log:
                        log_text = "\n".join(
                            [f"- {d['class']} ({d['confidence']}) [{d['alert']}]"
                             for d in detection_log]
                        )
                        log_box.markdown(log_text)

                    frame_window.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True
                    )

                cap.release()
                st.success(f"✅ Done! Total detections: {total_detections} | RED alerts: {red_alerts} | Clips saved: {clips_saved}")