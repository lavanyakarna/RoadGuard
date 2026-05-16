# 🚗 RoadGuard — Real-Time Road Hazard Detection System

A computer vision system that detects road hazards from a live dashcam or webcam feed and alerts the driver before a collision occurs — built for Indian road conditions.

---

## 🚨 The Problem

India records over 1.5 lakh road accident deaths every year. A significant portion happen at night on highways and semi-urban roads where stray cattle, pedestrians, and cyclists are nearly invisible in low light. Drivers have no warning system. RoadGuard is built to change that.

This project is an evolution of my earlier [cattle-alert](https://github.com/lavanyakarna/cattle-alert) system, expanded to detect all road hazards with a proximity-based alert system.
---

## 💡 What It Does

- Detects road hazards in real time: **cattle, persons, cyclists, dogs, cars**
- Enhances low-light frames using **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
- **3-tier proximity alert system:**
  - 🟢 GREEN — Road clear
  - 🟡 YELLOW — Hazard detected at safe distance
  - 🔴 RED — Hazard dangerously close, brake now
- Plays **audio alarm** on RED alert
- **Auto-saves clips** whenever RED alert fires
- Live **dashboard** with detection stats
- Works on **live webcam** and **uploaded video**

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| YOLOv8 | Real-time object detection |
| OpenCV | Video processing + CLAHE |
| Python | Core pipeline |
| Streamlit | Live dashboard and UI |
| Pygame | Audio alert |

---

## ⚙️ Setup

```bash
git clone https://github.com/lavanyakarna/RoadGuard.git
cd RoadGuard
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## 👩‍💻 Built By

**Lavanya Karna** — B.Tech CSE (AI & ML), VIT Bhopal
[GitHub](https://github.com/lavanyakarna)
