"""
app.py
Smart Classroom Engagement Tracker — Streamlit Dashboard

Run:
    streamlit run app.py
"""

import time
from collections import deque

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import yaml

from face_utils import FaceTracker, draw_face
from model import Predictor

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Classroom Tracker", page_icon="🎓", layout="wide")

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

CLASSES = cfg["classes"]
COLORS  = {"attentive": "#2ECC71", "distracted": "#F39C12", "disengaged": "#E74C3C"}
LOW_ENGAGEMENT_THRESHOLD = cfg["thresholds"]["alert"]


# ── Cached resources (loaded once) ────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    return Predictor(cfg["model"]["weights"])

@st.cache_resource
def load_tracker():
    return FaceTracker(max_faces=10)


# ── Session state ─────────────────────────────────────────────────────────────
def init():
    if "running" not in st.session_state:
        st.session_state.running  = False
        st.session_state.cap      = None
        st.session_state.history  = []         # list of (timestamp, pct_attentive)
        st.session_state.window   = deque(maxlen=200)  # recent labels for live %
        st.session_state.fps      = 0.0
        st.session_state.t_fps    = time.time()
        st.session_state.n_frames = 0

init()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🎓 Classroom Engagement Tracker")
    st.markdown("---")
    cam_idx = st.number_input("Camera index", 0, 5, 0)
    st.markdown("---")
    st.markdown("**Model:** MobileNetV2  \n**Landmarks:** MediaPipe FaceMesh  \n**Classes:** Attentive · Distracted · Disengaged")
    st.markdown("---")
    if st.button("Clear Session Data"):
        st.session_state.history = []
        st.session_state.window  = deque(maxlen=200)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🎓 Classroom Engagement Tracker")
st.caption("Real-time student focus detection using MobileNetV2 + MediaPipe")

# ── Start / Stop buttons ──────────────────────────────────────────────────────
c1, c2, c3 = st.columns([1, 1, 6])
with c1:
    if st.button("▶ Start", type="primary", disabled=st.session_state.running):
        cap = cv2.VideoCapture(int(cam_idx), cv2.CAP_DSHOW)
        # Set camera to use direct backend and allow for proper initialization
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to get latest frames
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Test that camera opens properly
        ok, test_frame = cap.read()
        if not ok or test_frame is None:
            st.error(f"Failed to open camera at index {cam_idx}. Check the camera index in the sidebar.")
            cap.release()
        else:
            st.session_state.cap     = cap
            st.session_state.running = True

with c2:
    if st.button("⏹ Stop", disabled=not st.session_state.running):
        if st.session_state.cap:
            st.session_state.cap.release()
        st.session_state.running = False

# Add a "Next Frame" button for manual control
with c3:
    if st.session_state.running:
        st.caption("📹 Live feed active — processing every frame automatically")
    else:
        st.caption("Click Start to begin camera capture")

# Export CSV button (only when not running and have data)
if st.session_state.history and not st.session_state.running:
    import pandas as pd, io
    df = pd.DataFrame(st.session_state.history, columns=["timestamp", "pct_attentive"])
    df["pct_attentive"] = df["pct_attentive"].round(1)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("📥 Export CSV", csv_buf.getvalue(),
                       file_name="session_report.csv", mime="text/csv")

st.markdown("---")

# ── Layout: webcam left, metrics right ───────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    frame_box = st.empty()

with right:
    st.markdown("#### Live Metrics")
    g1, g2, g3 = st.columns(3)
    gauge_attn  = g1.empty()
    gauge_dist  = g2.empty()
    gauge_dis   = g3.empty()
    metric_row  = st.empty()

chart_box   = st.empty()
alert_box   = st.empty()


# ── Gauge helper ──────────────────────────────────────────────────────────────
def gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 13}},
        number={"suffix": "%", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": color},
            "steps": [
                {"range": [0,  40], "color": "#fde8e8"},
                {"range": [40, 70], "color": "#fef9e7"},
                {"range": [70, 100],"color": "#eafaf1"},
            ],
        },
    ))
    fig.update_layout(height=170, margin=dict(t=30, b=5, l=5, r=5),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ── Main loop ─────────────────────────────────────────────────────────────────
predictor = load_predictor()
tracker   = load_tracker()

# ── Main processing loop ──────────────────────────────────────────────────────
if st.session_state.running:
    cap = st.session_state.cap
    if cap is None or not cap.isOpened():
        st.error("Camera disconnected. Click Start again.")
        st.session_state.running = False
        st.rerun()

    while st.session_state.running:
        ok, frame = cap.read()
        if not ok or frame is None:
            st.warning("Camera not accessible. Check the camera index.")
            st.session_state.running = False
            break

        frame = cv2.flip(frame, 1)

        # FPS tracking
        st.session_state.n_frames += 1
        curr_time = time.time()
        if curr_time - st.session_state.t_fps >= 1.0:
            st.session_state.fps      = st.session_state.n_frames
            st.session_state.n_frames = 0
            st.session_state.t_fps    = curr_time

        # Detect + predict
        faces = tracker.process(frame)
        for face in faces:
            label, conf, score = predictor.predict(face["roi"])
            if conf < 0.55:
                if face["is_drowsy"] or face["is_yawning"]:
                    label = "disengaged"
                elif face["is_away"]:
                    label = "distracted"
            st.session_state.window.append(label)
            draw_face(frame, face, label, score)

        # Stats overlay
        n   = len(st.session_state.window) or 1
        pct = {cls: st.session_state.window.count(cls) / n * 100 for cls in CLASSES}

        h_f, w_f = frame.shape[:2]
        panel = f"Attn:{pct['attentive']:.0f}%  Dist:{pct['distracted']:.0f}%  FPS:{st.session_state.fps}"
        cv2.rectangle(frame, (w_f-260, 5), (w_f-5, 28), (30, 30, 30), -1)
        cv2.putText(frame, panel, (w_f-255, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.47, (220, 220, 220), 1)

        # Render video
        frame_box.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        # Render Metrics efficiently (skip Plotly rendering every frame)
        if st.session_state.n_frames % 5 == 0:
            # We use st.progress to simulate gauges because plotly causes heavy lag and flickering
            gauge_attn.markdown(f"**✅ Attentive**: {pct['attentive']:.0f}%")
            gauge_dist.markdown(f"**⚠️ Distracted**: {pct['distracted']:.0f}%")
            gauge_dis.markdown(f"**❌ Disengaged**: {pct['disengaged']:.0f}%")
            
            with metric_row.container():
                m1, m2, m3 = st.columns(3)
                m1.metric("Faces Detected", len(faces))
                m2.metric("FPS", f"{st.session_state.fps:.0f}")
                m3.metric("Alert Threshold", f"{LOW_ENGAGEMENT_THRESHOLD}%")

        # Record history every ~5 s
        if st.session_state.n_frames % max(1, int(st.session_state.fps * 5)) == 0:
            st.session_state.history.append(
                (round(curr_time, 1), round(pct["attentive"], 1))
            )

        if len(st.session_state.history) >= 2 and st.session_state.n_frames % 15 == 0:
            ts   = [i * 5 for i in range(len(st.session_state.history))]
            vals = [h[1] for h in st.session_state.history]
            fig  = go.Figure()
            fig.add_trace(go.Scatter(x=ts, y=vals, mode="lines+markers",
                                     line=dict(color="#2ECC71", width=2),
                                     fill="tozeroy", fillcolor="rgba(46,204,113,0.13)",
                                     name="Attentive %"))
            fig.add_hline(y=LOW_ENGAGEMENT_THRESHOLD, line_dash="dash",
                          line_color="red", annotation_text="Alert threshold")
            fig.update_layout(title="Attentive % Over Session",
                              xaxis_title="Elapsed (s)", yaxis_title="%",
                              yaxis=dict(range=[0, 100]), height=250,
                              margin=dict(t=40, b=30, l=40, r=20),
                              paper_bgcolor="rgba(0,0,0,0)")
            chart_box.plotly_chart(fig, use_container_width=True)

        if pct["attentive"] < LOW_ENGAGEMENT_THRESHOLD and len(faces) > 0:
            alert_box.error(
                f"⚠️ Low engagement! Only {pct['attentive']:.0f}% attentive. "
                "Consider a break or Q&A."
            )
        else:
            alert_box.empty()

        time.sleep(0.03)

    st.rerun()

# ── Post-session summary ──────────────────────────────────────────────────────
elif st.session_state.history:
    frame_box.info("📹 Click Start to begin a new session")
    st.markdown("---")
    st.markdown("### 📊 Session Summary")

    vals = [h[1] for h in st.session_state.history]
    col1, col2, col3 = st.columns(3)
    col1.metric("Avg Attentive %",      f"{np.mean(vals):.1f}%")
    col2.metric("Min Attentive %",      f"{np.min(vals):.1f}%")
    col3.metric("Duration (snapshots)", len(vals))

    ts  = [i * 5 for i in range(len(vals))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=vals, mode="lines+markers",
                             line=dict(color="#2ECC71", width=2),
                             fill="tozeroy", fillcolor="rgba(46,204,113,0.13)"))
    fig.add_hline(y=LOW_ENGAGEMENT_THRESHOLD, line_dash="dash", line_color="red")
    fig.update_layout(title="Full Session — Attentive %",
                      xaxis_title="Elapsed (s)", yaxis_title="%",
                      height=280, margin=dict(t=40, b=30, l=40, r=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

else:
    frame_box.info("📹 Click Start to begin camera capture")