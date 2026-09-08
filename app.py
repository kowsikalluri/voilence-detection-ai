import os
import time
import threading
import smtplib
import uuid
import shutil
from pathlib import Path
from collections import deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import cv2
import numpy as np
import streamlit as st
from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

st.set_page_config(layout="wide", page_title="Live Violence Detection", page_icon="🚨")

MODEL_PATH = "modelnew.h5"
SNAPSHOT_ROOT = Path("/tmp/violence_detection_sessions")
VIOLENCE_THRESHOLD = 0.90
SNAPSHOT_INTERVAL = 10
EMAIL_INTERVAL = 30
STALE_SESSION_SECONDS = 15 * 60


def get_session_id():
    if "violence_session_id" not in st.session_state:
        st.session_state.violence_session_id = uuid.uuid4().hex
    return st.session_state.violence_session_id


SESSION_ID = get_session_id()
SESSION_SNAPSHOT_DIR = SNAPSHOT_ROOT / SESSION_ID
SESSION_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_stale_sessions():
    """Remove old temporary session folders without touching active sessions."""
    SNAPSHOT_ROOT.mkdir(parents=True, exist_ok=True)
    now = time.time()
    try:
        for item in SNAPSHOT_ROOT.iterdir():
            if not item.is_dir() or item.name == SESSION_ID:
                continue
            try:
                if now - item.stat().st_mtime > STALE_SESSION_SECONDS:
                    shutil.rmtree(item, ignore_errors=True)
            except OSError:
                pass
    except OSError:
        pass


cleanup_stale_sessions()


@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' was not found. Place the .h5 model in the same folder as app.py."
        )
    return load_model(MODEL_PATH)


def send_email_alert(image_path):
    sender_email = os.getenv("ALERT_SENDER_EMAIL")
    sender_password = os.getenv("ALERT_SENDER_PASSWORD")
    receiver_email = os.getenv("ALERT_RECEIVER_EMAIL")
    if not all([sender_email, sender_password, receiver_email]):
        return
    try:
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = "⚠️ Violence Detected with Snapshot"
        msg.attach(MIMEText("Violence detected. Snapshot attached.", "plain"))
        with open(image_path, "rb") as attachment:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(image_path)}")
        msg.attach(part)
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
    except Exception as exc:
        print(f"Email alert failed: {exc}")


class ViolenceDetector(VideoProcessorBase):
    def __init__(self, session_snapshot_dir):
        self.model = get_model()
        self.prediction_queue = deque(maxlen=128)
        self.last_snapshot_time = 0
        self.last_email_time = 0
        self.snapshot_dir = Path(session_snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        resized = cv2.resize(image, (128, 128))
        normalized = resized.astype("float32") / 255.0
        prediction = self.model.predict(np.expand_dims(normalized, axis=0), verbose=0)[0]
        self.prediction_queue.append(prediction)

        violence_score = float(np.ravel(prediction)[0])
        violence = violence_score > VIOLENCE_THRESHOLD
        text = (
            f"Violence: YES ({violence_score:.2%})"
            if violence else f"Violence: NO ({violence_score:.2%})"
        )
        text_color = (0, 0, 255) if violence else (0, 255, 0)
        cv2.putText(image, text, (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 3)

        current_time = time.time()
        if violence and current_time - self.last_snapshot_time >= SNAPSHOT_INTERVAL:
            timestamp = int(current_time)
            image_path = self.snapshot_dir / f"violence_{timestamp}.jpg"
            cv2.imwrite(str(image_path), image)

            if current_time - self.last_email_time >= EMAIL_INTERVAL:
                threading.Thread(
                    target=send_email_alert,
                    args=(str(image_path),),
                    daemon=True,
                ).start()
                self.last_email_time = current_time
            self.last_snapshot_time = current_time

        return av.VideoFrame.from_ndarray(image, format="bgr24")


st.title("🚨 Real-Time Violence Detection")
st.markdown(
    """
This application uses your trained deep-learning model to detect
possible violence through the browser camera.

**Features**
- 🎥 Real-time browser webcam detection
- 🚨 Violence / non-violence classification
- 📊 Prediction confidence
- 📸 Temporary session-only evidence snapshots
- 📧 Optional email alerts

**Privacy:** Snapshots captured by your camera are isolated to your
current browser session and are not shown to other users.
"""
)

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ `{MODEL_PATH}` is missing. Rename your uploaded model to `modelnew.h5` and place it beside `app.py`.")
    st.stop()

try:
    get_model()
    st.success("✅ Model loaded successfully.")
except Exception as exc:
    st.error(f"❌ Model could not be loaded: {exc}")
    st.stop()

st.info("Click START below and allow camera permission when your browser asks.")

# KEEP THE WORKING WEBRTC/TURN CONFIGURATION UNCHANGED
try:
    turn_username = st.secrets["turn"]["username"]
    turn_credential = st.secrets["turn"]["credential"]
except Exception:
    turn_username = None
    turn_credential = None

if turn_username and turn_credential:
    rtc_configuration = {
        "iceServers": [
            {"urls": ["stun:stun.relay.metered.ca:80"]},
            {"urls": ["turn:global.relay.metered.ca:80"], "username": turn_username, "credential": turn_credential},
            {"urls": ["turn:global.relay.metered.ca:80?transport=tcp"], "username": turn_username, "credential": turn_credential},
            {"urls": ["turn:global.relay.metered.ca:443"], "username": turn_username, "credential": turn_credential},
            {"urls": ["turns:global.relay.metered.ca:443?transport=tcp"], "username": turn_username, "credential": turn_credential},
        ]
    }
else:
    rtc_configuration = {"iceServers": [{"urls": ["stun:stun.relay.metered.ca:80"]}]}

webrtc_streamer(
    key="violence_detection_camera",
    video_processor_factory=lambda: ViolenceDetector(SESSION_SNAPSHOT_DIR),
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration=rtc_configuration,
    async_processing=True,
)

st.divider()
st.subheader("📧 Optional Email Alerts")
st.write("For deployment, email credentials are read from Streamlit Secrets / environment variables rather than being stored in GitHub.")
st.caption("Set ALERT_SENDER_EMAIL, ALERT_SENDER_PASSWORD and ALERT_RECEIVER_EMAIL only if you want email notifications.")

st.subheader("📸 Your Session Snapshots")
snapshots = sorted(
    [
        path for path in SESSION_SNAPSHOT_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png")
    ],
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

if snapshots:
    cols = st.columns(min(3, len(snapshots)))
    for index, path in enumerate(snapshots[:6]):
        with cols[index % len(cols)]:
            st.image(str(path), caption=path.name)
else:
    st.caption("No violence snapshots captured in this session yet.")

st.caption("🔒 These snapshots belong only to your current browser session. They are stored temporarily and old session folders are automatically cleaned up.")
