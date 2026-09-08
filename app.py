import os
import time
import threading
import smtplib
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


# -------------------- Streamlit Config --------------------
st.set_page_config(
    layout="wide",
    page_title="Live Violence Detection",
    page_icon="🚨",
)


# -------------------- Configuration --------------------
MODEL_PATH = "modelnew.h5"
SNAPSHOT_DIR = "snapshots"
VIOLENCE_THRESHOLD = 0.90
SNAPSHOT_INTERVAL = 10
EMAIL_INTERVAL = 30


# -------------------- Load Model --------------------
@st.cache_resource
def get_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' was not found. "
            "Place the .h5 model in the same folder as app.py."
        )
    return load_model(MODEL_PATH)


# -------------------- Email Alert --------------------
def send_email_alert(image_path):
    sender_email = os.getenv("ALERT_SENDER_EMAIL")
    sender_password = os.getenv("ALERT_SENDER_PASSWORD")
    receiver_email = os.getenv("ALERT_RECEIVER_EMAIL")

    # Email is optional. The app still works if credentials aren't configured.
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
        part.add_header(
            "Content-Disposition",
            f"attachment; filename={os.path.basename(image_path)}",
        )
        msg.attach(part)

        with smtplib.SMTP("smtp.gmail.com", 587, timeout=20) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

    except Exception as exc:
        print(f"Email alert failed: {exc}")


# -------------------- Webcam Processor --------------------
class ViolenceDetector(VideoProcessorBase):
    def __init__(self):
        self.model = get_model()
        self.prediction_queue = deque(maxlen=128)
        self.last_snapshot_time = 0
        self.last_email_time = 0

        os.makedirs(SNAPSHOT_DIR, exist_ok=True)

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Model input: 128x128 RGB image, normalized to [0, 1].
        resized = cv2.resize(image, (128, 128))
        normalized = resized.astype("float32") / 255.0

        prediction = self.model.predict(
            np.expand_dims(normalized, axis=0),
            verbose=0,
        )[0]

        self.prediction_queue.append(prediction)

        # Preserve the original project's threshold logic.
        violence_score = float(np.ravel(prediction)[0])
        violence = violence_score > VIOLENCE_THRESHOLD

        text = (
            f"Violence: YES ({violence_score:.2%})"
            if violence
            else f"Violence: NO ({violence_score:.2%})"
        )

        text_color = (0, 0, 255) if violence else (0, 255, 0)

        cv2.putText(
            image,
            text,
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            text_color,
            3,
        )

        current_time = time.time()

        if violence:
            # Capture a snapshot at most once every 10 seconds.
            if current_time - self.last_snapshot_time >= SNAPSHOT_INTERVAL:
                timestamp = int(current_time)
                image_path = os.path.join(
                    SNAPSHOT_DIR,
                    f"violence_{timestamp}.jpg",
                )

                cv2.imwrite(image_path, image)

                # Send email at most once every 30 seconds.
                if current_time - self.last_email_time >= EMAIL_INTERVAL:
                    threading.Thread(
                        target=send_email_alert,
                        args=(image_path,),
                        daemon=True,
                    ).start()
                    self.last_email_time = current_time

                self.last_snapshot_time = current_time

        return av.VideoFrame.from_ndarray(image, format="bgr24")


# -------------------- UI --------------------
st.title("🚨 Real-Time Violence Detection")

st.markdown(
    """
This application uses your trained deep-learning model to detect
possible violence through the browser camera.

**Features**
- 🎥 Real-time browser webcam detection
- 🚨 Violence / non-violence classification
- 📊 Prediction confidence
- 📸 Automatic evidence snapshots
- 📧 Optional email alerts
"""
)

if not os.path.exists(MODEL_PATH):
    st.error(
        f"❌ `{MODEL_PATH}` is missing. "
        "Rename your uploaded model to `modelnew.h5` and place it beside `app.py`."
    )
    st.stop()

try:
    get_model()
    st.success("✅ Model loaded successfully.")
except Exception as exc:
    st.error(f"❌ Model could not be loaded: {exc}")
    st.stop()

st.info(
    "Click START below and allow camera permission when your browser asks."
)

webrtc_streamer(
    key="violence-detection",
    video_processor_factory=ViolenceDetector,
    media_stream_constraints={
        "video": True,
        "audio": False,
    },
    async_processing=True,
)

st.divider()

st.subheader("📧 Optional Email Alerts")
st.write(
    "For deployment, email credentials are read from Streamlit Secrets / "
    "environment variables rather than being stored in GitHub."
)

st.caption(
    "Set ALERT_SENDER_EMAIL, ALERT_SENDER_PASSWORD and "
    "ALERT_RECEIVER_EMAIL only if you want email notifications."
)

st.subheader("📸 Saved Snapshots")
if os.path.exists(SNAPSHOT_DIR):
    snapshots = sorted(
        [
            os.path.join(SNAPSHOT_DIR, f)
            for f in os.listdir(SNAPSHOT_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ],
        reverse=True,
    )

    if snapshots:
        cols = st.columns(min(3, len(snapshots)))
        for index, path in enumerate(snapshots[:6]):
            with cols[index % len(cols)]:
                st.image(path, caption=os.path.basename(path))
    else:
        st.caption("No violence snapshots captured yet.")
