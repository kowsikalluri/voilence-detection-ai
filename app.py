import os
import time
import threading
import smtplib

from collections import deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

import av
import cv2
import numpy as np
import streamlit as st

from keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Live Violence Detection",
    page_icon="🚨",
    layout="wide",
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "modelnew.h5"

SNAPSHOT_DIR = "snapshots"

VIOLENCE_THRESHOLD = 0.90

SNAPSHOT_INTERVAL = 10

EMAIL_INTERVAL = 30


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():

    if not os.path.isfile(MODEL_PATH):

        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' was not found."
        )

    return load_model(MODEL_PATH)


# ============================================================
# EMAIL ALERT
# ============================================================

def send_email_alert(image_path):

    sender_email = os.getenv(
        "ALERT_SENDER_EMAIL"
    )

    sender_password = os.getenv(
        "ALERT_SENDER_PASSWORD"
    )

    receiver_email = os.getenv(
        "ALERT_RECEIVER_EMAIL"
    )

    # Email is optional
    if not all(
        [
            sender_email,
            sender_password,
            receiver_email,
        ]
    ):
        return

    try:

        message = MIMEMultipart()

        message["From"] = sender_email
        message["To"] = receiver_email
        message["Subject"] = "🚨 Violence Detected"

        body = """
Violence has been detected by the
AI Violence Detection System.

A snapshot has been attached.
"""

        message.attach(
            MIMEText(
                body,
                "plain"
            )
        )

        with open(
            image_path,
            "rb"
        ) as file:

            attachment = MIMEBase(
                "application",
                "octet-stream"
            )

            attachment.set_payload(
                file.read()
            )

        encoders.encode_base64(
            attachment
        )

        attachment.add_header(
            "Content-Disposition",
            f"attachment; filename={os.path.basename(image_path)}"
        )

        message.attach(
            attachment
        )

        with smtplib.SMTP(
            "smtp.gmail.com",
            587,
            timeout=20
        ) as server:

            server.starttls()

            server.login(
                sender_email,
                sender_password
            )

            server.send_message(
                message
            )

    except Exception as error:

        print(
            f"Email alert failed: {error}"
        )


# ============================================================
# VIOLENCE DETECTOR
# ============================================================

class ViolenceDetector(
    VideoProcessorBase
):

    def __init__(self):

        self.model = None

        self.prediction_queue = deque(
            maxlen=10
        )

        self.last_snapshot_time = 0

        self.last_email_time = 0

        os.makedirs(
            SNAPSHOT_DIR,
            exist_ok=True
        )

        try:

            self.model = get_model()

            print(
                "Violence detection model loaded."
            )

        except Exception as error:

            print(
                f"Model loading error: {error}"
            )


    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )


        # ====================================================
        # MODEL ERROR
        # ====================================================

        if self.model is None:

            cv2.putText(
                image,
                "MODEL ERROR",
                (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3
            )

            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24"
            )


        # ====================================================
        # PREPROCESSING
        # ====================================================

        resized = cv2.resize(
            image,
            (128, 128)
        )

        normalized = (
            resized.astype(
                np.float32
            ) / 255.0
        )

        input_data = np.expand_dims(
            normalized,
            axis=0
        )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        try:

            prediction = self.model.predict(
                input_data,
                verbose=0
            )

            prediction = np.asarray(
                prediction
            )

            violence_score = float(
                np.ravel(
                    prediction
                )[0]
            )

        except Exception as error:

            print(
                f"Prediction error: {error}"
            )

            cv2.putText(
                image,
                "PREDICTION ERROR",
                (25, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                3
            )

            return av.VideoFrame.from_ndarray(
                image,
                format="bgr24"
            )


        # ====================================================
        # VIOLENCE CLASSIFICATION
        # ====================================================

        self.prediction_queue.append(
            violence_score
        )

        violence = (
            violence_score
            >= VIOLENCE_THRESHOLD
        )


        if violence:

            text = (
                f"VIOLENCE: YES "
                f"({violence_score:.2%})"
            )

            text_color = (
                0,
                0,
                255
            )

        else:

            text = (
                f"VIOLENCE: NO "
                f"({violence_score:.2%})"
            )

            text_color = (
                0,
                255,
                0
            )


        # ====================================================
        # DISPLAY RESULT
        # ====================================================

        cv2.rectangle(
            image,
            (10, 10),
            (600, 75),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            image,
            text,
            (25, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            text_color,
            3
        )


        # ====================================================
        # SNAPSHOT + EMAIL
        # ====================================================

        current_time = time.time()


        if violence:

            if (
                current_time
                - self.last_snapshot_time
                >= SNAPSHOT_INTERVAL
            ):

                timestamp = int(
                    current_time
                )

                image_path = os.path.join(
                    SNAPSHOT_DIR,
                    f"violence_{timestamp}.jpg"
                )

                cv2.imwrite(
                    image_path,
                    image
                )

                self.last_snapshot_time = (
                    current_time
                )


                # Email alert
                if (
                    current_time
                    - self.last_email_time
                    >= EMAIL_INTERVAL
                ):

                    threading.Thread(
                        target=send_email_alert,
                        args=(image_path,),
                        daemon=True
                    ).start()

                    self.last_email_time = (
                        current_time
                    )


        # ====================================================
        # RETURN FRAME
        # ====================================================

        return av.VideoFrame.from_ndarray(
            image,
            format="bgr24"
        )


# ============================================================
# USER INTERFACE
# ============================================================

st.title(
    "🚨 Real-Time Violence Detection"
)

st.markdown(
    """
This application uses a trained deep-learning
model to detect possible violence through your
browser camera.

### Features

- 🎥 Real-time browser webcam detection
- 🚨 Violence / non-violence classification
- 📊 Prediction confidence
- 📸 Automatic evidence snapshots
- 📧 Optional email alerts
"""
)


# ============================================================
# MODEL CHECK
# ============================================================

if not os.path.isfile(
    MODEL_PATH
):

    st.error(
        f"""
❌ `{MODEL_PATH}` is missing.

Make sure the model file is located
in the same GitHub repository as app.py.
"""
    )

    st.stop()


try:

    get_model()

    st.success(
        "✅ AI model loaded successfully."
    )

except Exception as error:

    st.error(
        f"""
❌ Model could not be loaded.

{error}
"""
    )

    st.stop()


# ============================================================
# TURN / WEBRTC CONFIGURATION
# ============================================================

try:

    turn_username = st.secrets[
        "turn"
    ][
        "username"
    ]

    turn_credential = st.secrets[
        "turn"
    ][
        "credential"
    ]

except Exception:

    turn_username = None

    turn_credential = None


if turn_username and turn_credential:

    rtc_configuration = {

        "iceServers": [

            # STUN
            {
                "urls": [
                    "stun:stun.relay.metered.ca:80"
                ]
            },

            # TURN UDP
            {
                "urls": [
                    "turn:global.relay.metered.ca:80"
                ],
                "username": turn_username,
                "credential": turn_credential,
            },

            # TURN TCP
            {
                "urls": [
                    "turn:global.relay.metered.ca:80?transport=tcp"
                ],
                "username": turn_username,
                "credential": turn_credential,
            },

            # TURN 443
            {
                "urls": [
                    "turn:global.relay.metered.ca:443"
                ],
                "username": turn_username,
                "credential": turn_credential,
            },

            # TURN TLS / TCP 443
            {
                "urls": [
                    "turns:global.relay.metered.ca:443?transport=tcp"
                ],
                "username": turn_username,
                "credential": turn_credential,
            },
        ]
    }

else:

    # Fallback STUN-only configuration
    rtc_configuration = {

        "iceServers": [

            {
                "urls": [
                    "stun:stun.relay.metered.ca:80"
                ]
            }

        ]
    }

    st.warning(
        """
⚠️ TURN credentials are not configured.

Camera connectivity may fail on restrictive
networks. Add the TURN credentials to
Streamlit Secrets.
"""
    )


# ============================================================
# CAMERA
# ============================================================

st.info(
    """
Click START and allow camera permission
when your browser asks.
"""
)


webrtc_streamer(

    key="violence_detection_camera",

    video_processor_factory=ViolenceDetector,

    media_stream_constraints={
        "video": True,
        "audio": False,
    },

    rtc_configuration=rtc_configuration,

    async_processing=True,
)


# ============================================================
# EMAIL INFORMATION
# ============================================================

st.divider()

st.subheader(
    "📧 Optional Email Alerts"
)

st.write(
    """
Email alerts are optional.

Credentials should be stored using
Streamlit Secrets or environment variables.
"""
)


# ============================================================
# SNAPSHOTS
# ============================================================

st.subheader(
    "📸 Saved Snapshots"
)

if os.path.exists(
    SNAPSHOT_DIR
):

    snapshots = sorted(

        [
            os.path.join(
                SNAPSHOT_DIR,
                filename
            )

            for filename
            in os.listdir(
                SNAPSHOT_DIR
            )

            if filename.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            )
        ],

        reverse=True
    )


    if snapshots:

        number_of_columns = min(
            3,
            len(snapshots)
        )

        columns = st.columns(
            number_of_columns
        )


        for index, path in enumerate(
            snapshots[:6]
        ):

            with columns[
                index
                % number_of_columns
            ]:

                st.image(
                    path,
                    caption=os.path.basename(
                        path
                    )
                )

    else:

        st.caption(
            "No violence snapshots captured yet."
        )
