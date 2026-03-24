import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import mediapipe as mp

# MediaPipe Initialisierung
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # MediaPipe Bearbeitung
        img.flags.writeable = False
        results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img.flags.writeable = True

        # Zeichne die Landmarks auf das Bild
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("MediaPipe Body Tracking")
webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
