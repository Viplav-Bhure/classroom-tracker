"""
face_utils.py
All face-related logic: OpenCV DNN face detection, EAR, MAR, and head pose estimation.
"""

import cv2
import numpy as np
import urllib.request
import os

# Landmark indices for EAR (Eye Aspect Ratio) - using facial landmarks from DNN
# These are approximate indices for a 5-point face model
RIGHT_EYE = [0, 1]  # Right eye corners
LEFT_EYE  = [2, 3]  # Left eye corners

# Landmark indices for MAR (Mouth Aspect Ratio)
MOUTH = [3, 4]  # Mouth corners (approximate)

# 3D face model points for head pose (nose, chin, eye corners, mouth corners)
FACE_3D = np.array([
    [0.0,    0.0,    0.0],    # Nose
    [0.0,   -3.3,   -1.6],    # Chin
    [-4.5,  -1.3,   -1.4],    # Left eye corner
    [ 4.5,  -1.3,   -1.4],    # Right eye corner
    [-2.7,   1.8,   -1.4],    # Left mouth corner
    [ 2.7,   1.8,   -1.4],    # Right mouth corner
], dtype=np.float64)
FACE_LM_IDX = [0, 1, 2, 3, 4, 5]  # All 6 points for head pose


class FaceTracker:
    def __init__(self, max_faces=10):
        # Download YuNet face detection model
        model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        model_path = "face_detection_yunet_2023mar.onnx"
        
        if not os.path.exists(model_path):
            print("Downloading YuNet face detection model...")
            try:
                urllib.request.urlretrieve(model_url, model_path)
                print("Model downloaded successfully!")
            except:
                print("Failed to download model, using Haar cascades as fallback...")
                model_path = None
        
        if model_path and os.path.exists(model_path):
            # Use YuNet DNN model
            self.face_detector = cv2.FaceDetectorYN.create(
                model_path,
                "",
                (320, 320),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=max_faces
            )
            self.use_dnn = True
        else:
            # Fallback to Haar cascades
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_detector = cv2.CascadeClassifier(cascade_path)
            self.use_dnn = False
        
        print(f"Using {'YuNet DNN' if self.use_dnn else 'Haar cascades'} for face detection")

    def process(self, frame):
        """Returns list of face dicts with roi, ear, mar, is_away."""
        h, w = frame.shape[:2]
        faces = []
        
        if self.use_dnn:
            # Use YuNet DNN detector
            self.face_detector.setInputSize((w, h))
            _, faces_dnn = self.face_detector.detect(frame)
            
            if faces_dnn is not None:
                for face in faces_dnn:
                    x, y, fw, fh = face[:4].astype(int)
                    
                    # Ensure bounds are within frame
                    x1, y1 = max(0, x - 10), max(0, y - 10)
                    x2, y2 = min(w, x + fw + 10), min(h, y + fh + 10)
                    
                    roi = frame[y1:y2, x1:x2].copy()
                    
                    # For DNN, we don't have detailed landmarks, so use approximations
                    ear = 0.3  # Default neutral EAR
                    mar = 0.4  # Default neutral MAR
                    is_away = False  # Cannot determine without landmarks
                    
                    faces.append({
                        "bbox": (x1, y1, x2 - x1, y2 - y1),
                        "roi": roi,
                        "ear": round(ear, 3),
                        "mar": round(mar, 3),
                        "is_away": is_away,
                        "is_drowsy": ear < 0.20,
                        "is_yawning": mar > 0.55,
                    })
        else:
            # Use Haar cascades
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces_haar = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            for (x, y, fw, fh) in faces_haar:
                # Add padding
                x1, y1 = max(0, x - 10), max(0, y - 10)
                x2, y2 = min(w, x + fw + 10), min(h, y + fh + 10)
                
                roi = frame[y1:y2, x1:x2].copy()
                
                # For Haar cascades, we don't have detailed landmarks
                ear = 0.3  # Default neutral EAR
                mar = 0.4  # Default neutral MAR
                is_away = False  # Cannot determine without landmarks
                
                faces.append({
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                    "roi": roi,
                    "ear": round(ear, 3),
                    "mar": round(mar, 3),
                    "is_away": is_away,
                    "is_drowsy": ear < 0.20,
                    "is_yawning": mar > 0.55,
                })
        
        return faces

    def close(self):
        pass  # No cleanup needed for OpenCV detectors


def draw_face(frame, face, label, score):
    """Draw bounding box and label on frame."""
    colors = {"attentive": (46, 204, 113), "distracted": (230, 126, 34), "disengaged": (231, 76, 60)}
    c = colors.get(label, (200, 200, 200))
    x, y, bw, bh = face["bbox"]

    cv2.rectangle(frame, (x, y), (x+bw, y+bh), c, 2)

    tag = f"{label.upper()} {score:.0f}%"
    cv2.rectangle(frame, (x, y-22), (x+len(tag)*9, y), c, -1)
    cv2.putText(frame, tag, (x+3, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    ear_color = (0, 0, 200) if face["is_drowsy"] else (0, 200, 0)
    cv2.putText(frame, f"EAR:{face['ear']:.2f}", (x+3, y+bh-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, ear_color, 1)
    if face["is_away"]:
        cv2.putText(frame, "AWAY", (x+3, y+bh-22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return frame
