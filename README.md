# 🎓 Classroom Engagement Tracker

A real-time student engagement monitoring system that uses Computer Vision and Deep Learning to identify whether students are attentive, distracted, or disengaged during a class session.

The project analyzes facial features, eye movements, head orientation, and other visual cues from a webcam feed and provides live engagement insights through an interactive dashboard.

Developed as a B.Tech Artificial Intelligence & Machine Learning project by **Viplav Bhure** and **Poush Makade**.

---

## 📖 Introduction

In online and large classroom environments, it can be difficult for instructors to continuously monitor student attention levels. The Classroom Engagement Tracker aims to assist educators by automatically analyzing student engagement in real time.

Using facial analysis techniques and a deep learning model, the system classifies students into three categories:

* ✅ **Attentive** – Focused and actively participating
* ⚠️ **Distracted** – Looking away or showing signs of reduced attention
* ❌ **Disengaged** – Drowsy, inactive, or uninterested

The system provides live visual feedback and engagement statistics through a Streamlit dashboard.

---

## ✨ Features

* Real-time engagement detection using webcam input
* Deep learning-based classification using MobileNetV2
* Face detection and facial landmark tracking
* Eye Aspect Ratio (EAR) analysis for drowsiness detection
* Mouth Aspect Ratio (MAR) analysis for yawn detection
* Head pose estimation for attention tracking
* Interactive Streamlit dashboard
* Live engagement score visualization
* Session report generation in CSV format
* Support for both public and custom datasets

---

## 🛠️ Technology Stack

* Python
* PyTorch
* MobileNetV2
* OpenCV
* MediaPipe
* Streamlit
* Plotly
* NumPy

---

## 📂 Project Structure

```text
classroom_tracker/
│
├── app.py
├── model.py
├── face_utils.py
├── train.py
├── collect_data.py
├── config.yaml
├── requirements.txt
│
├── data/
├── weights/
└── exports/
```

### File Description

| File              | Description                             |
| ----------------- | --------------------------------------- |
| `app.py`          | Main Streamlit dashboard                |
| `model.py`        | Model architecture and prediction logic |
| `face_utils.py`   | Face detection and feature extraction   |
| `train.py`        | Model training script                   |
| `collect_data.py` | Dataset collection utility              |
| `config.yaml`     | Configuration settings                  |
| `weights/`        | Saved trained models                    |
| `exports/`        | Generated session reports               |

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/Viplav-Bhure/classroom-tracker.git
cd classroom-tracker
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset Preparation

### Option 1: DAiSEE Dataset

Download the DAiSEE dataset from Kaggle:

https://www.kaggle.com/datasets/joyee19/studentengagement

Extract the dataset into the `data/` directory.

### Option 2: Create Your Own Dataset

Capture training samples using:

```bash
python collect_data.py --class attentive --n 150
python collect_data.py --class distracted --n 150
python collect_data.py --class disengaged --n 150
```

Controls:

* **SPACE** → Capture image
* **Q** → Quit recording

---

## 🧠 Model Training

Train the model using:

```bash
python train.py
```

Training process:

* Uses pretrained MobileNetV2 weights
* Initially freezes the backbone layers
* Performs fine-tuning in later epochs
* Saves the best-performing model automatically

Output:

```text
weights/model.pth
```

---

## ▶️ Running the Application

Start the dashboard:

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually):

```text
http://localhost:8501
```

Click **Start** and allow camera access to begin monitoring.

---

## ⚙️ System Workflow

```text
Webcam Feed
     │
     ▼
Face Detection
     │
     ▼
Feature Extraction
(EAR, MAR, Head Pose)
     │
     ▼
MobileNetV2 Classification
     │
     ▼
Engagement Prediction
     │
     ▼
Dashboard Visualization
```

---

## 🔍 How It Works

### Face Detection

The system first detects faces from each video frame using OpenCV and MediaPipe.

### Eye Aspect Ratio (EAR)

EAR is used to detect prolonged eye closure, which may indicate drowsiness or disengagement.

### Mouth Aspect Ratio (MAR)

MAR helps identify yawning behavior, which can be a sign of fatigue.

### Head Pose Estimation

The orientation of the head is analyzed to determine whether the student is looking toward the screen or away from it.

### Deep Learning Classification

The extracted facial region is passed through a MobileNetV2 model, which predicts one of the following classes:

* Attentive
* Distracted
* Disengaged

---

## 📈 Dashboard Output

The dashboard displays:

* Current engagement state
* Prediction confidence
* Engagement score
* Real-time graphs
* Session statistics
* Exportable CSV reports

---

## 📊 Performance

| Metric              | Value              |
| ------------------- | ------------------ |
| Validation Accuracy | ~88%               |
| Macro F1 Score      | ~0.86              |
| Inference Speed     | ~25 ms/frame (CPU) |

Performance may vary depending on hardware, lighting conditions, and camera quality.

---

## 🎯 Applications

* Smart classrooms
* Online learning platforms
* Student engagement analysis
* Educational research
* Automated classroom monitoring

---

## 🔮 Future Improvements

* Multi-student engagement tracking
* Attendance integration
* Teacher analytics dashboard
* Cloud deployment
* Emotion recognition
* Advanced reporting and analytics

---

## 🏁 Conclusion

The Classroom Engagement Tracker demonstrates how Artificial Intelligence and Computer Vision can be used to improve classroom monitoring and learning outcomes. By combining facial feature analysis with deep learning, the system provides real-time insights into student engagement, helping educators better understand and support their students.

---

## 👨‍💻 Authors

**Viplav Bhure**
B.Tech Artificial Intelligence & Machine Learning

**Poush Makade**
B.Tech Artificial Intelligence & Machine Learning
