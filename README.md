# 🎓 Classroom Engagement Tracker

A real-time student engagement detection system built with MobileNetV2 and OpenCV.
Detects **Attentive / Distracted / Disengaged** students from webcam or video feed.

> B.Tech AIML 3rd Year Project by — Viplav Bhure & Poush Makade

---

## Project Structure

```
classroom_tracker/
├── app.py              ← Streamlit dashboard (main entry point)
├── model.py            ← MobileNetV2 model + dataset + predictor
├── face_utils.py       ← OpenCV face detection, EAR, MAR, head pose
├── train.py            ← Model training script
├── collect_data.py     ← Record your own training samples via webcam
├── config.yaml         ← All settings in one place
├── requirements.txt
├── data/               ← Put dataset here (attentive / distracted / disengaged)
├── weights/            ← Trained model saved here
└── exports/            ← CSV session reports
```

---

## Setup (VSCode)

**1. Clone and open**
```bash
git clone https://github.com/your-username/classroom_tracker.git
cd classroom_tracker
code .
```

**2. Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

> In VSCode: `Ctrl+Shift+P` → *Python: Select Interpreter* → pick `venv`

---

## Step-by-Step Usage

### Step 1 — Get training data (choose one or both)

**Option A: Use DAiSEE dataset (automatic download)**
```bash
# Dataset downloads automatically when you run training!
# No manual steps needed - just run python train.py
```

**Option B: Record your own samples**
```bash
python collect_data.py --class attentive  --n 150
python collect_data.py --class distracted --n 150
python collect_data.py --class disengaged --n 150
```
Press **SPACE** to capture a frame, **Q** to quit.

**Option C: Use DAiSEE dataset (manual download)**
- Download from: https://www.kaggle.com/datasets/joyee19/studentengagement
- Extract to `data/` folder (should create `data/Engaged/` and `data/Not engaged/`)
- The system auto-maps DAiSEE categories to your 3 classes:
  - **attentive**: `Engaged/engaged/`
  - **distracted**: `Engaged/confused/`, `Engaged/frustrated/`, `Not engaged/bored/`
  - **disengaged**: `Not engaged/Looking away/`, `Not engaged/drowsy/`

**Option D: Mix both** - place custom recordings AND DAiSEE data in the same `data/` folder.

### Step 2 — Train the model
```bash
python train.py
```
- Trains for 20 epochs by default
- Freezes backbone first, unfreezes at epoch 6 for fine-tuning
- Best weights saved to `weights/model.pth`

### Step 3 — Run the dashboard
```bash
streamlit run app.py
```
Open http://localhost:8501 → click **▶ Start** → point camera at your class.

---

## How It Works

```
Webcam Frame
    │
    ▼
OpenCV Face Detection  →  Face bounding boxes
    │
    ├── Eye Aspect Ratio (EAR)  →  detects drowsiness
    ├── Mouth Aspect Ratio (MAR) →  detects yawning
    ├── Head Pose (PnP Solver)   →  detects looking away
    │
    ▼
Face ROI cropped
    │
    ▼
MobileNetV2  →  [attentive, distracted, disengaged] softmax
    │
    ▼
Composite Score (0–100)  →  Streamlit Gauges + Chart
```

---

## Dataset Options

| Option | Details |
|--------|---------|
| **Record yourself** | Use `collect_data.py` — 150 samples/class is enough to start |
| **DAiSEE** | ~10k labelled frames from Kaggle — best results |
| **EngageActivity** | Alternative Kaggle dataset |

---

## Config (`config.yaml`)

```yaml
training:
  epochs: 20       # increase to 30 for better accuracy
  batch_size: 32
  lr: 0.0003

thresholds:
  ear: 0.20        # below = eyes closed / drowsy
  yaw: 25          # head turned more than 25° = distracted
  alert: 50        # fire warning if attentive% drops below 50%
```

---

## Results (DAiSEE dataset)

| Metric | Value |
|--------|-------|
| Validation Accuracy | ~88% |
| Macro F1-Score | ~0.86 |
| Inference Speed | ~25 ms/frame (CPU) |

---

## Tech Stack
- **PyTorch + timm** — MobileNetV2 pretrained on ImageNet
- **MediaPipe** — real-time face mesh
- **OpenCV** — webcam capture and annotation
- **Streamlit + Plotly** — live dashboard

🎓 Classroom Engagement Tracker — Simple Explanation
🟢 1. What is this project?

This is a real-time AI system that checks if students are:

✅ Attentive (focused)
⚠️ Distracted (looking away, confused)
❌ Disengaged (sleepy, not interested)

👉 It uses a camera (webcam) and gives live results on screen

🟢 2. Why we made this?

“In online or large classrooms, it’s hard for teachers to know who is paying attention.
So we built a system that automatically tracks student engagement using AI.”

🟢 3. Technologies Used (keep this short)
OpenCV → to capture video from camera
MediaPipe → to detect face & facial landmarks
MobileNetV2 (Deep Learning model) → to classify attention
Streamlit → to show dashboard
PyTorch → to train model

👉 One line:
“OpenCV + MediaPipe extract features, and MobileNetV2 classifies engagement.”

🟢 4. How system works (VERY IMPORTANT — explain clearly)

Say this step by step 👇

🔄 Flow:
Camera captures video
Face is detected using OpenCV
Important features are extracted:
👁 Eye (EAR → eyes closed = sleepy)
👄 Mouth (MAR → yawning)
🧠 Head pose (looking away)
Face image is sent to AI model (MobileNetV2)
Model predicts:
Attentive / Distracted / Disengaged
Output is shown on dashboard:
Label (text)
Score (0–100)
Graph (live tracking)
🟢 5. Input → Process → Output (MOST IMPORTANT PART)
✅ Input:
Live webcam video OR recorded video
⚙️ Process:
Face detection
Feature extraction (eyes, mouth, head)
AI model prediction
📊 Output:
Student state:
👉 Attentive / Distracted / Disengaged
Engagement score (like 75%)
Live dashboard with charts
🟢 6. Training the Model (simple explanation)

“We trained our model using dataset like DAiSEE and custom images.”

Steps:

Collect images (3 classes)
Train MobileNetV2
Save best model (model.pth)
Use it in real-time app
🟢 7. Accuracy / Performance
Accuracy ≈ 88%
Works in real-time (~25ms per frame)

👉 Say:
“Our model gives good accuracy and fast real-time performance.”

🟢 8. Demo Explanation (what to say while showing)

When you run:

streamlit run app.py

Say this 👇

“This is our live dashboard”
“When I start camera, system detects my face”
“Now you can see it classifies my state”
“If I look away → distracted”
“If I close eyes → disengaged”
“Graph updates in real time”
🟢 9. Key Features (quick points)
Real-time detection
Works on webcam
AI-based classification
Live dashboard
Can generate reports (CSV)
🟢 10. One-Line Conclusion (IMPORTANT)

👉 Say this at end:

“This project helps teachers automatically monitor student attention using AI, making classrooms more interactive and effective.”

🔥 BONUS: 1-Minute Full Script (MEMORIZE THIS)

“Good morning, today I am presenting our project Classroom Engagement Tracker.
This system uses AI to detect whether a student is attentive, distracted, or disengaged in real time.

We use OpenCV and MediaPipe to detect the face and extract features like eye movement, mouth movement, and head pose.
These features are passed into a MobileNetV2 deep learning model which classifies the student’s engagement level.

The system takes webcam input, processes each frame, and outputs the engagement state along with a score on a live Streamlit dashboard.

Our model is trained on datasets like DAiSEE and achieves around 88% accuracy with real-time performance.

This project can help teachers monitor student attention automatically, especially in online or large classrooms.”