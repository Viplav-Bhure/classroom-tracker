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
