"""
model.py
MobileNetV2 classifier, dataset loader, and inference — all in one place.
"""

import os
import random
from pathlib import Path

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = ["attentive", "distracted", "disengaged"]

# Image transforms
TRAIN_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.3, 0.3, 0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────

class EngagementDataset(Dataset):
    """
    Supports both custom format and DAiSEE dataset:
    - Custom: data/{attentive, distracted, disengaged}/*.jpg
    - DAiSEE: data/{Engaged, Not engaged}/{subfolders}/*.jpg
    - Auto-downloads DAiSEE if no data found
    """
    def __init__(self, root="data", split="train", train_ratio=0.8, seed=42):
        self.transform = TRAIN_TF if split == "train" else VAL_TF
        samples = []

        # Check if any data exists
        root_path = Path(root)
        has_any_data = False
        
        # Check for custom format
        custom_folders = [root_path / cls for cls in CLASSES]
        has_custom = any(folder.exists() and any(folder.rglob("*.jpg")) for folder in custom_folders)
        if has_custom:
            has_any_data = True

        # Check for DAiSEE format
        engaged_folder = root_path / "Engaged"
        not_engaged_folder = root_path / "Not engaged"
        has_daisee = (engaged_folder.exists() and not_engaged_folder.exists() and
                     any(engaged_folder.rglob("*.jpg")) and any(not_engaged_folder.rglob("*.jpg")))
        if has_daisee:
            has_any_data = True

        # Auto-download DAiSEE if no data found
        if not has_any_data:
            print("No dataset found. Downloading DAiSEE dataset from Kaggle...")
            try:
                import kagglehub
                dataset_path = kagglehub.dataset_download("joyee19/studentengagement")
                print(f"Downloaded to: {dataset_path}")
                
                # Copy DAiSEE folders to our data directory
                import shutil
                dataset_src = Path(dataset_path) / "Student-engagement-dataset"
                
                if dataset_src.exists():
                    root_path.mkdir(exist_ok=True)
                    for folder in ["Engaged", "Not engaged"]:
                        src_folder = dataset_src / folder
                        dst_folder = root_path / folder
                        if src_folder.exists():
                            if dst_folder.exists():
                                shutil.rmtree(dst_folder)
                            shutil.copytree(src_folder, dst_folder)
                            print(f"Copied {folder} to {dst_folder}")
                    
                    has_daisee = True
                    print("DAiSEE dataset ready!")
                else:
                    print(f"Warning: Expected folder not found in downloaded dataset: {dataset_src}")
                    
            except Exception as e:
                print(f"Failed to download DAiSEE dataset: {e}")
                print("Please download manually from https://www.kaggle.com/datasets/joyee19/studentengagement")
                raise FileNotFoundError("No dataset available")

        # Now load the data
        if has_custom:
            # Use custom format
            print("Using custom dataset format")
            for label, cls in enumerate(CLASSES):
                folder = root_path / cls
                if not folder.exists():
                    continue
                for ext in ("*.jpg", "*.jpeg", "*.png"):
                    for img in folder.rglob(ext):
                        samples.append((img, label))
        elif has_daisee:
            # Use DAiSEE format
            print("Using DAiSEE dataset format")
            # DAiSEE mapping to our 3 classes
            daisee_mapping = {
                # attentive (0)
                (engaged_folder / "engaged"): 0,
                # distracted (1)
                (engaged_folder / "confused"): 1,
                (engaged_folder / "frustrated"): 1,
                (not_engaged_folder / "bored"): 1,
                # disengaged (2)
                (not_engaged_folder / "Looking away"): 2,
                (not_engaged_folder / "drowsy"): 2,
            }

            for folder_path, label in daisee_mapping.items():
                if folder_path.exists():
                    for ext in ("*.jpg", "*.jpeg", "*.png"):
                        for img in folder_path.rglob(ext):
                            samples.append((img, label))
        else:
            # Fallback
            for label, cls in enumerate(CLASSES):
                folder = root_path / cls
                if not folder.exists():
                    continue
                for ext in ("*.jpg", "*.jpeg", "*.png"):
                    for img in folder.rglob(ext):
                        samples.append((img, label))

        random.seed(seed)
        random.shuffle(samples)
        cut = int(len(samples) * train_ratio)
        self.samples = samples[:cut] if split == "train" else samples[cut:]

        if len(self.samples) == 0:
            raise FileNotFoundError(
                "No images found in data/ folder.\n"
                "Options:\n"
                "1. Record samples: python collect_data.py --class attentive --n 150\n"
                "2. Download DAiSEE dataset and place in data/ folder\n"
                "3. Use custom folders: data/{attentive,distracted,disengaged}/"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ── Model ─────────────────────────────────────────────────────────────────────

class EngagementModel(nn.Module):
    """MobileNetV2 fine-tuned for 3-class engagement classification."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = timm.create_model("mobilenetv2_100", pretrained=True,
                                          num_classes=0, global_pool="avg")
        in_feats = self.backbone.num_features  # 1280

        # Freeze backbone initially
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_feats, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.head(self.backbone(x))


# ── Predictor (used by app.py) ────────────────────────────────────────────────

class Predictor:
    """Loads the trained model and predicts engagement from a face ROI (numpy BGR)."""

    def __init__(self, weights_path="weights/model.pth", device="cpu"):
        self.device = torch.device(device)
        self.model  = EngagementModel().to(self.device)
        self.model.eval()

        if os.path.exists(weights_path):
            ckpt = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(ckpt)
            print(f"Loaded weights: {weights_path}")
        else:
            print(f"[WARNING] No weights at '{weights_path}' — train first: python train.py")

    @torch.no_grad()
    def predict(self, roi_bgr: np.ndarray):
        """
        Returns (label, confidence, score_0_to_100)
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return "attentive", 0.33, 50.0

        img   = Image.fromarray(cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB))
        x     = VAL_TF(img).unsqueeze(0).to(self.device)
        probs = F.softmax(self.model(x), dim=-1).squeeze().cpu().numpy()

        idx   = int(np.argmax(probs))
        label = CLASSES[idx]
        conf  = float(probs[idx])

        # Engagement score: attentive = high, disengaged = low
        score = float(probs[0] * 100 - probs[1] * 30 - probs[2] * 60)
        score = float(np.clip(score, 0, 100))

        return label, round(conf, 3), round(score, 1)
