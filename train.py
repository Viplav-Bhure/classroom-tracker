"""
train.py
Train the MobileNetV2 engagement classifier.

Usage:
    python train.py
"""

import os
import time

import torch
import torch.nn as nn
import yaml
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import EngagementDataset, EngagementModel

# ── Load config ───────────────────────────────────────────────────────────────
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

EPOCHS     = cfg["training"]["epochs"]
BATCH_SIZE = cfg["training"]["batch_size"]
LR         = cfg["training"]["lr"]
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    print(f"Training on: {DEVICE}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = EngagementDataset(split="train")
    val_ds   = EngagementDataset(split="val")
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model     = EngagementModel(num_classes=3).to(DEVICE)
    
    # Calculate class weights for imbalanced data
    labels_list = [s[1] for s in train_ds.samples]
    counts = torch.tensor([labels_list.count(i) for i in range(3)], dtype=torch.float)
    weights = 1.0 / (counts + 1e-6)
    weights = (weights / weights.sum() * 3).to(DEVICE)
    print(f"Class weights: {weights.cpu().numpy()}")

    # Use label smoothing to handle noisy labels
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    os.makedirs("weights", exist_ok=True)
    best_acc = 0.0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):

        # Unfreeze backbone at epoch 6 for end-to-end fine-tuning
        if epoch == 6:
            model.unfreeze()
            optimizer = torch.optim.AdamW([
                {"params": model.backbone.parameters(), "lr": LR / 10},
                {"params": model.head.parameters(),     "lr": LR},
            ])
            # Re-initialize scheduler for the new optimizer
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS - 5)
            print(">> Backbone unfrozen — fine-tuning end-to-end")

        # Train
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        for imgs, labels in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS} [train]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total   += len(labels)
        scheduler.step()

        # Validate
        model.eval()
        val_correct, val_total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs).argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total   += len(labels)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total
        print(f"  Train acc: {correct/total:.3f}  |  Val acc: {val_acc:.3f}")

        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "weights/model.pth")
            print(f"  ✓ Saved best model (val acc = {best_acc:.3f})")
            best_report = classification_report(all_labels, all_preds, 
                                             target_names=["attentive", "distracted", "disengaged"],
                                             zero_division=0)

    # ── Final report ─────────────────────────────────────────────────────────
    print("\nClassification Report (Best Model):")
    print(best_report)
    print(f"\nDone! Best val accuracy: {best_acc:.3f}")
    print("Weights saved to: weights/model.pth")


if __name__ == "__main__":
    main()
