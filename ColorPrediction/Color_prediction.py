"""
train_color.py — Lightweight Color Classifier
=============================================
Uses ResNet18 — a small, fast CNN pretrained on ImageNet.
Only trains the final classification head on your 12 color classes.
Should train in under 10 minutes on the 5090.

Why ResNet18:
- Only 11M parameters vs 87M for ViT-B/16 or 30M for EfficientNet-B3
- Color is a global/simple feature — you dont need deep complex features
- Fast to train, hard to overfit on 12 classes
- Already understands basic visual concepts from ImageNet pretraining

Usage:
    python train_color.py --data_dir /workspace/Data/train --output_dir /workspace/outputs
"""

import os
import json
import pickle
import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from PIL import Image
import pandas as pd
import numpy as np

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",      type=str,   default="/workspace/Data/train")
parser.add_argument("--output_dir",    type=str,   default="/workspace/outputs")
parser.add_argument("--batch_size",    type=int,   default=64)
parser.add_argument("--num_workers",   type=int,   default=4)
parser.add_argument("--phase1_epochs", type=int,   default=5)
parser.add_argument("--phase2_epochs", type=int,   default=15)
parser.add_argument("--patience",      type=int,   default=4)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Load Data ─────────────────────────────────────────────────────────────────
print("\n--- Loading data ---")
PARQUET_PATH  = os.path.join(args.data_dir, "train.parquet")
TAXONOMY_PATH = os.path.join(args.data_dir, "level2_categories.json")
IMAGE_DIR     = os.path.join(args.data_dir, "images")

df = pd.read_parquet(PARQUET_PATH)

discovery_path = "/workspace/Data/discovery/discovery.parquet"
if os.path.exists(discovery_path):
    disc_df = pd.read_parquet(discovery_path)
    df = pd.concat([df, disc_df]).reset_index(drop=True)

# Drop rows with missing color_label
df = df.dropna(subset=["color_label"]).reset_index(drop=True)
df = df[df["color_label"] != "unknown"].reset_index(drop=True)

print(f"Shape: {df.shape}")
print(f"\nColor distribution:")
print(df["color_label"].value_counts())

# ── Encode Colors ─────────────────────────────────────────────────────────────
color_encoder = LabelEncoder()
df["color_encoded"] = color_encoder.fit_transform(df["color_label"])
NUM_COLORS = int(df["color_encoded"].nunique())
print(f"\nColor classes: {NUM_COLORS}")
print(f"Classes: {list(color_encoder.classes_)}")

# Save encoder
with open(os.path.join(args.output_dir, "color_encoder.pkl"), "wb") as f:
    pickle.dump(color_encoder, f)

# ── Split ─────────────────────────────────────────────────────────────────────
train_df, test_df = train_test_split(
    df, test_size=0.1,
    stratify=df["color_encoded"],
    random_state=42
)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)
print(f"\nTrain: {len(train_df)} | Test: {len(test_df)}")

# ── Transforms ────────────────────────────────────────────────────────────────
# Note: ColorJitter is intentionally mild here — we dont want to distort the
# actual color of the product which is what we're trying to classify
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Dataset ───────────────────────────────────────────────────────────────────
class ColorDataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, str(row["category_id"]), f"{row['anonymous_id']}.jpg")
        image    = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["color_encoded"])

# WeightedRandomSampler for color imbalance
color_counts   = train_df["color_encoded"].value_counts().sort_index()
weights        = 1.0 / color_counts
sample_weights = train_df["color_encoded"].map(weights).values.copy()
sampler        = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_dataset = ColorDataset(train_df, IMAGE_DIR, train_transforms)
test_dataset  = ColorDataset(test_df,  IMAGE_DIR, test_transforms)
train_loader  = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,   num_workers=args.num_workers, pin_memory=True)
test_loader   = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False,     num_workers=args.num_workers, pin_memory=True)

# ── Model — ResNet18 ──────────────────────────────────────────────────────────
# ResNet18 architecture:
# - Input: 224x224 image
# - Conv layer → 4 residual blocks (each block = 2 conv layers with skip connection)
# - Global average pooling → 512-dimensional feature vector
# - Linear head → your N classes
#
# Skip connections (the "residual" part) let gradients flow directly through
# the network, solving the vanishing gradient problem and making it much easier
# to train than plain CNNs. This is why ResNet was revolutionary in 2015.
#
# ResNet18 = 18 layers total = 4 blocks x 2 layers + some extra = 11M parameters
# Compare: ResNet50 = 25M params, EfficientNet-B3 = 12M, ViT-B/16 = 87M

print("\n--- Building ResNet18 color classifier ---")
backbone = models.resnet18(weights="IMAGENET1K_V1")
feature_dim = backbone.fc.in_features  # 512
backbone.fc = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(feature_dim, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_COLORS)
)
model     = backbone.to(device)
ckpt_path = os.path.join(args.output_dir, "color_resnet18_best.pth")

print(f"Feature dim: {feature_dim} | Color classes: {NUM_COLORS}")

# ── Class Weighted Loss ───────────────────────────────────────────────────────
color_weights = 1.0 / torch.tensor(color_counts.values, dtype=torch.float)
color_weights = (color_weights / color_weights.sum() * NUM_COLORS).to(device)
criterion     = nn.CrossEntropyLoss(weight=color_weights, label_smoothing=0.1)

# ── Early Stopping ────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=4, min_delta=0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_acc  = 0

    def step(self, acc):
        if acc > self.best_acc + self.min_delta:
            self.best_acc = acc
            self.counter  = 0
            return False
        else:
            self.counter += 1
            print(f"  Early stopping counter: {self.counter}/{self.patience}")
            return self.counter >= self.patience

# ── Training Function ─────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += labels.size(0)
    return total_loss / len(loader), correct / total

# ── Phase 1 — Head only ───────────────────────────────────────────────────────
print("\n--- Phase 1: Head only ---")
for name, param in model.named_parameters():
    if "fc" not in name:
        param.requires_grad = False

optimizer = Adam(model.fc.parameters(), lr=1e-3)
for epoch in range(args.phase1_epochs):
    loss, acc = train_epoch(model, train_loader, optimizer)
    print(f"Epoch {epoch+1}/{args.phase1_epochs} | Loss: {loss:.4f} | Acc: {acc:.4f}")

# ── Phase 2 — Full fine-tuning ────────────────────────────────────────────────
print("\n--- Phase 2: Full fine-tuning ---")
for param in model.parameters():
    param.requires_grad = True

optimizer = Adam([
    {"params": [p for n, p in model.named_parameters() if "fc" not in n], "lr": 1e-5, "weight_decay": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
])

scheduler      = CosineAnnealingLR(optimizer, T_max=args.phase2_epochs)
best_acc       = 0
early_stopping = EarlyStopping(patience=args.patience)

for epoch in range(args.phase2_epochs):
    loss, acc = train_epoch(model, train_loader, optimizer)
    scheduler.step()
    print(f"Epoch {epoch+1}/{args.phase2_epochs} | Loss: {loss:.4f} | Acc: {acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), ckpt_path)
        print(f"  -> Saved best model (Acc: {best_acc:.4f})")

    if early_stopping.step(acc):
        print("Early stopping triggered.")
        break

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\n--- Evaluating on test set ---")
model.load_state_dict(torch.load(ckpt_path))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        preds  = model(images).argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print(f"\n--- Color Classification (12 classes) ---")
print(f"Test Accuracy: {sum(p==l for p,l in zip(all_preds,all_labels))/len(all_labels):.4f}")
print(f"Train Accuracy: {best_acc:.4f}")
print(f"F1 Macro:      {f1_score(all_labels, all_preds, average='macro'):.4f}")
print(f"F1 Weighted:   {f1_score(all_labels, all_preds, average='weighted'):.4f}")
print(f"\nPer-class report:")
print(classification_report(all_labels, all_preds, target_names=color_encoder.classes_))
print("\nDone.")