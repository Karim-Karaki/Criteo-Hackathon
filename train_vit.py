"""
Criteo Image Classification — Training Script (Category Only)
=============================================================
Trains ViT-B/16 on category prediction only (191 classes).
Color and gender are predicted separately via rule-based and correlation methods.

Vast.ai usage:
    python train.py --data_dir /workspace/data --output_dir /workspace/outputs

Local usage:
    python train.py --data_dir "C:/Users/karim/..." --output_dir ./outputs
"""

import os
import json
import pickle
import argparse
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image

# ── Argument Parsing ─────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",      type=str, default="/workspace/data",    help="Folder with train.parquet, images/, level2_categories.json")
parser.add_argument("--output_dir",    type=str, default="/workspace/outputs", help="Where to save checkpoints and encoders")
parser.add_argument("--batch_size",    type=int, default=32)
parser.add_argument("--phase1_epochs", type=int, default=5)
parser.add_argument("--phase2_epochs", type=int, default=10)
parser.add_argument("--num_workers",   type=int, default=4,  help="Set to 0 on Windows")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────────────────────

PARQUET_PATH  = os.path.join(args.data_dir, "train.parquet")
IMAGE_DIR     = os.path.join(args.data_dir, "images")
TAXONOMY_PATH = os.path.join(args.data_dir, "level2_categories.json")

print(f"Data dir:   {args.data_dir}")
print(f"Output dir: {args.output_dir}")
print(f"Image dir:  {IMAGE_DIR}")

# ── Load & Clean Data ─────────────────────────────────────────────────────────

print("\n--- Loading data ---")
df = pd.read_parquet(PARQUET_PATH)
print(f"Shape: {df.shape}")
print(f"Nulls:\n{df.isnull().sum()}")

# Load taxonomy mapping
with open(TAXONOMY_PATH, "r") as f:
    mapping = json.load(f)

taxonomy = pd.DataFrame(mapping).rename(columns={"new_id": "category_id"})
df = df.merge(taxonomy[["category_id", "category_name"]], on="category_id", how="left")
df[["category", "subcategory"]] = df["category_name"].str.split(" > ", expand=True)
print(f"Missing category names: {df['category_name'].isna().sum()}")

# ── Label Encoding (Category Only) ───────────────────────────────────────────

print("\n--- Encoding labels ---")

cat_encoder = LabelEncoder()
df["category_encoded"] = cat_encoder.fit_transform(df["category_id"])

print(f"Category classes: {df['category_encoded'].nunique()} (encoded 0-{df['category_encoded'].max()})")

cat_encoder_path = os.path.join(args.output_dir, "category_encoder.pkl")
with open(cat_encoder_path, "wb") as f:
    pickle.dump(cat_encoder, f)
print(f"Saved {cat_encoder_path}")

# ── Train / Test Split ────────────────────────────────────────────────────────

print("\n--- Splitting data ---")

class_44      = df[df["category_id"] == 44].reset_index(drop=True)
df_without_44 = df[df["category_id"] != 44]

train_df, test_df = train_test_split(
    df_without_44, test_size=0.2,
    stratify=df_without_44["category_id"],
    random_state=42
)

train_df = pd.concat([train_df, class_44.iloc[[0]]]).reset_index(drop=True)
test_df  = pd.concat([test_df,  class_44.iloc[[1]]]).reset_index(drop=True)

print(f"Train: {len(train_df)} samples | {train_df['category_id'].nunique()} classes")
print(f"Test:  {len(test_df)}  samples | {test_df['category_id'].nunique()} classes")

# ── Transforms ────────────────────────────────────────────────────────────────

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Dataset & DataLoaders ─────────────────────────────────────────────────────

class ProductDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
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

        # Category only — no color or gender
        category = int(row["category_encoded"])
        return image, category


train_dataset = ProductDataset(train_df, IMAGE_DIR, transform=train_transforms)
test_dataset  = ProductDataset(test_df,  IMAGE_DIR, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

# Sanity check
images, categories = next(iter(train_loader))
print(f"\nBatch shape:    {images.shape}")
print(f"Category range: {categories.min()}–{categories.max()}")

# ── Model ─────────────────────────────────────────────────────────────────────

NUM_CLASSES = int(df["category_encoded"].nunique())
print(f"\n--- Building ViT-B/16 for {NUM_CLASSES} classes ---")

vit       = models.vit_b_16(weights="IMAGENET1K_V1")
vit.heads = nn.Identity()

class ViTCategory(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone      = backbone
        self.dropout       = nn.Dropout(0.1)
        self.category_head = nn.Linear(768, NUM_CLASSES)

    def forward(self, x):
        features = self.dropout(self.backbone(x))
        return self.category_head(features)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

model = ViTCategory(vit).to(device)

# ── Training ──────────────────────────────────────────────────────────────────

criterion = nn.CrossEntropyLoss()

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, correct = 0, 0

    for images, categories in loader:
        images     = images.to(device)
        categories = categories.to(device)

        optimizer.zero_grad()
        cat_out = model(images)
        loss    = criterion(cat_out, categories)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct    += (cat_out.argmax(1) == categories).sum().item()

    return total_loss / len(loader), correct / len(loader.dataset)

# Phase 1 — freeze backbone, train head only
print("\n--- Phase 1: Head only ---")
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = Adam(model.category_head.parameters(), lr=1e-3)

for epoch in range(args.phase1_epochs):
    loss, acc = train_epoch(model, train_loader, optimizer)
    print(f"Epoch {epoch+1}/{args.phase1_epochs} | Loss: {loss:.4f} | Acc: {acc:.4f}")

# Phase 2 — unfreeze full network
print("\n--- Phase 2: Full fine-tuning ---")
for param in model.backbone.parameters():
    param.requires_grad = True

optimizer = Adam([
    {"params": model.backbone.parameters(),      "lr": 1e-5, "weight_decay": 1e-4},
    {"params": model.category_head.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
])

best_acc  = 0
ckpt_path = os.path.join(args.output_dir, "vit_category_best.pth")

for epoch in range(args.phase2_epochs):
    loss, acc = train_epoch(model, train_loader, optimizer)
    print(f"Epoch {epoch+1}/{args.phase2_epochs} | Loss: {loss:.4f} | Acc: {acc:.4f}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), ckpt_path)
        print(f"  -> Saved best model ({best_acc:.4f}) to {ckpt_path}")

# ── Evaluation ────────────────────────────────────────────────────────────────

print("\n--- Evaluating on test set ---")
model.load_state_dict(torch.load(ckpt_path))
model.eval()

correct = 0
total   = len(test_loader.dataset)

with torch.no_grad():
    for images, categories in test_loader:
        images     = images.to(device)
        categories = categories.to(device)
        cat_out    = model(images)
        correct   += (cat_out.argmax(1) == categories).sum().item()

print(f"Test Category Accuracy: {correct/total:.4f}")
print(f"Best Train Accuracy:    {best_acc:.4f}")
print(f"Overfitting gap:        {best_acc - correct/total:.4f}")
print("\nDone.")
