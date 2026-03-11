"""
dataset.py — Hierarchical Data loading, augmentation, and splitting
Imported by train_model.py — do not run directly.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image

# ── Config from environment variables ────────────────────────────────────────
DATA_DIR    = os.environ.get("DATA_DIR",    "/workspace/Data/Data/")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR",  "/workspace/outputs")
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE",  "64"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS",  "4"))
IMG_SIZE    = int(os.environ.get("IMG_SIZE",     "224"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
PARQUET_PATH  = os.path.join(DATA_DIR, "train.parquet")
IMAGE_DIR     = os.path.join(DATA_DIR, "images")
TAXONOMY_PATH = os.path.join(DATA_DIR, "level2_categories.json")

for path in [PARQUET_PATH, IMAGE_DIR, TAXONOMY_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    print(f"OK {path}")

# ── Load & Clean Data ─────────────────────────────────────────────────────────
print("\n--- Loading data ---")
df = pd.read_parquet(PARQUET_PATH)
discovery_df = pd.read_parquet(os.path.join(DATA_DIR.replace("train", ""), "discovery", "discovery.parquet"))
df = pd.concat([df, discovery_df]).reset_index(drop=True)
print(f"Combined shape: {df.shape}")



with open(TAXONOMY_PATH, "r") as f:
    mapping = json.load(f)

taxonomy = pd.DataFrame(mapping).rename(columns={"new_id": "category_id"})
df = df.merge(taxonomy[["category_id", "category_name"]], on="category_id", how="left")
df[["category", "subcategory"]] = df["category_name"].str.split(" > ", expand=True)

# Merge class 44 into class 28
df.loc[df["category_id"] == 44, "category_id"] = 28
print(f"Shape: {df.shape} | Subcategories: {df['category_id'].nunique()} | Main categories: {df['category'].nunique()}")

# ── Label Encoding ────────────────────────────────────────────────────────────

# Main category encoder (21 classes)
main_encoder = LabelEncoder()
df["main_encoded"] = main_encoder.fit_transform(df["category"])

# Subcategory encoder (190 classes)
cat_encoder = LabelEncoder()
df["category_encoded"] = cat_encoder.fit_transform(df["category_id"])

NUM_MAIN    = int(df["main_encoded"].nunique())
NUM_CLASSES = int(df["category_encoded"].nunique())
print(f"Main categories: {NUM_MAIN} | Subcategories: {NUM_CLASSES}")

# Save encoders
with open(os.path.join(OUTPUT_DIR, "category_encoder.pkl"), "wb") as f:
    pickle.dump(cat_encoder, f)
with open(os.path.join(OUTPUT_DIR, "main_encoder.pkl"), "wb") as f:
    pickle.dump(main_encoder, f)

# ── Train / Test Split ────────────────────────────────────────────────────────
train_df, test_df = train_test_split(
    df, test_size=0.1,
    stratify=df["category_id"],
    random_state=42
)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# ── Transforms ────────────────────────────────────────────────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Dataset Class ─────────────────────────────────────────────────────────────
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
        return (
            image,
            int(row["main_encoded"]),      # main category label (0-20)
            int(row["category_encoded"])   # subcategory label (0-189)
        )

# ── DataLoaders ───────────────────────────────────────────────────────────────
def get_loaders(train_df, test_df, image_dir, batch_size, num_workers):
    train_dataset  = ProductDataset(train_df, image_dir, transform=train_transforms)
    test_dataset   = ProductDataset(test_df,  image_dir, transform=test_transforms)

    # Weight by subcategory to balance rare classes
    class_counts   = train_df["category_encoded"].value_counts().sort_index()
    weights        = 1.0 / class_counts
    sample_weights = train_df["category_encoded"].map(weights).values.copy()
    sampler        = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,   num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

train_loader, test_loader = get_loaders(train_df, test_df, IMAGE_DIR, BATCH_SIZE, NUM_WORKERS)

images, main_labels, sub_labels = next(iter(train_loader))
print(f"\nBatch shape:       {images.shape}")
print(f"Main label range:  {main_labels.min()}-{main_labels.max()} ({NUM_MAIN} classes)")
print(f"Sub label range:   {sub_labels.min()}-{sub_labels.max()} ({NUM_CLASSES} classes)")
print("dataset.py loaded successfully.")
