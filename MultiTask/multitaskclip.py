
"""
dataset_multitask.py — Multitask dataset for category + color + gender
Imported by train_multitask.py — do not run directly.
"""

import os
import json
import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR    = os.environ.get("DATA_DIR",    "/workspace/Data/train")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR",  "/workspace/outputs")
BATCH_SIZE  = int(os.environ.get("BATCH_SIZE",  "64"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS",  "4"))
IMG_SIZE    = int(os.environ.get("IMG_SIZE",     "224"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

PARQUET_PATH  = os.path.join(DATA_DIR, "train.parquet")
IMAGE_DIR     = os.path.join(DATA_DIR, "images")
TAXONOMY_PATH = os.path.join(DATA_DIR, "level2_categories.json")

for path in [PARQUET_PATH, IMAGE_DIR, TAXONOMY_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    print(f"OK {path}")

# ── Load Data ─────────────────────────────────────────────────────────────────
print("\n--- Loading data ---")
df = pd.read_parquet(PARQUET_PATH)

discovery_path = "/workspace/Data/discovery/discovery.parquet"
if os.path.exists(discovery_path):
    disc_df = pd.read_parquet(discovery_path)
    df = pd.concat([df, disc_df]).reset_index(drop=True)
    print(f"After discovery merge: {df.shape}")

with open(TAXONOMY_PATH, "r") as f:
    mapping = json.load(f)

taxonomy = pd.DataFrame(mapping).rename(columns={"new_id": "category_id"})
df = df.merge(taxonomy[["category_id", "category_name"]], on="category_id", how="left")
df[["category", "subcategory"]] = df["category_name"].str.split(" > ", expand=True)
df.loc[df["category_id"] == 44, "category_id"] = 28

# Fill missing values
df["gender"]     = df["gender"].fillna("UNISEX")
df["color_label"] = df["color_label"].fillna("unknown")

# Drop rows with unknown color for training
df_with_color = df[df["color_label"] != "unknown"].copy()
print(f"Samples with color: {len(df_with_color)} | without: {len(df) - len(df_with_color)}")

print(f"Shape: {df.shape} | Subcategories: {df['category_id'].nunique()} | Main: {df['category'].nunique()}")

# ── Label Encoding ────────────────────────────────────────────────────────────
main_encoder = LabelEncoder()
df["main_encoded"] = main_encoder.fit_transform(df["category"])

cat_encoder = LabelEncoder()
df["category_encoded"] = cat_encoder.fit_transform(df["category_id"])

color_encoder = LabelEncoder()
df_with_color["color_encoded"] = color_encoder.fit_transform(df_with_color["color_label"])
# Apply to full df
df["color_encoded"] = -1  # -1 for unknown
df.loc[df["color_label"] != "unknown", "color_encoded"] = color_encoder.transform(
    df.loc[df["color_label"] != "unknown", "color_label"]
)

gender_encoder = LabelEncoder()
df["gender_encoded"] = gender_encoder.fit_transform(df["gender"])

NUM_MAIN    = int(df["main_encoded"].nunique())
NUM_CLASSES = int(df["category_encoded"].nunique())
NUM_COLORS  = int(df_with_color["color_label"].nunique())
NUM_GENDERS = int(df["gender_encoded"].nunique())

print(f"Main: {NUM_MAIN} | Sub: {NUM_CLASSES} | Colors: {NUM_COLORS} | Genders: {NUM_GENDERS}")
print(f"Color classes: {list(color_encoder.classes_)}")
print(f"Gender classes: {list(gender_encoder.classes_)}")

# Save encoders
for name, enc in [("category_encoder", cat_encoder), ("main_encoder", main_encoder),
                  ("color_encoder", color_encoder), ("gender_encoder", gender_encoder)]:
    with open(os.path.join(OUTPUT_DIR, f"{name}.pkl"), "wb") as f:
        pickle.dump(enc, f)

# ── Train / Test Split ────────────────────────────────────────────────────────
train_df, test_df = train_test_split(
    df, test_size=0.1,
    stratify=df["category_id"],
    random_state=42
)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)
print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# ── Transforms — NO ColorJitter (hurts color prediction) ─────────────────────
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.85, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Dataset ───────────────────────────────────────────────────────────────────
class MultitaskDataset(Dataset):
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
            int(row["main_encoded"]),
            int(row["category_encoded"]),
            int(row["color_encoded"]),    # -1 if unknown
            int(row["gender_encoded"]),
        )

# ── DataLoaders ───────────────────────────────────────────────────────────────
class_counts   = train_df["category_encoded"].value_counts().sort_index()
weights        = 1.0 / class_counts
sample_weights = train_df["category_encoded"].map(weights).values.copy()
sampler        = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(MultitaskDataset(train_df, IMAGE_DIR, train_transforms),
                          batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(MultitaskDataset(test_df, IMAGE_DIR, test_transforms),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

images, main_labels, sub_labels, color_labels, gender_labels = next(iter(train_loader))
print(f"\nBatch shape:    {images.shape}")
print(f"Main range:     {main_labels.min()}-{main_labels.max()}")
print(f"Sub range:      {sub_labels.min()}-{sub_labels.max()}")
print(f"Color range:    {color_labels.min()}-{color_labels.max()} (-1 = unknown)")
print(f"Gender range:   {gender_labels.min()}-{gender_labels.max()}")
print("dataset_multitask.py loaded successfully.")