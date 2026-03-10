"""
dataset.py — Data loading, augmentation, and splitting
Run standalone to verify everything works:
    python dataset.py --data_dir /workspace/data/train
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image

# ── Argument Parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",   type=str, default="/workspace/data/train")
parser.add_argument("--output_dir", type=str, default="/workspace/outputs")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers",type=int, default=4)
parser.add_argument("--img_size",   type=int, default=224)
args, _ = parser.parse_known_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── Paths ─────────────────────────────────────────────────────────────────────

PARQUET_PATH  = os.path.join(args.data_dir, "train.parquet")
IMAGE_DIR     = os.path.join(args.data_dir, "images")
TAXONOMY_PATH = os.path.join(args.data_dir, "level2_categories.json")

for path in [PARQUET_PATH, IMAGE_DIR, TAXONOMY_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing: {path}")
    print(f"OK {path}")

# ── Load & Clean Data ─────────────────────────────────────────────────────────

print("\n--- Loading data ---")
df = pd.read_parquet(PARQUET_PATH)

with open(TAXONOMY_PATH, "r") as f:
    mapping = json.load(f)

taxonomy = pd.DataFrame(mapping).rename(columns={"new_id": "category_id"})
df = df.merge(taxonomy[["category_id", "category_name"]], on="category_id", how="left")
df[["category", "subcategory"]] = df["category_name"].str.split(" > ", expand=True)

# Merge class 44 into class 28
df.loc[df["category_id"] == 44, "category_id"] = 28
print(f"Shape: {df.shape} | Classes: {df['category_id'].nunique()}")

# ── Label Encoding ────────────────────────────────────────────────────────────

cat_encoder = LabelEncoder()
df["category_encoded"] = cat_encoder.fit_transform(df["category_id"])

with open(os.path.join(args.output_dir, "category_encoder.pkl"), "wb") as f:
    pickle.dump(cat_encoder, f)

NUM_CLASSES = int(df["category_encoded"].nunique())
print(f"Classes: {NUM_CLASSES}")

# ── Train / Test Split ────────────────────────────────────────────────────────

train_df, test_df = train_test_split(
    df, test_size=0.2,
    stratify=df["category_id"],
    random_state=42
)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"Train: {len(train_df)} | Test: {len(test_df)}")

# ── Transforms ────────────────────────────────────────────────────────────────

IMG_SIZE = args.img_size

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomGrayscale(p=0.1),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),  # learned augmentation
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
        return image, int(row["category_encoded"])

# ── MixUp ─────────────────────────────────────────────────────────────────────

def mixup_batch(images, labels, num_classes, alpha=0.4):
    """Blends two images and their labels together."""
    lam    = np.random.beta(alpha, alpha)
    index  = torch.randperm(images.size(0))
    mixed  = lam * images + (1 - lam) * images[index]

    # Convert to one-hot for soft labels
    labels_a = torch.zeros(images.size(0), num_classes).scatter_(1, labels.view(-1, 1), 1)
    labels_b = torch.zeros(images.size(0), num_classes).scatter_(1, labels[index].view(-1, 1), 1)
    mixed_labels = lam * labels_a + (1 - lam) * labels_b

    return mixed, mixed_labels

def mixup_criterion(criterion, pred, mixed_labels):
    """Loss for soft mixed labels."""
    log_prob = torch.nn.functional.log_softmax(pred, dim=1)
    return -(mixed_labels * log_prob).sum(dim=1).mean()

# ── CutMix ────────────────────────────────────────────────────────────────────

def cutmix_batch(images, labels, num_classes, alpha=1.0):
    """Cuts a patch from one image and pastes into another."""
    lam   = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0))
    
    _, _, H, W = images.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w   = int(W * cut_rat)
    cut_h   = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed        = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam          = 1 - ((x2 - x1) * (y2 - y1)) / (W * H)

    labels_a     = torch.zeros(images.size(0), num_classes).scatter_(1, labels.view(-1, 1), 1)
    labels_b     = torch.zeros(images.size(0), num_classes).scatter_(1, labels[index].view(-1, 1), 1)
    mixed_labels = lam * labels_a + (1 - lam) * labels_b

    return mixed, mixed_labels

# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_loaders(train_df, test_df, image_dir, batch_size, num_workers):
    train_dataset = ProductDataset(train_df, image_dir, transform=train_transforms)
    test_dataset  = ProductDataset(test_df,  image_dir, transform=test_transforms)

    # WeightedRandomSampler
    class_counts   = train_df["category_encoded"].value_counts().sort_index()
    weights        = 1.0 / class_counts
    sample_weights = train_df["category_encoded"].map(weights).values.copy()
    sampler        = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,   num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader

train_loader, test_loader = get_loaders(train_df, test_df, IMAGE_DIR, args.batch_size, args.num_workers)

# ── Sanity Check ──────────────────────────────────────────────────────────────

images, categories = next(iter(train_loader))
print(f"\nBatch shape:    {images.shape}")
print(f"Category range: {categories.min()}-{categories.max()}")
print(f"NUM_CLASSES:    {NUM_CLASSES}")
print("\ndataset.py loaded successfully.")
