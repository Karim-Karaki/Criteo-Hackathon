"""
ensemble_inference.py
---------------------
Ensembles CLIP hierarchical model + DINOv2-Base model for subcategory prediction.
Both models must have been trained with the same preprocessing and output 190 classes.

Usage:
    python ensemble_inference.py \
        --clip_path /workspace/outputs/clip_hierarchical_best.pth \
        --dino_path /workspace/outputs/dino_best.pth \
        --data_dir /workspace/Data/train \
        --output_path /workspace/outputs/ensemble_predictions.parquet \
        --clip_weight 0.6 \
        --dino_weight 0.4
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import CLIPVisionModel
import math

# ── Constants ─────────────────────────────────────────────────────────────────
NUM_MAIN   = 21
NUM_SUB    = 190
IMG_SIZE   = 224


# ── Preprocessing (same for both models) ──────────────────────────────────────
def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────
class ProductDataset(Dataset):
    def __init__(self, df, data_dir, transform):
        self.df        = df.reset_index(drop=True)
        self.data_dir  = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, "images",
                                str(int(row["category_id"])),
                                f"{row['anonymous_id']}.jpg")
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255))
        return self.transform(img), row["anonymous_id"]


# ── CLIP Hierarchical Model ────────────────────────────────────────────────────
class CLIPHierarchicalModel(nn.Module):
    def __init__(self, num_main=NUM_MAIN, num_sub=NUM_SUB):
        super().__init__()
        self.backbone  = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        hidden         = self.backbone.config.hidden_size  # 768

        self.main_head = nn.Sequential(
            nn.Linear(hidden, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_main)
        )
        self.sub_head  = nn.Sequential(
            nn.Linear(hidden + num_main, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_sub)
        )

    def forward(self, x):
        features   = self.backbone(pixel_values=x).pooler_output
        main_logits = self.main_head(features)
        sub_logits  = self.sub_head(torch.cat([features, main_logits], dim=1))
        return main_logits, sub_logits

    def get_sub_probs(self, x):
        _, sub_logits = self.forward(x)
        return F.softmax(sub_logits, dim=1)


# ── DINOv2-Base Hierarchical Model ────────────────────────────────────────────
class DINOv2HierarchicalModel(nn.Module):
    def __init__(self, num_main=NUM_MAIN, num_sub=NUM_SUB):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14',
                                       pretrained=False)
        hidden        = 768  # DINOv2-Base hidden size

        self.main_head = nn.Sequential(
            nn.Linear(hidden, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_main)
        )
        self.sub_head  = nn.Sequential(
            nn.Linear(hidden + num_main, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_sub)
        )

    def forward(self, x):
        features    = self.backbone(x)           # (B, 768) CLS token
        main_logits = self.main_head(features)
        sub_logits  = self.sub_head(torch.cat([features, main_logits], dim=1))
        return main_logits, sub_logits

    def get_sub_probs(self, x):
        _, sub_logits = self.forward(x)
        return F.softmax(sub_logits, dim=1)


# ── Load model from state dict ─────────────────────────────────────────────────
def load_clip_model(path, device):
    model = CLIPHierarchicalModel()
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"CLIP model loaded from {path}")
    return model


def load_dino_model(path, device):
    model = DINOv2HierarchicalModel()
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"DINO model loaded from {path}")
    return model


# ── Ensemble Inference ─────────────────────────────────────────────────────────
def run_ensemble(clip_model, dino_model, loader, device, clip_w, dino_w):
    all_ids   = []
    all_preds = []

    with torch.no_grad():
        for i, (imgs, ids) in enumerate(loader):
            imgs = imgs.to(device)

            clip_probs = clip_model.get_sub_probs(imgs)   # (B, 190)
            dino_probs = dino_model.get_sub_probs(imgs)   # (B, 190)

            # Weighted average ensemble
            ensemble_probs = clip_w * clip_probs + dino_w * dino_probs
            preds          = ensemble_probs.argmax(dim=1).cpu().numpy()

            all_ids.extend(ids)
            all_preds.extend(preds.tolist())

            if (i + 1) % 50 == 0:
                print(f"  Processed {(i+1) * loader.batch_size} images...")

    return all_ids, all_preds


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_path",   type=str, required=True)
    parser.add_argument("--dino_path",   type=str, required=True)
    parser.add_argument("--data_dir",    type=str, default="/workspace/Data/train")
    parser.add_argument("--output_path", type=str, default="/workspace/outputs/ensemble_predictions.parquet")
    parser.add_argument("--clip_weight", type=float, default=0.6)
    parser.add_argument("--dino_weight", type=float, default=0.4)
    parser.add_argument("--batch_size",  type=int, default=64)
    return parser.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ensemble weights — CLIP: {args.clip_weight}, DINO: {args.dino_weight}")

    # Load data
    df = pd.read_parquet(os.path.join(args.data_dir, "train.parquet"))
    print(f"Loaded {len(df)} samples")

    # Build dataset
    transform = get_transform()
    dataset   = ProductDataset(df, args.data_dir, transform)
    loader    = DataLoader(dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=4, pin_memory=True)

    # Load models
    clip_model = load_clip_model(args.clip_path, device)
    dino_model = load_dino_model(args.dino_path, device)

    # Run ensemble
    print("\nRunning ensemble inference...")
    ids, preds = run_ensemble(clip_model, dino_model, loader, device,
                              args.clip_weight, args.dino_weight)

    # Build submission
    # Map predicted index back to actual category_id
    # (assumes model was trained with sorted unique category_ids as class indices)
    unique_cats = sorted(df["category_id"].unique())
    idx_to_cat  = {i: c for i, c in enumerate(unique_cats)}
    predicted_category_ids = [idx_to_cat[p] for p in preds]

    out_df = pd.DataFrame({
        "anonymous_id":        ids,
        "predicted_category_id": predicted_category_ids,
    })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    out_df.to_parquet(args.output_path, index=False)
    print(f"\nSaved {len(out_df)} predictions → {args.output_path}")

    # Quick accuracy check if ground truth available
    if "category_id" in df.columns:
        merged   = out_df.merge(df[["anonymous_id", "category_id"]], on="anonymous_id")
        accuracy = (merged["predicted_category_id"] == merged["category_id"]).mean()
        print(f"Subcategory Accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()