"""
evaluate_tta.py — Evaluate CLIP model with Test Time Augmentation
=================================================================
Runs inference with multiple augmentations and averages predictions.

Usage:
    python evaluate_tta.py --data_dir /workspace/Data/train --output_dir /workspace/outputs
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",    type=str, default="/workspace/Data/train")
parser.add_argument("--output_dir",  type=str, default="/workspace/outputs")
parser.add_argument("--n_augments",  type=int, default=5)
parser.add_argument("--batch_size",  type=int, default=32)
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

os.environ["DATA_DIR"]    = args.data_dir
os.environ["OUTPUT_DIR"]  = args.output_dir
os.environ["BATCH_SIZE"]  = str(args.batch_size)
os.environ["NUM_WORKERS"] = str(args.num_workers)
os.environ["IMG_SIZE"]    = "224"

from dataset import (
    test_df, IMAGE_DIR,
    NUM_CLASSES, NUM_MAIN,
    test_loader,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── TTA Transforms ─────────────────────────────────────────────────────────────
tta_transforms = [
    # Original — no augmentation
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Horizontal flip
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Slight rotation
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Center crop
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Slight zoom
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    # Color jitter
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
]

# Use only n_augments transforms
tta_transforms = tta_transforms[:args.n_augments + 1]  # +1 for original
print(f"Using {len(tta_transforms)} augmentations (original + {len(tta_transforms)-1})")

# ── TTA Dataset ───────────────────────────────────────────────────────────────
class TTADataset(Dataset):
    def __init__(self, df, image_dir):
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, str(row["category_id"]), f"{row['anonymous_id']}.jpg")
        image    = Image.open(img_path).convert("RGB")
        return (
            image,
            int(row["main_encoded"]),
            int(row["category_encoded"])
        )

def tta_collate(batch):
    images, main_labels, sub_labels = zip(*batch)
    return list(images), torch.tensor(main_labels), torch.tensor(sub_labels)

tta_dataset = TTADataset(test_df, IMAGE_DIR)
tta_loader  = DataLoader(
    tta_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
    collate_fn=tta_collate
)

# ── Load CLIP Model ───────────────────────────────────────────────────────────
print("\nLoading CLIP model...")

clip_model  = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
feature_dim = 768

class CLIPHierarchicalModel(nn.Module):
    def __init__(self, clip_model, feature_dim, num_main, num_sub):
        super().__init__()
        self.backbone = clip_model.vision_model
        self.dropout  = nn.Dropout(0.3)

        self.main_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_main)
        )

        self.sub_head = nn.Sequential(
            nn.Linear(feature_dim + num_main, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_sub)
        )

    def forward(self, pixel_values):
        outputs     = self.backbone(pixel_values=pixel_values)
        features    = self.dropout(outputs.pooler_output)
        main_logits = self.main_head(features)
        sub_input   = torch.cat([features, main_logits], dim=1)
        sub_logits  = self.sub_head(sub_input)
        return main_logits, sub_logits

model     = CLIPHierarchicalModel(clip_model, feature_dim, NUM_MAIN, NUM_CLASSES).to(device)
ckpt_path = os.path.join(args.output_dir, "clip_hierarchical_best.pth")
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
print(f"Loaded checkpoint from {ckpt_path}")

# ── TTA Evaluation ────────────────────────────────────────────────────────────
print(f"\nRunning TTA evaluation with {len(tta_transforms)} augmentations...")

all_main_preds  = []
all_sub_preds   = []
all_main_labels = []
all_sub_labels  = []

with torch.no_grad():
    for batch_idx, (images, main_labels, sub_labels) in enumerate(tta_loader):
        main_labels = main_labels.to(device)
        sub_labels  = sub_labels.to(device)

        # Run each augmentation
        main_probs_list = []
        sub_probs_list  = []

        for transform in tta_transforms:
            # Apply transform to each image in batch
            aug_images = torch.stack([transform(img) for img in images]).to(device)

            main_logits, sub_logits = model(aug_images)
            main_probs_list.append(F.softmax(main_logits, dim=1))
            sub_probs_list.append(F.softmax(sub_logits,  dim=1))

        # Average probabilities across all augmentations
        avg_main_probs = torch.stack(main_probs_list).mean(dim=0)
        avg_sub_probs  = torch.stack(sub_probs_list).mean(dim=0)

        main_preds = avg_main_probs.argmax(dim=1)
        sub_preds  = avg_sub_probs.argmax(dim=1)

        all_main_preds.extend(main_preds.cpu().numpy())
        all_sub_preds.extend(sub_preds.cpu().numpy())
        all_main_labels.extend(main_labels.cpu().numpy())
        all_sub_labels.extend(sub_labels.cpu().numpy())

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch {batch_idx+1}/{len(tta_loader)}")

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"TTA Results ({len(tta_transforms)} augmentations)")
print(f"{'='*50}")
print(f"\n--- Main Category (21 classes) ---")
print(f"Test Accuracy: {accuracy_score(all_main_labels, all_main_preds):.4f}")
print(f"F1 Macro:      {f1_score(all_main_labels, all_main_preds, average='macro'):.4f}")

print(f"\n--- Subcategory (190 classes) ---")
print(f"Test Accuracy: {accuracy_score(all_sub_labels, all_sub_preds):.4f}")
print(f"F1 Macro:      {f1_score(all_sub_labels, all_sub_preds, average='macro'):.4f}")
print(f"F1 Weighted:   {f1_score(all_sub_labels, all_sub_preds, average='weighted'):.4f}")
print(f"{'='*50}")