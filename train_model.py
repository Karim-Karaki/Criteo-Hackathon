"""
train_model.py — Model training using dataset.py
=================================================
Vast.ai usage:
    python train_model.py --data_dir /workspace/data/train --output_dir /workspace/outputs --model vit
    python train_model.py --data_dir /workspace/data/train --output_dir /workspace/outputs --model efficientnet
"""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import models
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

# Import everything from dataset.py
from dataset import (
    args as data_args,
    train_df, test_df, IMAGE_DIR,
    NUM_CLASSES, cat_encoder,
    train_loader, test_loader,
    mixup_batch, mixup_criterion,
    cutmix_batch
)

# ── Argument Parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--model",         type=str, default="efficientnet", choices=["vit", "efficientnet"])
parser.add_argument("--output_dir",    type=str, default="/workspace/outputs")
parser.add_argument("--phase1_epochs", type=int, default=8)
parser.add_argument("--phase2_epochs", type=int, default=20)
parser.add_argument("--use_mixup",     action="store_true", default=True)
parser.add_argument("--use_cutmix",    action="store_true", default=True)
parser.add_argument("--mixup_prob",    type=float, default=0.5)  # probability of applying mixup/cutmix per batch
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Model ─────────────────────────────────────────────────────────────────────

print(f"\n--- Building {args.model} for {NUM_CLASSES} classes ---")

if args.model == "vit":
    vit       = models.vit_b_16(weights="IMAGENET1K_V1")
    vit.heads = nn.Identity()

    class ViTCategory(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone      = backbone
            self.dropout       = nn.Dropout(0.1)
            self.category_head = nn.Linear(768, NUM_CLASSES)

        def forward(self, x):
            return self.category_head(self.dropout(self.backbone(x)))

    model      = ViTCategory(vit).to(device)
    ckpt_path  = os.path.join(args.output_dir, "vit_best.pth")
    backbone   = model.backbone
    head       = model.category_head

else:  # efficientnet
    efficientnet = models.efficientnet_b3(weights="IMAGENET1K_V1")
    efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, NUM_CLASSES)
    model      = efficientnet.to(device)
    ckpt_path  = os.path.join(args.output_dir, "efficientnet_b3_best.pth")
    backbone   = model.features
    head       = model.classifier

# ── Training Functions ────────────────────────────────────────────────────────

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

import numpy as np

def train_epoch(model, loader, optimizer, use_augmix=False):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, categories in loader:
        images     = images.to(device)
        categories = categories.to(device)

        # Randomly apply MixUp or CutMix
        if use_augmix and np.random.random() < args.mixup_prob:
            if np.random.random() < 0.5:
                images, mixed_labels = mixup_batch(images, categories, NUM_CLASSES)
            else:
                images, mixed_labels = cutmix_batch(images, categories, NUM_CLASSES)
            mixed_labels = mixed_labels.to(device)

            optimizer.zero_grad()
            cat_out = model(images)
            loss    = mixup_criterion(criterion, cat_out, mixed_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (cat_out.argmax(1) == categories).sum().item()

        else:
            optimizer.zero_grad()
            cat_out = model(images)
            loss    = criterion(cat_out, categories)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (cat_out.argmax(1) == categories).sum().item()

        total += categories.size(0)

    return total_loss / len(loader), correct / total

# ── Phase 1 — Freeze backbone ─────────────────────────────────────────────────

print("\n--- Phase 1: Head only ---")
for param in backbone.parameters():
    param.requires_grad = False

optimizer = Adam(head.parameters(), lr=1e-3)

for epoch in range(args.phase1_epochs):
    loss, acc = train_epoch(model, train_loader, optimizer, use_augmix=False)
    print(f"Epoch {epoch+1}/{args.phase1_epochs} | Loss: {loss:.4f} | Acc: {acc:.4f}")

# ── Phase 2 — Full fine-tuning ────────────────────────────────────────────────

print("\n--- Phase 2: Full fine-tuning with MixUp + CutMix ---")
for param in backbone.parameters():
    param.requires_grad = True

optimizer = Adam([
    {"params": backbone.parameters(), "lr": 1e-5, "weight_decay": 1e-4},
    {"params": head.parameters(),     "lr": 1e-4, "weight_decay": 1e-4},
])

scheduler = CosineAnnealingLR(optimizer, T_max=args.phase2_epochs)

best_acc = 0

for epoch in range(args.phase2_epochs):
    loss, acc = train_epoch(model, train_loader, optimizer, use_augmix=True)
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}/{args.phase2_epochs} | Loss: {loss:.4f} | Acc: {acc:.4f} | LR: {current_lr:.2e}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), ckpt_path)
        print(f"  -> Saved best model ({best_acc:.4f})")

# ── Evaluation ────────────────────────────────────────────────────────────────

print("\n--- Evaluating on test set ---")
model.load_state_dict(torch.load(ckpt_path))
model.eval()

correct    = 0
total      = len(test_loader.dataset)
all_preds  = []
all_labels = []

with torch.no_grad():
    for images, categories in test_loader:
        images     = images.to(device)
        categories = categories.to(device)
        cat_out    = model(images)
        preds      = cat_out.argmax(1)
        correct   += (preds == categories).sum().item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(categories.cpu().numpy())

print(f"Test Accuracy:   {correct/total:.4f}")
print(f"Train Accuracy:  {best_acc:.4f}")
print(f"Overfitting gap: {best_acc - correct/total:.4f}")
print(f"F1 Macro:        {f1_score(all_labels, all_preds, average='macro'):.4f}")
print(f"F1 Weighted:     {f1_score(all_labels, all_preds, average='weighted'):.4f}")
print("\nDone.")
