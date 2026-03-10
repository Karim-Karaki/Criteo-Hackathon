"""
train_model.py — Hierarchical Classification
=============================================
Predicts main category (21 classes) first, then subcategory (190 classes).
Two-head model with shared EfficientNet-B3 or ViT backbone.

Vast.ai usage:
    python train_model.py --data_dir /workspace/Data/train --output_dir /workspace/outputs --model efficientnet
    python train_model.py --data_dir /workspace/Data/train --output_dir /workspace/outputs --model vit
"""

import os
import argparse
import torch
import torch.nn as nn
from torchvision import models
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

# ── Argument Parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",      type=str, default="/workspace/Data/train")
parser.add_argument("--output_dir",    type=str, default="/workspace/outputs")
parser.add_argument("--model",         type=str, default="efficientnet", choices=["vit", "efficientnet"])
parser.add_argument("--batch_size",    type=int, default=64)
parser.add_argument("--num_workers",   type=int, default=4)
parser.add_argument("--img_size",      type=int, default=224)
parser.add_argument("--phase1_epochs", type=int, default=8)
parser.add_argument("--phase2_epochs", type=int, default=20)
parser.add_argument("--main_weight",   type=float, default=0.3,  help="Loss weight for main category head")
parser.add_argument("--sub_weight",    type=float, default=0.7,  help="Loss weight for subcategory head")
args = parser.parse_args()

# Pass args to dataset.py via environment variables
os.environ["DATA_DIR"]    = args.data_dir
os.environ["OUTPUT_DIR"]  = args.output_dir
os.environ["BATCH_SIZE"]  = str(args.batch_size)
os.environ["NUM_WORKERS"] = str(args.num_workers)
os.environ["IMG_SIZE"]    = str(args.img_size)

# ── Import dataset ────────────────────────────────────────────────────────────
from dataset import (
    train_df, test_df, IMAGE_DIR,
    NUM_CLASSES, NUM_MAIN,
    train_loader, test_loader,
)

os.makedirs(args.output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Hierarchical Model ────────────────────────────────────────────────────────
print(f"\n--- Building Hierarchical {args.model} ---")
print(f"Main categories: {NUM_MAIN} | Subcategories: {NUM_CLASSES}")

if args.model == "efficientnet":
    backbone     = models.efficientnet_b3(weights="IMAGENET1K_V1")
    feature_dim  = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()  # remove original classifier

elif args.model == "vit":
    backbone     = models.vit_b_16(weights="IMAGENET1K_V1")
    feature_dim  = 768
    backbone.heads = nn.Identity()

class HierarchicalModel(nn.Module):
    def __init__(self, backbone, feature_dim, num_main, num_sub):
        super().__init__()
        self.backbone = backbone
        self.dropout  = nn.Dropout(0.3)

        # Main category head (21 classes)
        self.main_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_main)
        )

        # Subcategory head (190 classes)
        # Takes features + main category logits as extra signal
        self.sub_head = nn.Sequential(
            nn.Linear(feature_dim + num_main, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_sub)
        )

    def forward(self, x):
        features   = self.dropout(self.backbone(x))        # shared features
        main_logits = self.main_head(features)              # predict main category
        # Concatenate features with main logits to inform subcategory prediction
        sub_input  = torch.cat([features, main_logits], dim=1)
        sub_logits = self.sub_head(sub_input)               # predict subcategory
        return main_logits, sub_logits

model     = HierarchicalModel(backbone, feature_dim, NUM_MAIN, NUM_CLASSES).to(device)
ckpt_path = os.path.join(args.output_dir, f"hierarchical_{args.model}_best.pth")

# ── Loss & Training ───────────────────────────────────────────────────────────
main_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
sub_criterion  = nn.CrossEntropyLoss(label_smoothing=0.1)

def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, main_correct, sub_correct, total = 0, 0, 0, 0

    for images, main_labels, sub_labels in loader:
        images      = images.to(device)
        main_labels = main_labels.to(device)
        sub_labels  = sub_labels.to(device)

        optimizer.zero_grad()
        main_logits, sub_logits = model(images)

        main_loss = main_criterion(main_logits, main_labels)
        sub_loss  = sub_criterion(sub_logits,  sub_labels)
        loss      = args.main_weight * main_loss + args.sub_weight * sub_loss

        loss.backward()
        optimizer.step()

        total_loss   += loss.item()
        main_correct += (main_logits.argmax(1) == main_labels).sum().item()
        sub_correct  += (sub_logits.argmax(1) == sub_labels).sum().item()
        total        += sub_labels.size(0)

    return total_loss / len(loader), main_correct / total, sub_correct / total

# ── Phase 1 — Freeze backbone ─────────────────────────────────────────────────
print("\n--- Phase 1: Heads only ---")
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = Adam([
    {"params": model.main_head.parameters(), "lr": 1e-3},
    {"params": model.sub_head.parameters(),  "lr": 1e-3},
])

for epoch in range(args.phase1_epochs):
    loss, main_acc, sub_acc = train_epoch(model, train_loader, optimizer)
    print(f"Epoch {epoch+1}/{args.phase1_epochs} | Loss: {loss:.4f} | Main Acc: {main_acc:.4f} | Sub Acc: {sub_acc:.4f}")

# ── Phase 2 — Full fine-tuning ────────────────────────────────────────────────
print("\n--- Phase 2: Full fine-tuning ---")
for param in model.backbone.parameters():
    param.requires_grad = True

optimizer = Adam([
    {"params": model.backbone.parameters(),  "lr": 1e-5, "weight_decay": 1e-4},
    {"params": model.main_head.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
    {"params": model.sub_head.parameters(),  "lr": 1e-4, "weight_decay": 1e-4},
])

scheduler = CosineAnnealingLR(optimizer, T_max=args.phase2_epochs)
best_sub_acc = 0

for epoch in range(args.phase2_epochs):
    loss, main_acc, sub_acc = train_epoch(model, train_loader, optimizer)
    scheduler.step()
    print(f"Epoch {epoch+1}/{args.phase2_epochs} | Loss: {loss:.4f} | Main Acc: {main_acc:.4f} | Sub Acc: {sub_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    if sub_acc > best_sub_acc:
        best_sub_acc = sub_acc
        torch.save(model.state_dict(), ckpt_path)
        print(f"  -> Saved best model (Sub Acc: {best_sub_acc:.4f})")

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\n--- Evaluating on test set ---")
model.load_state_dict(torch.load(ckpt_path))
model.eval()

main_correct, sub_correct = 0, 0
total = len(test_loader.dataset)
all_sub_preds, all_sub_labels = [], []
all_main_preds, all_main_labels = [], []

with torch.no_grad():
    for images, main_labels, sub_labels in test_loader:
        images      = images.to(device)
        main_labels = main_labels.to(device)
        sub_labels  = sub_labels.to(device)

        main_logits, sub_logits = model(images)

        main_preds = main_logits.argmax(1)
        sub_preds  = sub_logits.argmax(1)

        main_correct += (main_preds == main_labels).sum().item()
        sub_correct  += (sub_preds  == sub_labels).sum().item()

        all_main_preds.extend(main_preds.cpu().numpy())
        all_main_labels.extend(main_labels.cpu().numpy())
        all_sub_preds.extend(sub_preds.cpu().numpy())
        all_sub_labels.extend(sub_labels.cpu().numpy())

print(f"\n--- Main Category (21 classes) ---")
print(f"Test Accuracy: {main_correct/total:.4f}")
print(f"F1 Macro:      {f1_score(all_main_labels, all_main_preds, average='macro'):.4f}")

print(f"\n--- Subcategory (190 classes) ---")
print(f"Test Accuracy:   {sub_correct/total:.4f}")
print(f"Train Accuracy:  {best_sub_acc:.4f}")
print(f"Overfitting gap: {best_sub_acc - sub_correct/total:.4f}")
print(f"F1 Macro:        {f1_score(all_sub_labels, all_sub_preds, average='macro'):.4f}")
print(f"F1 Weighted:     {f1_score(all_sub_labels, all_sub_preds, average='weighted'):.4f}")
print("\nDone.")
