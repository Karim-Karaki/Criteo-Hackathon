"""
train_multitask.py — CLIP Multitask: Category + Color + Gender
==============================================================
Architecture:
    Image → CLIP → features
                    ├── main_head  → 21 classes
                    ├── sub_head   → concat(features, main_logits) → 190 classes
                    ├── color_head → 12 classes
                    └── gender_head→ concat(features, main_logits, color_logits) → 4 classes

Loss weights: category 0.6, color 0.2, gender 0.2
Main/sub split: 0.3 main + 0.7 sub (within category loss)

Usage:
    python train_multitask.py --data_dir /workspace/Data/train --output_dir /workspace/outputs
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, classification_report
from transformers import CLIPModel

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",      type=str,  default="/workspace/Data/train")
parser.add_argument("--output_dir",    type=str,  default="/workspace/outputs")
parser.add_argument("--batch_size",    type=int,  default=64)
parser.add_argument("--num_workers",   type=int,  default=4)
parser.add_argument("--phase1_epochs", type=int,  default=8)
parser.add_argument("--phase2_epochs", type=int,  default=25)
parser.add_argument("--patience",      type=int,  default=5)
parser.add_argument("--min_delta",     type=float, default=0.001)
args = parser.parse_args()

os.environ["DATA_DIR"]    = args.data_dir
os.environ["OUTPUT_DIR"]  = args.output_dir
os.environ["BATCH_SIZE"]  = str(args.batch_size)
os.environ["NUM_WORKERS"] = str(args.num_workers)
os.environ["IMG_SIZE"]    = "224"

from Datasetmultitask import (
    train_df, test_df, IMAGE_DIR,
    NUM_MAIN, NUM_CLASSES, NUM_COLORS, NUM_GENDERS,
    color_encoder, gender_encoder, cat_encoder, main_encoder,
    train_loader, test_loader,
)

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── Early Stopping ────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_acc  = 0

    def step(self, acc):
        if acc > self.best_acc + self.min_delta:
            self.best_acc = acc
            self.counter  = 0
            return False
        self.counter += 1
        print(f"  Early stopping: {self.counter}/{self.patience}")
        return self.counter >= self.patience

# ── Class Weighted Losses ─────────────────────────────────────────────────────
def make_weights(train_df, col, num_classes, device):
    counts  = train_df[col].value_counts().sort_index()
    weights = 1.0 / torch.tensor(counts.values, dtype=torch.float)
    return (weights / weights.sum() * num_classes).to(device)

main_criterion   = nn.CrossEntropyLoss(weight=make_weights(train_df, "main_encoded",     NUM_MAIN,    device), label_smoothing=0.1)
sub_criterion    = nn.CrossEntropyLoss(weight=make_weights(train_df, "category_encoded", NUM_CLASSES, device), label_smoothing=0.1)
gender_criterion = nn.CrossEntropyLoss(weight=make_weights(train_df, "gender_encoded",   NUM_GENDERS, device), label_smoothing=0.1)

# Color loss — only on samples where color is known (color_encoded != -1)
color_counts  = train_df[train_df["color_encoded"] != -1]["color_encoded"].value_counts().sort_index()
color_weights = 1.0 / torch.tensor(color_counts.values, dtype=torch.float)
color_weights = (color_weights / color_weights.sum() * NUM_COLORS).to(device)
color_criterion = nn.CrossEntropyLoss(weight=color_weights, label_smoothing=0.1)

# ── Model ─────────────────────────────────────────────────────────────────────
print("\n--- Building CLIP Multitask Model ---")
clip_model  = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
feature_dim = 768

class CLIPMultitaskModel(nn.Module):
    def __init__(self, clip_model, feature_dim, num_main, num_sub, num_colors, num_genders):
        super().__init__()
        self.backbone = clip_model.vision_model
        self.dropout  = nn.Dropout(0.3)

        # Category heads (hierarchical)
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

        # Color head
        self.color_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_colors)
        )

        # Gender head — uses features + main logits + color logits
        self.gender_head = nn.Sequential(
            nn.Linear(feature_dim + num_main + num_colors, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_genders)
        )

    def forward(self, x):
        features     = self.dropout(self.backbone(pixel_values=x).pooler_output)

        # Category (hierarchical)
        main_logits  = self.main_head(features)
        sub_logits   = self.sub_head(torch.cat([features, main_logits], dim=1))

        # Color
        color_logits = self.color_head(features)

        # Gender — conditioned on main category + color
        gender_input  = torch.cat([features, main_logits, color_logits], dim=1)
        gender_logits = self.gender_head(gender_input)

        return main_logits, sub_logits, color_logits, gender_logits

model     = CLIPMultitaskModel(clip_model, feature_dim, NUM_MAIN, NUM_CLASSES, NUM_COLORS, NUM_GENDERS).to(device)
ckpt_path = os.path.join(args.output_dir, "clip_multitask_best.pth")
print(f"Model built — Main: {NUM_MAIN} | Sub: {NUM_CLASSES} | Colors: {NUM_COLORS} | Genders: {NUM_GENDERS}")

# ── Training Function ─────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    main_correct = sub_correct = color_correct = gender_correct = total = 0

    for images, main_labels, sub_labels, color_labels, gender_labels in loader:
        images        = images.to(device)
        main_labels   = main_labels.to(device)
        sub_labels    = sub_labels.to(device)
        color_labels  = color_labels.to(device)
        gender_labels = gender_labels.to(device)

        optimizer.zero_grad()
        main_logits, sub_logits, color_logits, gender_logits = model(images)

        # Category loss (hierarchical) — weight 0.6
        main_loss = main_criterion(main_logits, main_labels)
        sub_loss  = sub_criterion(sub_logits,   sub_labels)
        cat_loss  = 0.3 * main_loss + 0.7 * sub_loss

        # Color loss — only on known samples, weight 0.2
        color_mask = color_labels != -1
        if color_mask.sum() > 0:
            color_loss = color_criterion(color_logits[color_mask], color_labels[color_mask])
        else:
            color_loss = torch.tensor(0.0, device=device)

        # Gender loss — weight 0.2
        gender_loss = gender_criterion(gender_logits, gender_labels)

        # Combined loss
        loss = 0.6 * cat_loss + 0.2 * color_loss + 0.2 * gender_loss
        loss.backward()
        optimizer.step()

        total_loss    += loss.item()
        main_correct  += (main_logits.argmax(1) == main_labels).sum().item()
        sub_correct   += (sub_logits.argmax(1)  == sub_labels).sum().item()
        gender_correct += (gender_logits.argmax(1) == gender_labels).sum().item()
        if color_mask.sum() > 0:
            color_correct += (color_logits[color_mask].argmax(1) == color_labels[color_mask]).sum().item()
        total += sub_labels.size(0)

    return (total_loss / len(loader),
            main_correct / total,
            sub_correct  / total,
            color_correct / total,
            gender_correct / total)

# ── Phase 1 — Heads only ──────────────────────────────────────────────────────
print("\n--- Phase 1: Heads only ---")
for param in model.backbone.parameters():
    param.requires_grad = False

optimizer = Adam([
    {"params": model.main_head.parameters(),   "lr": 1e-3},
    {"params": model.sub_head.parameters(),    "lr": 1e-3},
    {"params": model.color_head.parameters(),  "lr": 1e-3},
    {"params": model.gender_head.parameters(), "lr": 1e-3},
])

for epoch in range(args.phase1_epochs):
    loss, main_acc, sub_acc, color_acc, gender_acc = train_epoch(model, train_loader, optimizer)
    print(f"Epoch {epoch+1}/{args.phase1_epochs} | Loss: {loss:.4f} | Sub: {sub_acc:.4f} | Color: {color_acc:.4f} | Gender: {gender_acc:.4f}")

# ── Phase 2 — Unfreeze last 3 CLIP blocks ────────────────────────────────────
print("\n--- Phase 2: Full fine-tuning ---")
for param in model.backbone.parameters():
    param.requires_grad = False

for i, layer in enumerate(model.backbone.encoder.layers):
    if i >= len(model.backbone.encoder.layers) - 3:
        for param in layer.parameters():
            param.requires_grad = True
for param in model.backbone.post_layernorm.parameters():
    param.requires_grad = True

optimizer = Adam([
    {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 1e-5, "weight_decay": 1e-4},
    {"params": model.main_head.parameters(),   "lr": 1e-4, "weight_decay": 1e-4},
    {"params": model.sub_head.parameters(),    "lr": 1e-4, "weight_decay": 1e-4},
    {"params": model.color_head.parameters(),  "lr": 1e-4, "weight_decay": 1e-4},
    {"params": model.gender_head.parameters(), "lr": 1e-4, "weight_decay": 1e-4},
])

scheduler      = CosineAnnealingLR(optimizer, T_max=args.phase2_epochs)
best_sub_acc   = 0
early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

for epoch in range(args.phase2_epochs):
    loss, main_acc, sub_acc, color_acc, gender_acc = train_epoch(model, train_loader, optimizer)
    scheduler.step()
    print(f"Epoch {epoch+1}/{args.phase2_epochs} | Loss: {loss:.4f} | Sub: {sub_acc:.4f} | Color: {color_acc:.4f} | Gender: {gender_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    if sub_acc > best_sub_acc:
        best_sub_acc = sub_acc
        torch.save(model.state_dict(), ckpt_path)
        print(f"  -> Saved best model (Sub Acc: {best_sub_acc:.4f})")

    if early_stopping.step(sub_acc):
        print("Early stopping triggered.")
        break

# ── Evaluation ────────────────────────────────────────────────────────────────
print("\n--- Evaluating on test set ---")
model.load_state_dict(torch.load(ckpt_path))
model.eval()

all_main_preds = []; all_main_labels  = []
all_sub_preds  = []; all_sub_labels   = []
all_color_preds= []; all_color_labels = []
all_gen_preds  = []; all_gen_labels   = []
total = 0

with torch.no_grad():
    for images, main_labels, sub_labels, color_labels, gender_labels in test_loader:
        images        = images.to(device)
        main_labels   = main_labels.to(device)
        sub_labels    = sub_labels.to(device)
        color_labels  = color_labels.to(device)
        gender_labels = gender_labels.to(device)

        main_logits, sub_logits, color_logits, gender_logits = model(images)

        all_main_preds.extend(main_logits.argmax(1).cpu().numpy())
        all_main_labels.extend(main_labels.cpu().numpy())
        all_sub_preds.extend(sub_logits.argmax(1).cpu().numpy())
        all_sub_labels.extend(sub_labels.cpu().numpy())
        all_gen_preds.extend(gender_logits.argmax(1).cpu().numpy())
        all_gen_labels.extend(gender_labels.cpu().numpy())

        # Color — only evaluate on known samples
        color_mask = color_labels != -1
        if color_mask.sum() > 0:
            all_color_preds.extend(color_logits[color_mask].argmax(1).cpu().numpy())
            all_color_labels.extend(color_labels[color_mask].cpu().numpy())

        total += sub_labels.size(0)

print(f"\n{'='*55}")
print(f"--- Main Category (21 classes) ---")
print(f"Accuracy: {sum(p==l for p,l in zip(all_main_preds, all_main_labels))/total:.4f}")
print(f"F1 Macro: {f1_score(all_main_labels, all_main_preds, average='macro'):.4f}")

print(f"\n--- Subcategory (190 classes) ---")
print(f"Test Accuracy:   {sum(p==l for p,l in zip(all_sub_preds, all_sub_labels))/total:.4f}")
print(f"Train Accuracy:  {best_sub_acc:.4f}")
print(f"F1 Macro:        {f1_score(all_sub_labels, all_sub_preds, average='macro'):.4f}")
print(f"F1 Weighted:     {f1_score(all_sub_labels, all_sub_preds, average='weighted'):.4f}")

print(f"\n--- Color (12 classes) ---")
print(f"Accuracy: {sum(p==l for p,l in zip(all_color_preds, all_color_labels))/len(all_color_labels):.4f}")
print(f"F1 Macro: {f1_score(all_color_labels, all_color_preds, average='macro'):.4f}")
print(classification_report(all_color_labels, all_color_preds, target_names=color_encoder.classes_))

print(f"\n--- Gender (4 classes) ---")
print(f"Accuracy: {sum(p==l for p,l in zip(all_gen_preds, all_gen_labels))/total:.4f}")
print(f"F1 Macro: {f1_score(all_gen_labels, all_gen_preds, average='macro'):.4f}")
print(classification_report(all_gen_labels, all_gen_preds, target_names=gender_encoder.classes_))
print(f"{'='*55}")
print("\nDone.")