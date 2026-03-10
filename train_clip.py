"""
train_clip.py — Hierarchical Classification with CLIP backbone
==============================================================
CLIP was pretrained on 400M image-text pairs including product images.
Features transfer much better to e-commerce classification than ImageNet.

Vast.ai usage:
    python train_clip.py --data_dir /workspace/Data/train --output_dir /workspace/outputs
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ── Argument Parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",      type=str, default="/workspace/Data/train")
parser.add_argument("--output_dir",    type=str, default="/workspace/outputs")
parser.add_argument("--batch_size",    type=int, default=64)
parser.add_argument("--num_workers",   type=int, default=4)
parser.add_argument("--phase1_epochs", type=int, default=8)
parser.add_argument("--phase2_epochs", type=int, default=20)
parser.add_argument("--main_weight",   type=float, default=0.3)
parser.add_argument("--sub_weight",    type=float, default=0.7)
parser.add_argument("--patience",      type=int,   default=5)
parser.add_argument("--min_delta",     type=float, default=0.001)
args = parser.parse_args()

os.environ["DATA_DIR"]    = args.data_dir
os.environ["OUTPUT_DIR"]  = args.output_dir
os.environ["BATCH_SIZE"]  = str(args.batch_size)
os.environ["NUM_WORKERS"] = str(args.num_workers)
os.environ["IMG_SIZE"]    = "224"

from dataset import (
    train_df, test_df, IMAGE_DIR,
    NUM_CLASSES, NUM_MAIN,
)

os.makedirs(args.output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── CLIP Dataset (uses CLIP processor instead of torchvision transforms) ──────
print("\n--- Loading CLIP processor ---")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class CLIPProductDataset(Dataset):
    def __init__(self, df, image_dir, processor, augment=False):
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.processor = processor
        self.augment   = augment

        if augment:
            from torchvision import transforms
            self.aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            ])
        else:
            self.aug = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, str(row["category_id"]), f"{row['anonymous_id']}.jpg")
        image    = Image.open(img_path).convert("RGB")

        if self.aug:
            image = self.aug(image)

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        return (
            pixel_values,
            int(row["main_encoded"]),
            int(row["category_encoded"])
        )

train_dataset = CLIPProductDataset(train_df, IMAGE_DIR, clip_processor, augment=True)
test_dataset  = CLIPProductDataset(test_df,  IMAGE_DIR, clip_processor, augment=False)

# WeightedRandomSampler
class_counts   = train_df["category_encoded"].value_counts().sort_index()
weights        = 1.0 / class_counts
sample_weights = train_df["category_encoded"].map(weights).values.copy()
sampler        = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False,   num_workers=args.num_workers, pin_memory=True)

images, main_labels, sub_labels = next(iter(train_loader))
print(f"Batch shape:    {images.shape}")
print(f"Main range:     {main_labels.min()}-{main_labels.max()}")
print(f"Sub range:      {sub_labels.min()}-{sub_labels.max()}")

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
        else:
            self.counter += 1
            print(f"  Early stopping counter: {self.counter}/{self.patience}")
            return self.counter >= self.patience

# ── Class Weighted Loss ───────────────────────────────────────────────────────
sub_counts   = train_df["category_encoded"].value_counts().sort_index()
sub_weights  = 1.0 / torch.tensor(sub_counts.values, dtype=torch.float)
sub_weights  = (sub_weights / sub_weights.sum() * NUM_CLASSES).to(device)

main_counts  = train_df["main_encoded"].value_counts().sort_index()
main_weights = 1.0 / torch.tensor(main_counts.values, dtype=torch.float)
main_weights = (main_weights / main_weights.sum() * NUM_MAIN).to(device)

main_criterion = nn.CrossEntropyLoss(weight=main_weights, label_smoothing=0.1)
sub_criterion  = nn.CrossEntropyLoss(weight=sub_weights,  label_smoothing=0.1)

# ── CLIP Hierarchical Model ───────────────────────────────────────────────────
print("\n--- Loading CLIP backbone ---")
clip_model  = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
feature_dim = 768  # CLIP ViT-B/32 visual feature dim

class CLIPHierarchicalModel(nn.Module):
    def __init__(self, clip_model, feature_dim, num_main, num_sub):
        super().__init__()
        self.backbone = clip_model.vision_model  # just the vision encoder
        self.dropout  = nn.Dropout(0.3)

        self.main_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_main)
        )

        self.sub_head = nn.Sequential(
            nn.Linear(feature_dim + num_main, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_sub)
        )

    def forward(self, pixel_values):
        outputs     = self.backbone(pixel_values=pixel_values)
        features    = self.dropout(outputs.pooler_output)  # [B, 768]
        main_logits = self.main_head(features)
        sub_input   = torch.cat([features, main_logits], dim=1)
        sub_logits  = self.sub_head(sub_input)
        return main_logits, sub_logits

model     = CLIPHierarchicalModel(clip_model, feature_dim, NUM_MAIN, NUM_CLASSES).to(device)
ckpt_path = os.path.join(args.output_dir, "clip_hierarchical_best.pth")
print(f"Model built — Feature dim: {feature_dim} | Main: {NUM_MAIN} | Sub: {NUM_CLASSES}")

# ── Training Function ─────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, main_correct, sub_correct, total = 0, 0, 0, 0

    for pixel_values, main_labels, sub_labels in loader:
        pixel_values = pixel_values.to(device)
        main_labels  = main_labels.to(device)
        sub_labels   = sub_labels.to(device)

        optimizer.zero_grad()
        main_logits, sub_logits = model(pixel_values)

        main_loss = main_criterion(main_logits, main_labels)
        sub_loss  = sub_criterion(sub_logits,  sub_labels)
        loss      = args.main_weight * main_loss + args.sub_weight * sub_loss

        loss.backward()
        optimizer.step()

        total_loss   += loss.item()
        main_correct += (main_logits.argmax(1) == main_labels).sum().item()
        sub_correct  += (sub_logits.argmax(1)  == sub_labels).sum().item()
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

scheduler      = CosineAnnealingLR(optimizer, T_max=args.phase2_epochs)
best_sub_acc   = 0
early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)

for epoch in range(args.phase2_epochs):
    loss, main_acc, sub_acc = train_epoch(model, train_loader, optimizer)
    scheduler.step()
    print(f"Epoch {epoch+1}/{args.phase2_epochs} | Loss: {loss:.4f} | Main Acc: {main_acc:.4f} | Sub Acc: {sub_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

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

main_correct, sub_correct       = 0, 0
total                           = len(test_loader.dataset)
all_sub_preds,  all_sub_labels  = [], []
all_main_preds, all_main_labels = [], []

with torch.no_grad():
    for pixel_values, main_labels, sub_labels in test_loader:
        pixel_values = pixel_values.to(device)
        main_labels  = main_labels.to(device)
        sub_labels   = sub_labels.to(device)

        main_logits, sub_logits = model(pixel_values)
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