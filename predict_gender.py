"""
predict_gender.py — Gender prediction using main category correlation
=====================================================================
Uses the already-predicted main category to assign gender.
Run after training CLIP to build the gender lookup table.

Usage:
    python predict_gender.py --data_dir /workspace/Data/train --output_dir /workspace/outputs
"""

import os
import argparse
import pickle
import pandas as pd
import json

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",   type=str, default="/workspace/Data/train")
parser.add_argument("--output_dir", type=str, default="/workspace/outputs")
parser.add_argument("--threshold",  type=float, default=0.5)  # dominant gender threshold
args = parser.parse_args()

# ── Load Data ─────────────────────────────────────────────────────────────────
PARQUET_PATH  = os.path.join(args.data_dir, "train.parquet")
TAXONOMY_PATH = os.path.join(args.data_dir, "level2_categories.json")

df = pd.read_parquet(PARQUET_PATH)

# Add discovery if available
discovery_path = "/workspace/Data/discovery/discovery.parquet"
if os.path.exists(discovery_path):
    disc_df = pd.read_parquet(discovery_path)
    df = pd.concat([df, disc_df]).reset_index(drop=True)

with open(TAXONOMY_PATH, "r") as f:
    mapping = json.load(f)

taxonomy = pd.DataFrame(mapping).rename(columns={"new_id": "category_id"})
df = df.merge(taxonomy[["category_id", "category_name"]], on="category_id", how="left")
df[["category", "subcategory"]] = df["category_name"].str.split(" > ", expand=True)
df.loc[df["category_id"] == 44, "category_id"] = 28
df["gender"] = df["gender"].fillna("UNISEX")

print(f"Loaded {len(df)} samples")
print(f"\nGender distribution overall:")
print(df["gender"].value_counts())

# ── Build Main Category → Gender Lookup ───────────────────────────────────────
print(f"\n--- Building gender lookup table (threshold={args.threshold}) ---")

# Also build subcategory level for finer granularity
main_gender_lookup = {}
sub_gender_lookup  = {}

# Main category level
for main_cat, group in df.groupby("category"):
    gender_counts = group["gender"].value_counts()
    total         = len(group)
    dominant      = gender_counts.index[0]
    dominant_pct  = gender_counts.iloc[0] / total

    if dominant_pct >= args.threshold:
        main_gender_lookup[main_cat] = dominant
    else:
        main_gender_lookup[main_cat] = "UNISEX"

    print(f"  {main_cat:<45} → {main_gender_lookup[main_cat]:<10} ({dominant_pct:.0%} {dominant})")

# Subcategory level for finer lookup
for cat_id, group in df.groupby("category_id"):
    gender_counts = group["gender"].value_counts()
    total         = len(group)
    dominant      = gender_counts.index[0]
    dominant_pct  = gender_counts.iloc[0] / total

    if dominant_pct >= args.threshold:
        sub_gender_lookup[int(cat_id)] = dominant
    else:
        sub_gender_lookup[int(cat_id)] = "UNISEX"

# ── Evaluate on training data ─────────────────────────────────────────────────
print(f"\n--- Evaluating gender lookup accuracy ---")

# Using main category lookup
df["predicted_gender_main"] = df["category"].map(main_gender_lookup).fillna("UNISEX")
main_acc = (df["predicted_gender_main"] == df["gender"]).mean()
print(f"Main category lookup accuracy: {main_acc:.4f}")

# Using subcategory lookup
df["predicted_gender_sub"] = df["category_id"].map(sub_gender_lookup).fillna("UNISEX")
sub_acc = (df["predicted_gender_sub"] == df["gender"]).mean()
print(f"Subcategory lookup accuracy:   {sub_acc:.4f}")

# ── Save lookup tables ─────────────────────────────────────────────────────────
main_lookup_path = os.path.join(args.output_dir, "main_gender_lookup.json")
sub_lookup_path  = os.path.join(args.output_dir, "sub_gender_lookup.json")

with open(main_lookup_path, "w") as f:
    json.dump(main_gender_lookup, f, indent=2)

with open(sub_lookup_path, "w") as f:
    json.dump(sub_gender_lookup, f, indent=2)

print(f"\nSaved main category lookup → {main_lookup_path}")
print(f"Saved subcategory lookup   → {sub_lookup_path}")

# ── Show the full lookup table ────────────────────────────────────────────────
print(f"\n--- Final Gender Lookup Table (Main Category) ---")
for cat, gender in sorted(main_gender_lookup.items()):
    print(f"  {cat:<45} → {gender}")

print(f"\n--- Summary ---")
print(f"Main category lookup accuracy: {main_acc:.4f}")
print(f"Subcategory lookup accuracy:   {sub_acc:.4f}")
print(f"\nRecommendation: Use {'subcategory' if sub_acc > main_acc else 'main category'} lookup for best accuracy")
print("\nDone.")