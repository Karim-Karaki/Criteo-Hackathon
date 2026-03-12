import pandas as pd
import matplotlib.pyplot as plt
import json

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = r"C:\Users\karim\Desktop\Data Science Projects\Criteo Image Classification\Data"
PARQUET_PATH  = BASE_DIR + r"\train\train.parquet"
TAXONOMY_PATH = BASE_DIR + r"\level2_categories.json"

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_parquet(PARQUET_PATH)

with open(TAXONOMY_PATH, "r") as f:
    mapping = json.load(f)

taxonomy = pd.DataFrame(mapping).rename(columns={"new_id": "category_id"})
df = df.merge(taxonomy[["category_id", "category_name"]], on="category_id", how="left")
df[["category", "subcategory"]] = df["category_name"].str.split(" > ", expand=True)
df.loc[df["category_id"] == 44, "category_id"] = 28

# ── Stats ─────────────────────────────────────────────────────────────────────
sub_counts  = df["category_id"].value_counts().sort_index()
main_counts = df["category"].value_counts().sort_index()

print(f"Subcategory (190 classes):")
print(f"  Min:  {sub_counts.min()}")
print(f"  Max:  {sub_counts.max()}")
print(f"  Mean: {sub_counts.mean():.1f}")
print(f"  Std:  {sub_counts.std():.1f}")
print(f"\nBottom 10 rarest:")
print(sub_counts.sort_values().head(10))
print(f"\nTop 10 most common:")
print(sub_counts.sort_values(ascending=False).head(10))

print(f"\nMain category (21 classes):")
print(f"  Min:  {main_counts.min()}")
print(f"  Max:  {main_counts.max()}")
print(f"  Mean: {main_counts.mean():.1f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(20, 10))

axes[0].bar(range(len(sub_counts)), sub_counts.sort_values(ascending=False).values, color="#4C72B0")
axes[0].axhline(sub_counts.mean(), color="red", linestyle="--", label=f"Mean: {sub_counts.mean():.0f}")
axes[0].set_title("Subcategory Distribution (190 classes)")
axes[0].set_xlabel("Class rank")
axes[0].set_ylabel("Sample count")
axes[0].legend()

axes[1].bar(range(len(main_counts)), main_counts.sort_values(ascending=False).values, color="#55A868")
axes[1].axhline(main_counts.mean(), color="red", linestyle="--", label=f"Mean: {main_counts.mean():.0f}")
axes[1].set_title("Main Category Distribution (21 classes)")
axes[1].set_xlabel("Class rank")
axes[1].set_ylabel("Sample count")
axes[1].legend()

plt.tight_layout()
plt.savefig("class_imbalance.png", dpi=120)
plt.show()
print("Saved to class_imbalance.png")