"""
train_gender_lr.py
------------------
Trains a Logistic Regression model to predict gender from category features.
Gender is almost entirely determined by category — no need for a neural network.

Usage:
    python train_gender_lr.py \
        --data_dir /workspace/Data/train \
        --output_dir /workspace/outputs

Outputs:
    gender_lr_model.pkl     — trained sklearn model
    gender_label_encoder.pkl — LabelEncoder for gender strings
    gender_feature_cols.pkl  — list of feature column names (for inference)
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=str, default="./Data/train")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--max_iter",   type=int, default=1000)
    parser.add_argument("--C",          type=float, default=1.0,
                        help="Regularization strength (smaller = stronger regularization)")
    return parser.parse_args()


def load_data(data_dir):
    parquet_path = os.path.join(data_dir, "train.parquet")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows from {parquet_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nGender distribution:")
    print(df["gender"].value_counts())
    return df


def build_features(df):
    """
    Features:
      - main_category_id  (level_2_id maps subcategory to 21 main categories)
      - category_id       (190 subcategories — the strongest signal)
      - color_label       (weak but nonzero signal, e.g. pink → FEMALE)

    We one-hot encode all three. category_id alone is almost sufficient,
    but adding color gives a small lift on ambiguous cases.
    """
    feature_cols = []

    # Core category features
    if "level_2_id" in df.columns:
        df["main_cat"] = df["level_2_id"].astype(str)
        feature_cols.append("main_cat")

    df["sub_cat"] = df["category_id"].astype(str)
    feature_cols.append("sub_cat")

    # Color — fill missing with 'unknown'
    if "color_label" in df.columns:
        df["color_feat"] = df["color_label"].fillna("unknown").astype(str)
        feature_cols.append("color_feat")

    print(f"\nUsing features: {feature_cols}")
    return df, feature_cols


def build_pipeline(feature_cols, C, max_iter):
    """
    Pipeline:
      OneHotEncoder (handle_unknown='ignore') → LogisticRegression
      class_weight='balanced' handles the 92/7/1 UNISEX/FEMALE/MALE imbalance
    """
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    ct = ColumnTransformer([("ohe", ohe, feature_cols)], remainder="drop")

    lr = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight="balanced",   # critical — without this it predicts UNISEX for everything
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )

    pipeline = Pipeline([("features", ct), ("clf", lr)])
    return pipeline


def evaluate(pipeline, X, y_encoded, label_encoder, cv=5):
    """5-fold stratified cross-validation to get honest per-class metrics."""
    print(f"\nRunning {cv}-fold stratified cross-validation...")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipeline, X, y_encoded, cv=skf, n_jobs=-1)

    target_names = label_encoder.classes_
    print("\n" + "="*55)
    print("CROSS-VALIDATED CLASSIFICATION REPORT")
    print("="*55)
    print(classification_report(y_encoded, y_pred, target_names=target_names))

    f1_macro = f1_score(y_encoded, y_pred, average="macro")
    f1_weighted = f1_score(y_encoded, y_pred, average="weighted")
    print(f"F1 Macro   : {f1_macro:.4f}")
    print(f"F1 Weighted: {f1_weighted:.4f}")

    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_encoded, y_pred),
        index=[f"TRUE_{c}" for c in target_names],
        columns=[f"PRED_{c}" for c in target_names]
    ))

    return f1_macro


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    df = load_data(args.data_dir)

    # Drop rows with missing gender
    df = df.dropna(subset=["gender"])
    print(f"\nAfter dropping missing gender: {len(df)} rows")

    # ── 2. Encode target ──────────────────────────────────────────────────────
    le = LabelEncoder()
    y = le.fit_transform(df["gender"])
    print(f"\nGender classes: {list(le.classes_)}")
    print(f"Encoded as:     {list(range(len(le.classes_)))}")

    # ── 3. Build features ─────────────────────────────────────────────────────
    df, feature_cols = build_features(df)
    X = df[feature_cols]

    # ── 4. Cross-validate ─────────────────────────────────────────────────────
    pipeline = build_pipeline(feature_cols, args.C, args.max_iter)
    f1_macro = evaluate(pipeline, X, y, le)

    # ── 5. Fit final model on all data ────────────────────────────────────────
    print("\nFitting final model on full training set...")
    pipeline.fit(X, y)
    print("Done.")

    # ── 6. Save artifacts ─────────────────────────────────────────────────────
    model_path   = os.path.join(args.output_dir, "gender_lr_model.pkl")
    encoder_path = os.path.join(args.output_dir, "gender_label_encoder.pkl")
    cols_path    = os.path.join(args.output_dir, "gender_feature_cols.pkl")

    with open(model_path,   "wb") as f: pickle.dump(pipeline, f)
    with open(encoder_path, "wb") as f: pickle.dump(le, f)
    with open(cols_path,    "wb") as f: pickle.dump(feature_cols, f)

    print(f"\nSaved model      → {model_path}")
    print(f"Saved encoder    → {encoder_path}")
    print(f"Saved feat cols  → {cols_path}")
    print(f"\nFinal CV F1 Macro: {f1_macro:.4f}")

    # ── 7. Show top category → gender mappings (sanity check) ─────────────────
    print("\n── Sanity check: top category→gender predictions ──")
    sample_cats = df[["sub_cat", "gender"]].groupby("sub_cat")["gender"] \
                    .agg(lambda x: x.value_counts().index[0]).reset_index()
    sample_cats.columns = ["category_id", "most_common_gender"]

    # Predict on unique categories
    unique_cats = df[feature_cols].drop_duplicates().head(20)
    preds = le.inverse_transform(pipeline.predict(unique_cats))
    unique_cats = unique_cats.copy()
    unique_cats["predicted_gender"] = preds
    print(unique_cats.to_string(index=False))


if __name__ == "__main__":
    main()