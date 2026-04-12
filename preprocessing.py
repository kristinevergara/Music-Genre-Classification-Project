"""
preprocessing.py

Loads features.csv, encodes labels, normalizes features,
and saves train/test splits to .npy files for model training.

Usage:
    python preprocessing.py --features features.csv --out_dir ./data
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib


def preprocess(features_csv: str, out_dir: str, test_size: float = 0.2, random_state: int = 42):
    os.makedirs(out_dir, exist_ok=True)

    # Load
    print(f"Loading {features_csv}...")
    df = pd.read_csv(features_csv)

    feature_cols = [c for c in df.columns if c.startswith("feat_")]
    X = df[feature_cols].values
    y_raw = df["label"].values

    print(f"  Total samples : {len(X)}")
    print(f"  Feature count : {X.shape[1]}")
    print(f"  Class distribution:\n{pd.Series(y_raw).value_counts().to_string()}\n")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"  Classes (encoded): {list(le.classes_)}\n")

    # Train / test split (stratified so every genre is balanced)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Normalize — fit ONLY on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Save splits
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "y_test.npy"),  y_test)

    # Save scaler and encoder for inference later
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(le,     os.path.join(out_dir, "label_encoder.pkl"))

    print(f"Train set : {X_train.shape}")
    print(f"Test  set : {X_test.shape}")
    print(f"\nAll files saved to: {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Preprocess features for model training.")
    parser.add_argument("--features",    type=str,   default="features.csv")
    parser.add_argument("--out_dir",     type=str,   default="./data")
    parser.add_argument("--test_size",   type=float, default=0.2)
    parser.add_argument("--random_state",type=int,   default=42)
    args = parser.parse_args()

    preprocess(args.features, args.out_dir, args.test_size, args.random_state)


if __name__ == "__main__":
    main()
