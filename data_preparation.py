import os
import argparse
import numpy as np
import pandas as pd
import librosa
import joblib
import sys
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']


def extract_features(file_path: str, n_mfcc: int = 40) -> np.ndarray:
    try:
        y, sr = librosa.load(file_path, duration=30, mono=True)

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std  = np.std(mfccs,  axis=1)

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std  = np.std(chroma,  axis=1)

        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)
        centroid_std  = np.std(centroid)

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_std  = np.std(rolloff)

        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std  = np.std(zcr)

        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo) 

        features = np.concatenate([
            mfcc_mean, mfcc_std,
            chroma_mean, chroma_std,
            [centroid_mean, centroid_std],
            [rolloff_mean,  rolloff_std],
            [zcr_mean,      zcr_std],
            [tempo],
        ])
        return features

    except Exception as e:
        print(f"  [WARN] Could not process {file_path}: {e}")
        return None


def build_dataset(data_dir: str, n_mfcc: int = 40) -> pd.DataFrame:
    rows = []
    for genre in GENRES:
        genre_path = os.path.join(data_dir, genre)
        if not os.path.isdir(genre_path):
            print(f"[WARN] Directory not found: {genre_path}")
            continue

        files = []
        for f in os.listdir(genre_path):
            if f.endswith('.wav'):
                files.append(f)
        print(f"Processing {genre} ({len(files)} files)...")

        for fname in tqdm(files, desc=f"  {genre}", leave=False):
            fpath = os.path.join(genre_path, fname)
            feats = extract_features(fpath, n_mfcc=n_mfcc)
            if feats is not None:
                row = {"label": genre, "filename": fname}
                for i, v in enumerate(feats):
                    row[f"feat_{i}"] = v
                rows.append(row)

    return pd.DataFrame(rows)


def preprocess(df: pd.DataFrame, out_dir: str, test_size: float = 0.2, random_state: int = 42):
    os.makedirs(out_dir, exist_ok=True)

    exclude_cols = ['filename', 'label']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].values
    y = df["label"].values

    print(f"  Samples: {len(X)}, Features: {X.shape[1]}")

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "y_test.npy"),  y_test)

    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(le,     os.path.join(out_dir, "label_encoder.pkl"))

    print(f"Train set : {X_train.shape}")
    print(f"Test  set : {X_test.shape}")
    print(f"\nAll files saved to: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features and preprocess for model training.")
    parser.add_argument("--data_dir",    type=str,   default="./gtzan")
    parser.add_argument("--out_dir",     type=str,   default="./data")
    parser.add_argument("--n_mfcc",      type=int,   default=40)
    parser.add_argument("--test_size",   type=float, default=0.2)
    parser.add_argument("--random_state",type=int,   default=42)
    args = parser.parse_args()

    print(f"Extracting features from: {args.data_dir}")
    df = build_dataset(args.data_dir, n_mfcc=args.n_mfcc)
    print(f"\nDataset shape: {df.shape}")

    print("\nPreprocessing...")
    preprocess(df, args.out_dir, args.test_size, args.random_state)
