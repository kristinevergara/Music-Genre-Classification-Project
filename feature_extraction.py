"""
feature_extraction.py
Loops over the GTZAN dataset folder structure, extracts audio features
using librosa, and saves them to features.csv.

Expected folder layout:
    gtzan/
        blues/      (100 x .wav files)
        classical/
        country/
        disco/
        hiphop/
        jazz/
        metal/
        pop/
        reggae/
        rock/

Usage:
    python feature_extraction.py --data_dir ./gtzan --out features.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']


def extract_features(file_path: str, n_mfcc: int = 40) -> np.ndarray:
    """
    Extract a fixed-length feature vector from a .wav file.

    Features extracted:
        - MFCCs: mean + std of 40 coefficients  → 80 values
        - Chroma STFT: mean + std of 12 bins     → 24 values
        - Spectral centroid: mean + std           →  2 values
        - Spectral roll-off: mean + std           →  2 values
        - Zero crossing rate: mean + std          →  2 values
        - Tempo                                   →  1 value
    Total: 111 features
    """
    try:
        y, sr = librosa.load(file_path, duration=30, mono=True)

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std  = np.std(mfccs,  axis=1)

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std  = np.std(chroma,  axis=1)

        # Spectral centroid
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        centroid_mean = np.mean(centroid)
        centroid_std  = np.std(centroid)

        # Spectral roll-off
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_std  = np.std(rolloff)

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std  = np.std(zcr)

        # Tempo
        tempo_raw, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.atleast_1d(tempo_raw)[0])

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

        files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
        print(f"Processing {genre} ({len(files)} files)...")

        for fname in tqdm(files, desc=f"  {genre}", leave=False):
            fpath = os.path.join(genre_path, fname)
            feats = extract_features(fpath, n_mfcc=n_mfcc)
            if feats is not None:
                rows.append({**{f"feat_{i}": v for i, v in enumerate(feats)},
                             "label": genre, "filename": fname})

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Extract audio features from GTZAN dataset.")
    parser.add_argument("--data_dir", type=str, default="./gtzan",
                        help="Path to GTZAN root folder (default: ./gtzan)")
    parser.add_argument("--out",      type=str, default="features.csv",
                        help="Output CSV path (default: features.csv)")
    parser.add_argument("--n_mfcc",   type=int, default=40,
                        help="Number of MFCC coefficients (default: 40)")
    args = parser.parse_args()

    print(f"Extracting features from: {args.data_dir}")
    df = build_dataset(args.data_dir, n_mfcc=args.n_mfcc)

    print(f"\nDataset shape: {df.shape}")
    print(f"Samples per genre:\n{df['label'].value_counts()}")

    df.to_csv(args.out, index=False)
    print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()