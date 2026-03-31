"""
prediction.py
Classify a single audio file using a trained model.

Usage:
    python prediction.py --file path/to/song.wav
    python prediction.py --file song.wav --model svm        (default)
    python prediction.py --file song.wav --model rf
    python prediction.py --file song.wav --model knn
    python prediction.py --file song.wav --model nn
    python prediction.py --file song.wav --model all        (run all models)
"""

import os
import argparse
import numpy as np
import joblib
import tensorflow as tf

import librosa

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

DATA_DIR   = "./data"
MODELS_DIR = "./models"


def extract_features(file_path: str, n_mfcc: int = 40) -> np.ndarray:
    y, sr = librosa.load(file_path, duration=30, mono=True)

    mfccs    = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    chroma   = librosa.feature.chroma_stft(y=y, sr=sr)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr      = librosa.feature.zero_crossing_rate(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    return np.concatenate([
        np.mean(mfccs, axis=1),  np.std(mfccs, axis=1),
        np.mean(chroma, axis=1), np.std(chroma, axis=1),
        [np.mean(centroid), np.std(centroid)],
        [np.mean(rolloff),  np.std(rolloff)],
        [np.mean(zcr),      np.std(zcr)],
        [float(tempo)]
    ])


def predict_genre(features: np.ndarray, model_key: str) -> dict:
    scaler = joblib.load(os.path.join(DATA_DIR, "scaler.pkl"))
    le     = joblib.load(os.path.join(DATA_DIR, "label_encoder.pkl"))
    X = scaler.transform(features.reshape(1, -1))

    if model_key == 'svm':
        model = joblib.load(os.path.join(MODELS_DIR, "svm.pkl"))
        label = le.inverse_transform(model.predict(X))[0]
        probs = model.predict_proba(X)[0]

    elif model_key == 'rf':
        model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
        label = le.inverse_transform(model.predict(X))[0]
        probs = model.predict_proba(X)[0]

    elif model_key == 'knn':
        model = joblib.load(os.path.join(MODELS_DIR, "knn.pkl"))
        label = le.inverse_transform(model.predict(X))[0]
        probs = model.predict_proba(X)[0]

    elif model_key == 'nn':
        model = tf.keras.models.load_model(os.path.join(MODELS_DIR, "neural_net.keras"))
        probs = model.predict(X, verbose=0)[0]
        label = le.inverse_transform([np.argmax(probs)])[0]

    else:
        raise ValueError(f"Unknown model key: {model_key}")

    classes = le.classes_
    top3 = sorted(zip(classes, probs), key=lambda x: -x[1])[:3]
    return {'prediction': label, 'confidence': float(max(probs)), 'top3': top3}


def main():
    parser = argparse.ArgumentParser(description="Predict genre of an audio file.")
    parser.add_argument("--file",  type=str, required=True, help="Path to .wav file")
    parser.add_argument("--model", type=str, default="svm",
                        choices=["svm", "rf", "knn", "nn", "all"],
                        help="Which model to use (default: svm)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: file not found — {args.file}")
        return

    print(f"Extracting features from: {args.file}")
    features = extract_features(args.file)

    model_keys = ["svm", "rf", "knn", "nn"] if args.model == "all" else [args.model]
    model_names = {"svm": "SVM", "rf": "Random Forest", "knn": "KNN", "nn": "Neural Net"}

    print()
    for key in model_keys:
        try:
            result = predict_genre(features, key)
            print(f"── {model_names[key]} ──────────────────────")
            print(f"  Prediction : {result['prediction'].upper()}")
            print(f"  Confidence : {result['confidence']*100:.1f}%")
            print(f"  Top 3:")
            for genre, prob in result['top3']:
                bar = '█' * int(prob * 20)
                print(f"    {genre:<12} {prob*100:5.1f}%  {bar}")
            print()
        except Exception as e:
            print(f"  [{key}] Error: {e}\n")


if __name__ == "__main__":
    main()