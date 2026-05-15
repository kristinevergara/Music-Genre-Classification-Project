import os
import argparse
import numpy as np
import joblib
import sys
from data_preparation import extract_features

try:
    import torch
    from training_module import GenreNet
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

DATA_DIR   = "./data"
MODELS_DIR = "./models"


def predict_genre(features: np.ndarray, model_key: str) -> dict:
    scaler = joblib.load(os.path.join(DATA_DIR, "scaler.pkl"))
    le     = joblib.load(os.path.join(DATA_DIR, "label_encoder.pkl"))
    X = scaler.transform(features.reshape(1, -1))

    sklearn_files = {'svm': 'svm.pkl', 'rf':  'random_forest.pkl', 'knn': 'knn.pkl'}

    if model_key in sklearn_files:
        model = joblib.load(os.path.join(MODELS_DIR, sklearn_files[model_key]))
        label = le.inverse_transform(model.predict(X))[0]
        probs = model.predict_proba(X)[0]
    elif model_key == 'nn':
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available.")
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = torch.load(os.path.join(MODELS_DIR, "neural_net.pt"),
                           map_location=device, weights_only=False)
        model.eval()
        X_te = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(X_te)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        label = le.inverse_transform([np.argmax(probs)])[0]

    else:
        raise ValueError(f"Unknown model key: {model_key}")

    classes = le.classes_
    top3 = sorted(zip(classes, probs), key=lambda x: x[1], reverse=True)[:3]
    return {'prediction': label, 'confidence': float(max(probs)), 'top3': top3}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict genre of an audio file.")
    parser.add_argument("--file",  type=str, required=True, help="Path to .wav file")
    parser.add_argument("--model", type=str, default="svm",
                        choices=["svm", "rf", "knn", "nn", "all"],
                        help="Which model to use (default: svm)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: file not found — {args.file}")
        sys.exit(1)

    print(f"Extracting features from: {args.file}")
    features = extract_features(args.file)

    choices = {"svm": "SVM", "rf": "Random Forest", "knn": "KNN", "nn": "Neural Net"}
    keys = list(choices.keys()) if args.model == "all" else [args.model]

    print()
    for key in keys:
        try:
            result = predict_genre(features, key)
            print(f"{choices[key]}")
            print(f"  Prediction : {result['prediction'].upper()}")
            print(f"  Confidence : {result['confidence']*100:.1f}%")
            print(f"  Top 3:")
            for genre, prob in result['top3']:
                print(f"    {genre:<12} {prob*100:5.1f}%")
            print()
        except FileNotFoundError:
            print(f"  [{key}] Model file not found, skipping.")
        except Exception as e:
            raise