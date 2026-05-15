# https://seaborn.pydata.org/generated/seaborn.heatmap.html

import os
import json
import argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import torch
from training_module import GenreNet

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']


def load_models(models_dir: str):
    models = {}
    sklearn_files = {'SVM': 'svm.pkl', 'Random Forest': 'random_forest.pkl', 'KNN': 'knn.pkl'}
    for name, fname in sklearn_files.items():
        path = os.path.join(models_dir, fname)
        if os.path.exists(path):
            models[name] = joblib.load(path)

    nn_path  = os.path.join(models_dir, "neural_net.pt")
    if os.path.exists(nn_path):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = torch.load(nn_path, map_location=device, weights_only=False)
        model.eval()
        models['Neural Net'] = model

    return models

def predict(model, X_test, model_name: str):
    if model_name == 'Neural Net':
        device = next(model.parameters()).device
        X_te = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(X_te)
        return logits.argmax(dim=1).cpu().numpy()
    return model.predict(X_test)

def plot_confusion_matrix(y_true, y_pred, model_name: str, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    _, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=GENRES, yticklabels=GENRES, ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")

def plot_model_comparison(results: dict, out_path: str):
    names = list(results.keys())
    acc = [results[n]['accuracy'] for n in names]
    f1  = [results[n]['f1'] for n in names]

    x = np.arange(len(names))
    plt.bar(x - 0.2, acc, 0.4, label='Accuracy')
    plt.bar(x + 0.2, f1,  0.4, label='F1 Score')
    plt.xticks(x, names)
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(out_path)
    plt.close()

def plot_nn_training(history_path: str, out_path: str):
    if not os.path.exists(history_path):
        print("  [SKIP] nn_history.json not found, skipping training curves.")
        return
    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history['train_loss'])
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')

    axes[1].plot(history['val_acc'])
    axes[1].set_title('Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate music genre classifiers.")
    parser.add_argument("--data_dir",   type=str, default="./data")
    parser.add_argument("--models_dir", type=str, default="./models")
    parser.add_argument("--out_dir",    type=str, default="./results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_test  = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_test  = np.load(os.path.join(args.data_dir, "y_test.npy"))
    le = joblib.load(os.path.join(args.data_dir, "label_encoder.pkl"))

    models = load_models(args.models_dir)
    print(f"Loaded models: {list(models.keys())}\n")

    results = {}

    for name, model in models.items():
        print(f"Evaluating {name}...")
        preds = predict(model, X_test, name)

        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average='weighted')
        results[name] = {'accuracy': acc, 'f1': f1}

        cm_path = os.path.join(args.out_dir, f"confusion_{name.lower().replace(' ', '_')}.png")
        plot_confusion_matrix(y_test, preds, name, cm_path)
        print(classification_report(y_test, preds, target_names=le.classes_))


    comparison_path = os.path.join(args.out_dir, "model_comparison.png")
    plot_model_comparison(results, comparison_path)

    nn_curves_path  = os.path.join(args.out_dir, "nn_training_curves.png")
    nn_history_path = os.path.join(args.models_dir, "nn_history.json")
    plot_nn_training(nn_history_path, nn_curves_path)