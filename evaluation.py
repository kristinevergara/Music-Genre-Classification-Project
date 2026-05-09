"""
evaluation.py
Loads all trained models, runs predictions on the test set,
and produces:
  - Per-model confusion matrix heatmaps  (confusion_<model>.png)
  - Accuracy + F1 comparison bar chart   (model_comparison.png)
  - Neural network training curves       (nn_training_curves.png)
  - Full classification report in console

Usage:
    python evaluation.py --data_dir ./data --models_dir ./models --out_dir ./results
"""

# https://seaborn.pydata.org/generated/seaborn.heatmap.html

import os
import json
import argparse
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

try:
    import torch
    from training_module import GenreNet
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

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
    if os.path.exists(nn_path) and TORCH_AVAILABLE:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = torch.load(nn_path, map_location=device, weights_only=False)
        model.eval()
        models['Neural Net'] = (model, device)

    return models

def predict(model, X_test, model_name: str):
    if model_name == 'Neural Net':
        net, device = model
        X_te = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = net(X_te)
        return logits.argmax(dim=1).cpu().numpy()
    return model.predict(X_test)


# Plots 

def plot_confusion_matrix(y_true, y_pred, model_name: str, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    _, ax = plt.subplots()
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=GENRES,yticklabels=GENRES,ax=ax,linewidths=0.5)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=14, pad=16)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual',    fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

def plot_model_comparison(results: dict, out_path: str):
    names = list(results.keys())
    acc = [results[n]['accuracy'] for n in names]
    f1  = [results[n]['f1'] for n in names]

    x = np.arange(len(names))
    _, ax = plt.subplots()
    ax.bar(x - 0.2, acc, 0.4, label='Accuracy')
    ax.bar(x + 0.2, f1,  0.4, label='F1 Score')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")

def plot_nn_training(history_path: str, out_path: str):
    if not os.path.exists(history_path):
        print("  [SKIP] nn_history.json not found, skipping training curves.")
        return
    with open(history_path) as f:
        history = json.load(f)

    _, axes = plt.subplots(1,2)
    axes[0].plot(history['val_acc'])
    axes[0].set_title('Accuracy')
    axes[1].plot(history['train_loss'])
    axes[1].set_title('Loss')
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate music genre classifiers.")
    parser.add_argument("--data_dir",   type=str, default="./data")
    parser.add_argument("--models_dir", type=str, default="./models")
    parser.add_argument("--out_dir",    type=str, default="./results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load data
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

        # Confusion matrix
        cm_path = os.path.join(args.out_dir, f"confusion_{name.lower().replace(' ', '_')}.png")
        plot_confusion_matrix(y_test, preds, name, cm_path)

        # Classification report
        print(classification_report(y_test, preds, target_names=le.classes_))


    # Comparison chart
    comparison_path = os.path.join(args.out_dir, "model_comparison.png")
    plot_model_comparison(results, comparison_path)

    # Neural network training curves
    nn_curves_path  = os.path.join(args.out_dir, "nn_training_curves.png")
    nn_history_path = os.path.join(args.models_dir, "nn_history.json")
    plot_nn_training(nn_history_path, nn_curves_path)