"""
training module.py
Trains SVM, Random Forest, KNN, and a Feedforward Neural Network
on the preprocessed feature splits. Saves all trained models to ./models/.

Usage:
    python training module.py --data_dir ./data --models_dir ./models
"""

import os
import json
import argparse
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] PyTorch not available ({e}). Skipping NN model.")
except Exception as e:
    TORCH_AVAILABLE = False
    print(f"[ERROR] Error importing PyTorch: {e}")


# ── Sklearn models ──────────────────────────────────────────────────────────

def train_svm(X_train, y_train):
    print("Training SVM ...")
    model = SVC(kernel='rbf', C=100, gamma='auto', probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    print("Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_knn(X_train, y_train):
    print("Training KNN (k=5)...")
    model = KNeighborsClassifier(n_neighbors=5, metric='cosine', n_jobs=-1)
    model.fit(X_train, y_train)
    return model


# ── Neural Network ──────────────────────────────────────────────────────────
class GenreNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_nn(X_train, y_train, X_test, y_test, models_dir: str):
    print("Training Neural Network...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"  Using device: {device}")

    num_classes = len(np.unique(y_train))

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.long)
    X_te = torch.tensor(X_test,  dtype=torch.float32)
    y_te = torch.tensor(y_test,  dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True
    )

    model = GenreNet(X_train.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=7, min_lr=1e-5
    )

    best_val_acc = 0.0
    patience_counter = 0
    patience = 15
    history = {"train_loss": [], "val_acc": []}

    for epoch in range(100):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_train)

        model.eval()
        with torch.no_grad():
            logits = model(X_te.to(device))
            preds = logits.argmax(dim=1).cpu().numpy()
        val_acc = accuracy_score(y_test, preds)

        scheduler.step(epoch_loss)
        history["train_loss"].append(epoch_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d} | loss {epoch_loss:.4f} | val_acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(models_dir, "neural_net_best.pt"))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(os.path.join(models_dir, "neural_net_best.pt")))
    model.eval()
    with torch.no_grad():
        preds = model(X_te.to(device)).argmax(dim=1).cpu().numpy()
    nn_acc = accuracy_score(y_test, preds)

    torch.save(model, os.path.join(models_dir, "neural_net.pt"))
    with open(os.path.join(models_dir, "nn_history.json"), "w") as f:
        json.dump(history, f)

    print(f"  Neural Net test accuracy: {nn_acc:.4f}\n")
    return nn_acc

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train music genre classifiers.")
    parser.add_argument("--data_dir",   type=str, default="./data")
    parser.add_argument("--models_dir", type=str, default="./models")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    print("Loading data...")
    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
    X_test  = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(args.data_dir, "y_test.npy"))
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}\n")

    results = {}

    # SVM
    svm = train_svm(X_train, y_train)
    results['SVM'] = accuracy_score(y_test, svm.predict(X_test))
    joblib.dump(svm, os.path.join(args.models_dir, "svm.pkl"))
    print(f"  SVM test accuracy: {results['SVM']:.4f}\n")

    # Random Forest
    rf = train_random_forest(X_train, y_train)
    results['Random Forest'] = accuracy_score(y_test, rf.predict(X_test))
    joblib.dump(rf, os.path.join(args.models_dir, "random_forest.pkl"))
    print(f"  Random Forest test accuracy: {results['Random Forest']:.4f}\n")

    # KNN
    knn = train_knn(X_train, y_train)
    results['KNN'] = accuracy_score(y_test, knn.predict(X_test))
    joblib.dump(knn, os.path.join(args.models_dir, "knn.pkl"))
    print(f"  KNN test accuracy: {results['KNN']:.4f}\n")

    # Neural Network
    if TORCH_AVAILABLE:
        results['Neural Net'] = train_nn(X_train, y_train, X_test, y_test, args.models_dir)
    else:
        print("[SKIP] Neural Network — PyTorch unavailable.\n")

    # Summary
    print("=" * 40)
    print("  Model accuracy summary")
    print("=" * 40)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<20} {acc:.4f}")
    if not TORCH_AVAILABLE:
        print(f"  {'Neural Net':<20} skipped (no PyTorch)")
    print("=" * 40)
    print(f"\nAll models saved to: {args.models_dir}/")


if __name__ == "__main__":
    main()
