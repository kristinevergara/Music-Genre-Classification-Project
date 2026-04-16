"""
training module.py
Trains SVM, Random Forest, KNN, and a Feedforward Neural Network
on the preprocessed feature splits. Saves all trained models to ./models/.

Usage:
    python training module.py --data_dir ./data --models_dir ./models
"""

import os
import argparse
import numpy as np
import joblib

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from training_module_fixed import train_nn

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TF_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] TensorFlow not available ({e}). Skipping NN model.")
    TF_AVAILABLE = False


# ── Sklearn models ──────────────────────────────────────────────────────────

def train_svm(X_train, y_train):
    print("Training SVM (RBF kernel)...")
    model = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
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
    model = KNeighborsClassifier(n_neighbors=5, metric='euclidean', n_jobs=-1)
    model.fit(X_train, y_train)
    return model


# ── Neural Network ──────────────────────────────────────────────────────────

    def build_nn(input_dim: int, num_classes: int) -> 'tf.keras.Model':
        model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        Dropout(0.2),

        Dense(num_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


    def train_nn(X_train, y_train, X_test, y_test):
        print("Training Neural Network...")
        num_classes = len(np.unique(y_train))
        model = build_nn(X_train.shape[1], num_classes)

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
            ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-5, monitor='val_loss')
        ]

        history = model.fit(
            X_train, y_train,
            epochs=200,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        return model, history


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train music genre classifiers.")
    parser.add_argument("--data_dir",   type=str, default="./data")
    parser.add_argument("--models_dir", type=str, default="./models")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    # Load preprocessed data
    print("Loading data...")
    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
    X_test  = np.load(os.path.join(args.data_dir, "X_test.npy"))
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(args.data_dir, "y_test.npy"))
    print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}\n")

    results = {}

    # SVM
    svm = train_svm(X_train, y_train)
    svm_acc = accuracy_score(y_test, svm.predict(X_test))
    results['SVM'] = svm_acc
    joblib.dump(svm, os.path.join(args.models_dir, "svm.pkl"))
    print(f"  SVM test accuracy: {svm_acc:.4f}\n")

    # Random Forest
    rf = train_random_forest(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    results['Random Forest'] = rf_acc
    joblib.dump(rf, os.path.join(args.models_dir, "random_forest.pkl"))
    print(f"  Random Forest test accuracy: {rf_acc:.4f}\n")

    # KNN
    knn = train_knn(X_train, y_train)
    knn_acc = accuracy_score(y_test, knn.predict(X_test))
    results['KNN'] = knn_acc
    joblib.dump(knn, os.path.join(args.models_dir, "knn.pkl"))
    print(f"  KNN test accuracy: {knn_acc:.4f}\n")


    if TF_AVAILABLE:
        nn, history = train_nn(X_train, y_train, X_test, y_test)
        nn_acc = accuracy_score(y_test, np.argmax(nn.predict(X_test), axis=1))
        results['Neural Net'] = nn_acc
        nn.save(os.path.join(args.models_dir, "neural_net.keras"))
            
        # Save history
        import json
        with open(os.path.join(args.models_dir, "nn_history.json"), "w") as f:
            json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
            print(f"  Neural Net test accuracy: {nn_acc:.4f}\n")
    else:
        print("  Neural Net: SKIPPED (TensorFlow unavailable)")
    
        
        


    # Summary
    print("=" * 40)
    print("  Model accuracy summary")
    print("=" * 40)
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:<20} {acc:.4f}")
    print("=" * 40)
    print(f"\nAll models saved to: {args.models_dir}/")


if __name__ == "__main__":
    main()
