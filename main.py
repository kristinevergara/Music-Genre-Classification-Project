import os
import numpy as np
import joblib
import argparse

from data_preparation import build_dataset, preprocess
from training_module import train_svm, train_random_forest, train_knn, train_nn
from evaluation import load_models, predict, plot_confusion_matrix, plot_model_comparison, plot_nn_training

from sklearn.metrics import accuracy_score, f1_score, classification_report

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    type=str, default="./Data/genres_original")
    parser.add_argument("--out_dir",     type=str, default="./data")
    parser.add_argument("--models_dir",  type=str, default="./models")
    parser.add_argument("--results_dir", type=str, default="./results")
    args = parser.parse_args()

    os.makedirs(args.out_dir,     exist_ok=True)
    os.makedirs(args.models_dir,  exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)


    print("Extracting features...")
    X, y = build_dataset(args.data_dir)
    preprocess(X, y, args.out_dir)

    print("\nLoading preprocessed data...")
    X_train = np.load(os.path.join(args.out_dir, "X_train.npy"))
    X_test  = np.load(os.path.join(args.out_dir, "X_test.npy"))
    y_train = np.load(os.path.join(args.out_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(args.out_dir, "y_test.npy"))

    print("\nTraining models...")
    svm = train_svm(X_train, y_train)
    joblib.dump(svm, os.path.join(args.models_dir, "svm.pkl"))

    rf = train_random_forest(X_train, y_train)
    joblib.dump(rf, os.path.join(args.models_dir, "random_forest.pkl"))

    knn = train_knn(X_train, y_train)
    joblib.dump(knn, os.path.join(args.models_dir, "knn.pkl"))

    train_nn(X_train, y_train, X_test, y_test, args.models_dir)


    print("\nEvaluating models...")
    le = joblib.load(os.path.join(args.out_dir, "label_encoder.pkl"))
    models = load_models(args.models_dir)
    results = {}

    for name, model in models.items():
        preds = predict(model, X_test, name)
        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average='weighted')
        results[name] = {'accuracy': acc, 'f1': f1}

        cm_path = os.path.join(args.results_dir, f"confusion_{name.lower().replace(' ', '_')}.png")
        plot_confusion_matrix(y_test, preds, name, cm_path)
        print(classification_report(y_test, preds, target_names=le.classes_))

    plot_model_comparison(results, os.path.join(args.results_dir, "model_comparison.png"))
    plot_nn_training(
        os.path.join(args.models_dir, "nn_history.json"),
        os.path.join(args.results_dir, "nn_training_curves.png")
    )

