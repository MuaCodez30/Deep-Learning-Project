"""
train.py
--------
Model training pipeline: PCA fitting, classifier training, model selection, and artifact saving.
Three classifiers are trained (Logistic Regression, SVM, MLP) and the best by F1 is kept.
"""

from pathlib import Path

import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def fit_pca(X, n_components=0.99, save_dir=None):
    """
    Z-score normalize X, then fit PCA retaining `n_components` variance.
    If `save_dir` is provided, persists scaler and PCA to disk.

    Returns:
        scaler   — fitted StandardScaler
        pca      — fitted PCA
        X_pca    — transformed feature matrix
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # n_components=0.99 automatically determines how many components explain 99% of variance
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, save_dir / "scaler.pkl")
        joblib.dump(pca,    save_dir / "pca.pkl")
        np.save(save_dir / "X_video_features_pca.npy", X_pca)
        print(f"Saved scaler and PCA to {save_dir}")

    return scaler, pca, X_pca


def fit_lda(X_lda_input, Y_lda, n_components=2, save_dir=None):
    """
    Z-score normalize X_lda_input, then fit multi-class LDA for visualization.
    If `save_dir` is provided, persists scaler and LDA to disk.

    Returns:
        scaler_lda  — fitted StandardScaler
        lda         — fitted LDA
        X_lda       — 2-D transformed matrix
    """
    scaler_lda = StandardScaler()
    X_scaled = scaler_lda.fit_transform(X_lda_input)

    lda = LDA(n_components=n_components)
    X_lda = lda.fit_transform(X_scaled, Y_lda)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler_lda, save_dir / "scaler_lda.pkl")
        joblib.dump(lda,        save_dir / "lda.pkl")
        np.save(save_dir / "X_video_features_lda.npy", X_lda)
        print(f"Saved LDA scaler and model to {save_dir}")

    return scaler_lda, lda, X_lda


def train_all_models(X_pca, y, test_size=0.2, random_state=42,
                     mlp_hidden=(256, 128), save_dir=None):
    """
    Split data, train LR / SVM / MLP, select the best by F1, and save all artifacts.

    Args:
        X_pca:          PCA-reduced feature matrix.
        y:              Binary labels (0=real, 1=fake).
        test_size:      Fraction of data held out for evaluation.
        random_state:   Seed for reproducibility.
        mlp_hidden:     Hidden layer sizes for the MLP.
        save_dir:       If provided, all models and the scaler are saved here.

    Returns:
        best_model  — the highest-F1 trained classifier
        results     — list of {model, f1} dicts for all three classifiers
        split       — dict with train/test arrays and index arrays for later use
    """
    # Stratified split preserves the real-to-fake class ratio in both sets
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_pca, y, np.arange(len(y)),
        test_size=test_size, random_state=random_state, stratify=y
    )

    # Re-scale after the split to avoid data leakage (fit only on training data)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    lr  = LogisticRegression(max_iter=1000, random_state=random_state)
    # RBF kernel SVM — effective in high-dimensional PCA space
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=random_state)
    # Two-layer MLP with early stopping to prevent overfitting
    mlp = MLPClassifier(hidden_layer_sizes=mlp_hidden, max_iter=300,
                        random_state=random_state,
                        early_stopping=True, validation_fraction=0.1)

    results = []
    for name, model in [("LogisticRegression", lr), ("SVM", svm), ("MLP", mlp)]:
        model.fit(X_train_s, y_train)
        f1 = f1_score(y_test, model.predict(X_test_s))
        results.append({"model": name, "f1": f1})
        print(f"  {name}: F1 = {f1:.4f}")

    # Select the model with the highest F1 score
    best_name  = max(results, key=lambda r: r["f1"])["model"]
    best_model = {"LogisticRegression": lr, "SVM": svm, "MLP": mlp}[best_name]
    print(f"\nBest model: {best_name}")

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler,     save_dir / "scaler.pkl")
        joblib.dump(lr,         save_dir / "video_lr.pkl")
        joblib.dump(svm,        save_dir / "video_svm.pkl")
        joblib.dump(mlp,        save_dir / "video_mlp.pkl")
        joblib.dump(best_model, save_dir / "video_classifier.pkl")
        joblib.dump(best_model, save_dir / "final_model.pkl")
        print(f"Saved all models to {save_dir}")

    split = dict(
        X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test,
        idx_train=idx_train, idx_test=idx_test,
        X_train_scaled=X_train_s, X_test_scaled=X_test_s,
        scaler=scaler,
    )
    return best_model, results, split
