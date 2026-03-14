"""
evaluate.py
-----------
All evaluation and visualization utilities:
  - Classification report + confusion matrix
  - PCA and LDA 2-D scatter plots
  - PCA cumulative explained variance curve
  - Misclassified sample listing
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def print_report(y_true, y_pred):
    """Print a full precision / recall / F1 classification report."""
    print(classification_report(y_true, y_pred, target_names=["Real", "Fake"]))


def plot_confusion_matrix(y_true, y_pred):
    """
    Plot an annotated confusion matrix heatmap.
    Rows = true label, columns = predicted label.
    Off-diagonal cells show false positives and false negatives.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


def plot_pca_scatter(X_pca, y):
    """
    Scatter plot of the first two PCA components colored by real/fake label.
    Clear cluster separation indicates the CNN features are linearly discriminative.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c="green", label="Real", alpha=0.6)
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c="red",   label="Fake", alpha=0.6)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.title("PCA 2D Projection")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_lda_scatter(X_lda, Y_lda, class_names):
    """
    Scatter plot of the first two LDA components, one color per class.
    Well-separated clusters indicate CNN features can distinguish manipulation methods.

    Args:
        class_names: dict mapping class id → name, e.g. {0: "real", 1: "deepfakes", ...}
    """
    plt.figure(figsize=(8, 6))
    for cls_id, name in class_names.items():
        mask = Y_lda == cls_id
        if not np.any(mask):
            continue
        plt.scatter(X_lda[mask, 0], X_lda[mask, 1], s=10, alpha=0.5, label=name)
    plt.xlabel("LD1")
    plt.ylabel("LD2")
    plt.title("LDA 2D Projection")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_pca_variance(pca):
    """
    Plot cumulative explained variance vs. number of PCA components.
    Useful for understanding how many components are needed to retain most variance.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.axhline(0.20, color="red", linestyle="--", label="20% reference")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def list_misclassified(idx_test, y_pred, y_test, video_ids, n=10):
    """
    Print the first `n` misclassified video IDs.
    `idx_test` tracks the original dataset indices so we can map back to video_ids.
    """
    misclassified = idx_test[y_pred != y_test]
    print(f"Misclassified samples: {len(misclassified)}")
    if video_ids is not None:
        print(video_ids[misclassified][:n])
    else:
        print("video_ids not available.")
