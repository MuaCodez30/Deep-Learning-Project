import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def print_metrics(y_true, y_pred, title: str, target_names=None):
    # default class names
    if target_names is None:
        target_names = ["fake", "real"]

    print(f"\n{title}")
    # accuracy
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    # full class report 
    print(classification_report(
        y_true, y_pred, target_names=target_names, digits=4, zero_division=0
    ))


def plot_confusion(y_true, y_pred, title: str, labels=None, ax=None):
    # default class names
    if labels is None:
        labels = ["fake", "real"]

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)

    # if no axes provided make a new figure
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    return ax


def plot_confusions_grid(items, figsize=None):

    n = len(items)
    if figsize is None:
        figsize = (5 * n, 4)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, (title, y_true, y_pred) in zip(axes, items):
        plot_confusion(y_true, y_pred, title, ax=ax)

    plt.tight_layout()
    plt.show()
