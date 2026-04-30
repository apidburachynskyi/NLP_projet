from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def save_barh(df, label_col, value_col, path: Path, title: str, xlabel: str = "Count"):
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df.sort_values(value_col, ascending=True)
    plt.figure(figsize=(10, max(4, 0.45 * len(plot_df))))
    plt.barh(plot_df[label_col], plot_df[value_col])
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_heatmap(table, path: Path, title: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, max(4, 0.45 * len(table))))
    sns.heatmap(table, cmap="Blues", linewidths=0.3)
    plt.title(title)
    plt.xlabel("Topic")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion_matrix(cm, labels, path: Path, title: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap="Blues", linewidths=0.3)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
