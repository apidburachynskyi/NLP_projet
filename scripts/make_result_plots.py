import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    PREPARED_CSV,
    NMF_DOC_TOPICS_CSV,
    NMF_TOPICS_CSV,
    FIGURES_OUT,
    METADATA_OUT,
)


def extract_year(x):
    if pd.isna(x):
        return None
    m = re.search(r"(19\d{2}|20\d{2})", str(x))
    return int(m.group(1)) if m else None


def load_topic_labels(topics_path):
    topics = pd.read_csv(topics_path)

    labels = {}
    for _, row in topics.iterrows():
        topic_id = int(row["topic"])
        manual = str(row.get("manual_label", "")).strip()

        if manual and manual.lower() != "nan":
            label = manual
        else:
            label = f"Topic {topic_id}"

        labels[topic_id] = f"{topic_id}: {label}"

    return labels, topics


def save_overall_topic_distribution(df):
    counts = df["topic_label"].value_counts().sort_values()

    plt.figure(figsize=(9, 5))
    counts.plot(kind="barh")
    plt.xlabel("Number of documents")
    plt.ylabel("Topic")
    plt.title("Overall NMF topic distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_OUT / "topic_size_distribution_nmf_clean.png", dpi=250)
    plt.close()

    counts.to_csv(METADATA_OUT / "topic_size_distribution_nmf_clean.csv", header=["count"])


def save_topic_share_heatmap(topic_share):
    plt.figure(figsize=(10, 5.8))

    data = topic_share.T
    plt.imshow(data.values, aspect="auto")

    plt.colorbar(label="Share of documents")
    plt.xticks(
        range(len(data.columns)),
        [str(int(y)) for y in data.columns],
        rotation=45,
        ha="right",
    )
    plt.yticks(range(len(data.index)), data.index)

    plt.xlabel("Election year")
    plt.ylabel("Topic")
    plt.title("NMF topic share by election year")

    plt.tight_layout()
    plt.savefig(FIGURES_OUT / "topic_share_by_year_heatmap_nmf_clean.png", dpi=250)
    plt.close()


def save_topic_share_lineplot(topic_share):
    plt.figure(figsize=(10, 6))

    for topic in topic_share.columns:
        plt.plot(
            topic_share.index,
            topic_share[topic],
            marker="o",
            label=topic,
        )

    plt.xlabel("Election year")
    plt.ylabel("Share of documents")
    plt.title("NMF topic share over time")
    plt.xticks(topic_share.index, [str(int(y)) for y in topic_share.index], rotation=45)
    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        frameon=True,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_OUT / "topic_share_by_year_lines_nmf_clean.png", dpi=250)
    plt.close()


def save_topic_counts_table(df):
    tab = pd.crosstab(df["year"], df["topic_label"])
    tab.to_csv(METADATA_OUT / "topic_counts_by_year_nmf_clean.csv")

    share = pd.crosstab(df["year"], df["topic_label"], normalize="index")
    share.to_csv(METADATA_OUT / "topic_share_by_year_nmf_clean.csv")

    return tab, share


def main():
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    METADATA_OUT.mkdir(parents=True, exist_ok=True)

    docs = pd.read_csv(PREPARED_CSV, low_memory=False)
    topic_docs = pd.read_csv(NMF_DOC_TOPICS_CSV)

    labels, topics = load_topic_labels(NMF_TOPICS_CSV)

    df = docs.merge(topic_docs, on="id", how="inner")

    df["year"] = df["date"].map(extract_year)
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    df["topic_label"] = df["nmf_topic"].map(labels)
    df["topic_label"] = df["topic_label"].fillna("Topic " + df["nmf_topic"].astype(str))

    save_overall_topic_distribution(df)

    topic_counts, topic_share = save_topic_counts_table(df)

    # Sort years and topics for stable plots
    topic_share = topic_share.sort_index()
    topic_share = topic_share[sorted(topic_share.columns)]

    save_topic_share_heatmap(topic_share)
    save_topic_share_lineplot(topic_share)

    print("Saved clean result plots:")
    print(FIGURES_OUT / "topic_size_distribution_nmf_clean.png")
    print(FIGURES_OUT / "topic_share_by_year_heatmap_nmf_clean.png")
    print(FIGURES_OUT / "topic_share_by_year_lines_nmf_clean.png")


if __name__ == "__main__":
    main()