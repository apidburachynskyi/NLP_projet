import re
import unicodedata

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from src.config import (
    BERTOPIC_DOC_TOPICS_CSV,
    FIGURES_OUT,
    LDA_DOC_TOPICS_CSV,
    LDA_TOPICS_CSV,
    MAIN_TOPIC_SOURCE,
    MANUAL_LABELS_CSV,
    METADATA_OUT,
    NMF_DOC_TOPICS_CSV,
    NMF_TOPICS_CSV,
    PREPARED_CSV,
)
from src.plots import save_barh, save_confusion_matrix, save_heatmap


def strip_accents(text):
    text = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def clean_string(x):
    if pd.isna(x):
        return "Unknown"
    x = str(x).strip()
    if not x or x.lower() in {"nan", "none", "non mentionné", "non mentionne"}:
        return "Unknown"
    return x


def simplify_support(x):
    x = clean_string(x)
    if x == "Unknown":
        return x
    return x.split(";")[0].strip()


def group_profession(x):
    x = clean_string(x)
    y = strip_accents(x.lower())

    if y == "unknown":
        return "Unknown"

    if re.search(r"agric|cultivateur|paysan|exploitant agricole", y):
        return "Agriculture"
    if re.search(r"ouvrier|employe|salarie|travailleur", y):
        return "Workers / employees"
    if re.search(r"enseign|professeur|instituteur|education", y):
        return "Education"
    if re.search(r"avocat|medecin|notaire|pharmacien|ingenieur|cadre", y):
        return "Liberal professions / executives"
    if re.search(r"commerc|artisan|industriel|entrepreneur|chef", y):
        return "Business / commerce"
    if re.search(r"retrait", y):
        return "Retired"

    return "Other"


def cramers_v(table):
    chi2, p, dof, expected = chi2_contingency(table)
    n = table.to_numpy().sum()
    r, k = table.shape
    denom = n * min(k - 1, r - 1)
    v = np.sqrt(chi2 / denom) if denom > 0 else np.nan
    return chi2, p, dof, v


def load_topic_labels(path, topic_col="topic"):
    if not path.exists():
        print(f"Missing topic label file: {path}")
        return {}

    labels = pd.read_csv(path)

    if topic_col not in labels.columns:
        raise ValueError(f"{path} must contain column '{topic_col}'")

    if "manual_label" not in labels.columns:
        labels["manual_label"] = ""

    labels["manual_label"] = labels["manual_label"].fillna("").astype(str)

    out = {}
    for _, row in labels.iterrows():
        topic_id = int(row[topic_col])
        manual_label = row["manual_label"].strip()

        if manual_label:
            out[topic_id] = manual_label
        else:
            out[topic_id] = f"Topic {topic_id}"

    return out


def load_final_dataset(topic_source: str = MAIN_TOPIC_SOURCE):
    topic_source = topic_source.lower()

    df = pd.read_csv(PREPARED_CSV, low_memory=False)

    if topic_source == "nmf":
        topics = pd.read_csv(NMF_DOC_TOPICS_CSV)
        labels = load_topic_labels(NMF_TOPICS_CSV, topic_col="topic")

        df = df.merge(topics, on="id", how="inner")
        df["topic_id"] = df["nmf_topic"].astype(int)
        df["topic_score"] = df["nmf_score"]
        prob_cols = [c for c in df.columns if c.startswith("nmf_prob_")]

    elif topic_source == "lda":
        topics = pd.read_csv(LDA_DOC_TOPICS_CSV)
        labels = load_topic_labels(LDA_TOPICS_CSV, topic_col="topic")

        df = df.merge(topics, on="id", how="inner")
        df["topic_id"] = df["lda_topic"].astype(int)
        df["topic_score"] = df["lda_score"]
        prob_cols = [c for c in df.columns if c.startswith("lda_prob_")]

    elif topic_source == "bertopic":
        topics = pd.read_csv(BERTOPIC_DOC_TOPICS_CSV)
        labels = load_topic_labels(MANUAL_LABELS_CSV, topic_col="topic_id")

        df = df.merge(topics, on="id", how="inner")
        df["topic_id"] = df["bertopic_topic"].astype(int)
        df["topic_score"] = df["bertopic_score"]
        prob_cols = [c for c in df.columns if c.startswith("bertopic_prob_")]

    else:
        raise ValueError("topic_source must be 'nmf', 'lda', or 'bertopic'")

    df["topic_name"] = df["topic_id"].map(labels).fillna(
        "Topic " + df["topic_id"].astype(str)
    )
    df["topic_label"] = df["topic_id"].astype(str) + ": " + df["topic_name"].astype(str)

    df["support_clean"] = df.get(
        "titulaire-soutien",
        pd.Series(["Unknown"] * len(df), index=df.index),
    ).map(simplify_support)

    df["profession_group"] = df.get(
        "titulaire-profession",
        pd.Series(["Unknown"] * len(df), index=df.index),
    ).map(group_profession)

    if "titulaire-age-tranche" in df.columns:
        df["age_group"] = df["titulaire-age-tranche"].map(clean_string)
    elif "titulaire-age" in df.columns:
        df["age_group"] = df["titulaire-age"].map(clean_string)
    else:
        df["age_group"] = "Unknown"

    return df, prob_cols


def top_categories(s, n=8):
    top = s.value_counts().head(n).index
    return s.where(s.isin(top), "Other")


def valid_topic_mask(df):
    return df["topic_id"] != -1


def make_heatmap(df, group_col, filename, title, top_n=8):
    tmp = df.copy()
    tmp = tmp[(tmp[group_col] != "Unknown") & valid_topic_mask(tmp)]

    if tmp.empty:
        print(f"No data for heatmap: {group_col}")
        return None

    tmp[group_col + "_top"] = top_categories(tmp[group_col], top_n)

    tab = pd.crosstab(
        tmp[group_col + "_top"],
        tmp["topic_label"],
        normalize="index",
    )

    METADATA_OUT.mkdir(parents=True, exist_ok=True)
    tab.to_csv(METADATA_OUT / f"{filename}.csv")

    save_heatmap(
        tab,
        FIGURES_OUT / f"{filename}.png",
        title,
    )

    return tab


def association_tests(df):
    rows = []

    for col in ["support_clean", "profession_group", "age_group"]:
        tmp = df[(df[col] != "Unknown") & valid_topic_mask(df)].copy()

        if tmp.empty or tmp[col].nunique() < 2 or tmp["topic_id"].nunique() < 2:
            rows.append(
                {
                    "metadata_variable": col,
                    "chi2": np.nan,
                    "p_value": np.nan,
                    "dof": np.nan,
                    "cramers_v": np.nan,
                    "n_rows": len(tmp),
                }
            )
            continue

        table = pd.crosstab(tmp[col], tmp["topic_id"])
        chi2, p, dof, v = cramers_v(table)

        rows.append(
            {
                "metadata_variable": col,
                "chi2": chi2,
                "p_value": p,
                "dof": dof,
                "cramers_v": v,
                "n_rows": len(tmp),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(METADATA_OUT / "metadata_association_tests.csv", index=False)

    print("Association tests:")
    print(out)

    return out


def prediction_task(df, prob_cols, target_col="support_clean", min_docs=100):
    if not prob_cols:
        print("No topic probability columns found; skipping prediction.")
        return None

    pred_df = df[(df[target_col] != "Unknown") & valid_topic_mask(df)].copy()

    counts = pred_df[target_col].value_counts()
    valid_classes = counts[counts >= min_docs].index
    pred_df = pred_df[pred_df[target_col].isin(valid_classes)]

    if pred_df[target_col].nunique() < 2:
        print("Not enough classes for prediction.")
        return None

    X = pred_df[prob_cols].values
    y = pred_df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_dummy = dummy.predict(X_test)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    results = pd.DataFrame(
        [
            {
                "model": "Dummy majority baseline",
                "accuracy": accuracy_score(y_test, y_dummy),
                "macro_f1": f1_score(y_test, y_dummy, average="macro"),
            },
            {
                "model": f"Logistic regression on {MAIN_TOPIC_SOURCE.upper()} topic proportions",
                "accuracy": accuracy_score(y_test, y_pred),
                "macro_f1": f1_score(y_test, y_pred, average="macro"),
            },
        ]
    )

    results.to_csv(METADATA_OUT / "prediction_results.csv", index=False)

    print("Prediction results:")
    print(results)

    labels = sorted(np.unique(y_test))
    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize="true")

    save_confusion_matrix(
        cm,
        labels,
        FIGURES_OUT / "confusion_matrix_support_prediction.png",
        f"Political support prediction from {MAIN_TOPIC_SOURCE.upper()} topics",
    )

    return results


def run_analysis(topic_source: str = MAIN_TOPIC_SOURCE):
    topic_source = topic_source.lower()

    METADATA_OUT.mkdir(parents=True, exist_ok=True)
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    print(f"Running metadata analysis with topic source: {topic_source}")

    df, prob_cols = load_final_dataset(topic_source)
    df.to_csv(METADATA_OUT / "final_topic_metadata_dataset.csv", index=False)

    topic_counts = (
        df[valid_topic_mask(df)]["topic_label"]
        .value_counts()
        .reset_index()
    )
    topic_counts.columns = ["topic_label", "count"]

    save_barh(
        topic_counts,
        "topic_label",
        "count",
        FIGURES_OUT / "topic_size_distribution.png",
        f"Topic size distribution ({topic_source.upper()})",
    )

    topic_counts.to_csv(METADATA_OUT / "topic_size_distribution.csv", index=False)

    make_heatmap(
        df,
        "support_clean",
        "topic_by_support_heatmap",
        f"{topic_source.upper()} topic distribution by political support",
    )

    make_heatmap(
        df,
        "profession_group",
        "topic_by_profession_heatmap",
        f"{topic_source.upper()} topic distribution by profession group",
    )

    make_heatmap(
        df,
        "age_group",
        "topic_by_age_heatmap",
        f"{topic_source.upper()} topic distribution by age group",
    )

    association_tests(df)
    prediction_task(df, prob_cols)

    print("Analysis complete.")


if __name__ == "__main__":
    run_analysis()