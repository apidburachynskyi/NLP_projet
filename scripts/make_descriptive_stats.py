import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import PREPARED_CSV, FIGURES_OUT, METADATA_OUT


def extract_year(x):
    if pd.isna(x):
        return None
    m = re.search(r"(19\d{2}|20\d{2})", str(x))
    return int(m.group(1)) if m else None


def clean_category(x):
    if pd.isna(x):
        return "Unknown"
    x = str(x).strip()
    if not x or x.lower() in {"nan", "none", "non mentionné", "non mentionne"}:
        return "Unknown"
    return x


def top_counts(series, top_n=15):
    s = series.map(clean_category)
    counts = s.value_counts()
    if len(counts) > top_n:
        top = counts.head(top_n)
        other = counts.iloc[top_n:].sum()
        counts = pd.concat([top, pd.Series({"Other": other})])
    return counts


def save_bar(counts, path, title, xlabel="Count"):
    plt.figure(figsize=(9, 5))
    counts.sort_values().plot(kind="barh")
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    METADATA_OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(PREPARED_CSV, low_memory=False)

    text_col = "clean_text" if "clean_text" in df.columns else "text"
    df["doc_length_words"] = (
        df[text_col]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    if "date" in df.columns:
        df["year"] = df["date"].map(extract_year)
    else:
        df["year"] = None

    # -----------------------------
    # Descriptive statistics table
    # -----------------------------
    stats = {
        "Number of documents": len(df),
        "Average document length words": round(df["doc_length_words"].mean(), 1),
        "Median document length words": round(df["doc_length_words"].median(), 1),
        "Min document length words": int(df["doc_length_words"].min()),
        "Max document length words": int(df["doc_length_words"].max()),
        "Number of election years": int(df["year"].nunique()) if "year" in df.columns else None,
        "Number of political support categories": int(df["titulaire-soutien"].nunique()) if "titulaire-soutien" in df.columns else None,
        "Number of professions": int(df["titulaire-profession"].nunique()) if "titulaire-profession" in df.columns else None,
        "Number of age groups": int(df["titulaire-age-tranche"].nunique()) if "titulaire-age-tranche" in df.columns else None,
        "Number of departments": int(df["departement-insee"].nunique()) if "departement-insee" in df.columns else None,
    }

    stats_df = pd.DataFrame(
        [{"Statistic": k, "Value": v} for k, v in stats.items()]
    )

    stats_df.to_csv(METADATA_OUT / "descriptive_statistics.csv", index=False)
    print("\nDescriptive statistics:")
    print(stats_df.to_string(index=False))

    # -----------------------------
    # Document length distribution
    # -----------------------------
    plt.figure(figsize=(8, 5))
    df["doc_length_words"].clip(upper=df["doc_length_words"].quantile(0.99)).hist(bins=50)
    plt.xlabel("Document length, words")
    plt.ylabel("Number of documents")
    plt.title("Distribution of manifesto length")
    plt.tight_layout()
    plt.savefig(FIGURES_OUT / "document_length_distribution.png", dpi=200)
    plt.close()

    # -----------------------------
    # Documents by year
    # -----------------------------
    if df["year"].notna().any():
        docs_by_year = df["year"].dropna().astype(int).value_counts().sort_index()
        docs_by_year.to_csv(METADATA_OUT / "documents_by_year.csv", header=["count"])

        plt.figure(figsize=(9, 4.5))
        docs_by_year.plot(kind="bar")
        plt.xlabel("Election year")
        plt.ylabel("Number of manifestos")
        plt.title("Number of manifestos by year")
        plt.tight_layout()
        plt.savefig(FIGURES_OUT / "documents_by_year.png", dpi=200)
        plt.close()

    # -----------------------------
    # Political support distribution
    # -----------------------------
    if "titulaire-soutien" in df.columns:
        support_counts = top_counts(df["titulaire-soutien"], top_n=15)
        support_counts.to_csv(METADATA_OUT / "support_counts.csv", header=["count"])
        save_bar(
            support_counts,
            FIGURES_OUT / "support_distribution.png",
            "Top political support categories",
        )

    # -----------------------------
    # Profession distribution
    # -----------------------------
    if "titulaire-profession" in df.columns:
        prof_counts = top_counts(df["titulaire-profession"], top_n=15)
        prof_counts.to_csv(METADATA_OUT / "profession_counts.csv", header=["count"])
        save_bar(
            prof_counts,
            FIGURES_OUT / "profession_distribution.png",
            "Top candidate professions",
        )

    # -----------------------------
    # Age distribution
    # -----------------------------
    if "titulaire-age" in df.columns:
        age = pd.to_numeric(df["titulaire-age"], errors="coerce")
        age = age[(age >= 18) & (age <= 100)]

        if len(age) > 0:
            age.to_csv(METADATA_OUT / "candidate_age_values.csv", index=False, header=["age"])

            plt.figure(figsize=(8, 5))
            age.hist(bins=30)
            plt.xlabel("Candidate age")
            plt.ylabel("Number of documents")
            plt.title("Distribution of candidate age")
            plt.tight_layout()
            plt.savefig(FIGURES_OUT / "age_distribution.png", dpi=200)
            plt.close()

    if "titulaire-age-tranche" in df.columns:
        age_group_counts = top_counts(df["titulaire-age-tranche"], top_n=10)
        age_group_counts.to_csv(METADATA_OUT / "age_group_counts.csv", header=["count"])
        save_bar(
            age_group_counts,
            FIGURES_OUT / "age_group_distribution.png",
            "Candidate age groups",
        )

    # -----------------------------
    # Sex distribution
    # -----------------------------
    if "titulaire-sexe" in df.columns:
        sex_counts = top_counts(df["titulaire-sexe"], top_n=10)
        sex_counts.to_csv(METADATA_OUT / "sex_counts.csv", header=["count"])
        save_bar(
            sex_counts,
            FIGURES_OUT / "sex_distribution.png",
            "Candidate sex distribution",
        )

    # -----------------------------
    # Department distribution
    # -----------------------------
    dep_col = "departement-nom" if "departement-nom" in df.columns else "departement-insee"
    if dep_col in df.columns:
        dep_counts = top_counts(df[dep_col], top_n=20)
        dep_counts.to_csv(METADATA_OUT / "department_counts.csv", header=["count"])
        save_bar(
            dep_counts,
            FIGURES_OUT / "department_distribution.png",
            "Top departments in the corpus",
        )

    print("\nSaved descriptive outputs to:")
    print(METADATA_OUT)
    print(FIGURES_OUT)


if __name__ == "__main__":
    main()