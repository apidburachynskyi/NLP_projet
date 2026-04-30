import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer

from src.config import (
    BERTOPIC_DOC_TOPICS_CSV,
    BERTOPIC_TOPICS_CSV,
    EMBEDDING_MODEL_NAME,
    EMBEDDINGS_META,
    EMBEDDINGS_NPY,
    MANUAL_LABELS_CSV,
    N_TOPICS,
    PREPARED_CSV,
    RANDOM_STATE,
    STOPWORDS_TXT,
    TOPIC_DISTR_NPY,
    TOPIC_EVAL_CSV,
)
from src.evaluate_topics import evaluate_model_topics
from src.preprocessing import load_stopwords


def get_docs(df: pd.DataFrame) -> list[str]:
    for col in ["clean_text", "text", "no_stop_text", "lemmatized_text"]:
        if col in df.columns:
            return df[col].fillna("").astype(str).tolist()
    raise ValueError("No usable text column found")


def compute_or_load_embeddings(docs: list[str], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    EMBEDDINGS_NPY.parent.mkdir(parents=True, exist_ok=True)
    expected_meta = {"model_name": model_name, "n_docs": len(docs)}

    if EMBEDDINGS_NPY.exists() and EMBEDDINGS_META.exists():
        try:
            meta = json.loads(EMBEDDINGS_META.read_text(encoding="utf-8"))
            if meta == expected_meta:
                print(f"Loading cached embeddings from {EMBEDDINGS_NPY}")
                return np.load(EMBEDDINGS_NPY)
        except Exception:
            pass

    print(f"Computing embeddings with {model_name}")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, show_progress_bar=True, batch_size=64)
    np.save(EMBEDDINGS_NPY, embeddings)
    EMBEDDINGS_META.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    return embeddings


def one_hot_from_topics(topics: list[int]) -> np.ndarray:
    unique_topics = sorted([t for t in set(topics) if t != -1])
    if not unique_topics:
        unique_topics = sorted(set(topics))
    topic_to_col = {t: i for i, t in enumerate(unique_topics)}
    probs = np.zeros((len(topics), len(unique_topics)))
    for i, t in enumerate(topics):
        if t in topic_to_col:
            probs[i, topic_to_col[t]] = 1.0
    return probs


def run_bertopic(n_topics: int = N_TOPICS):
    try:
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN
    except Exception as e:
        raise ImportError("Install BERTopic dependencies: pip install bertopic umap-learn hdbscan") from e

    df = pd.read_csv(PREPARED_CSV, low_memory=False)
    docs = get_docs(df)
    stopwords = load_stopwords(STOPWORDS_TXT)
    embeddings = compute_or_load_embeddings(docs)

    vectorizer_model = CountVectorizer(
        stop_words=stopwords,
        min_df=1,
        max_df=1.0,
        ngram_range=(1, 2)
    )

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=50,
        min_samples=1,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = BERTopic(
        vectorizer_model=vectorizer_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=n_topics,
        calculate_probabilities=True,
        verbose=True,
        language="multilingual",
    )

    topics, probs = topic_model.fit_transform(docs, embeddings)

    if probs is None:
        probs = one_hot_from_topics(topics)
    np.save(TOPIC_DISTR_NPY, probs)

    topic_info = topic_model.get_topic_info()
    BERTOPIC_TOPICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    topic_info.to_csv(BERTOPIC_TOPICS_CSV, index=False)

    doc_topics = df[["id"]].copy()
    doc_topics["bertopic_topic"] = topics
    doc_topics["bertopic_score"] = probs.max(axis=1) if probs.ndim == 2 else np.nan
    for k in range(probs.shape[1]):
        doc_topics[f"bertopic_prob_{k}"] = probs[:, k]
    doc_topics.to_csv(BERTOPIC_DOC_TOPICS_CSV, index=False)

    label_rows = []
    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        words = topic_model.get_topic(topic_id)
        if not words:
            top_words = ""
        else:
            top_words = ", ".join([w for w, _ in words[:15]])
        label_rows.append({
            "topic_id": topic_id,
            "count": int(row.get("Count", 0)),
            "top_words": top_words,
            "manual_label": "Outliers / heterogeneous documents" if topic_id == -1 else "",
        })
    label_df = pd.DataFrame(label_rows)
    if MANUAL_LABELS_CSV.exists():
        print(f"Manual topic label file already exists: {MANUAL_LABELS_CSV}")
        print("Delete it if you want to regenerate it.")
    else:
        label_df.to_csv(MANUAL_LABELS_CSV, index=False)
        print(f"topic label file created: {MANUAL_LABELS_CSV}")
        print("Open it and fill the manual_label column, then rerun scripts/run_04_analysis.py.")

    try:
        bertopic_topics = []
        for topic_id in topic_info["Topic"].tolist():
            if int(topic_id) == -1:
                continue
            words = topic_model.get_topic(int(topic_id)) or []
            bertopic_topics.append([w for w, _ in words[:15]])
        analyzer = vectorizer_model.build_analyzer()
        tokenized = [analyzer(d) for d in docs]
        bert_eval = evaluate_model_topics("BERTopic", bertopic_topics, tokenized, top_n=10)
        if TOPIC_EVAL_CSV.exists():
            eval_df = pd.read_csv(TOPIC_EVAL_CSV)
            eval_df = eval_df[eval_df["model"] != "BERTopic"]
            eval_df = pd.concat([eval_df, pd.DataFrame([bert_eval])], ignore_index=True)
        else:
            eval_df = pd.DataFrame([bert_eval])
        TOPIC_EVAL_CSV.parent.mkdir(parents=True, exist_ok=True)
        eval_df.to_csv(TOPIC_EVAL_CSV, index=False)
        print("Topic model evaluation:")
        print(eval_df)
    except Exception as e:
        print(f"Could not evaluate BERTopic: {e}")

    return topic_model, doc_topics


if __name__ == "__main__":
    run_bertopic()
