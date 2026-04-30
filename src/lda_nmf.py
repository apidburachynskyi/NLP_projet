import re
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from src.config import (
    LDA_DOC_TOPICS_CSV,
    LDA_TOPICS_CSV,
    NMF_DOC_TOPICS_CSV,
    NMF_TOPICS_CSV,
    N_TOPICS,
    PREPARED_CSV,
    RANDOM_STATE,
    STOPWORDS_TXT,
    TOPIC_EVAL_CSV,
)
from src.evaluate_topics import evaluate_model_topics, save_evaluation
from src.preprocessing import load_stopwords, strip_accents


def make_analyzer(stopwords):
    stop = set(stopwords)
    stop = stop | {strip_accents(w) for w in stop}

    def analyzer(text):
        text = strip_accents(str(text).lower())

        # remove apostrophes before tokenization
        text = re.sub(r"[’'`´ʼ‘]", " ", text)

        # remove hyphens as separators
        text = re.sub(r"[-]", " ", text)

        tokens = re.findall(r"[a-z]+", text)

        tokens = [
            t for t in tokens
            if t not in stop
            and len(t) > 2
        ]

        bigrams = [
            tokens[i] + " " + tokens[i + 1]
            for i in range(len(tokens) - 1)
        ]

        return tokens + bigrams

    return analyzer


def extract_topics(components, feature_names, top_n=15):
    topics = []
    for topic in components:
        idx = np.argsort(topic)[::-1][:top_n]
        topics.append([feature_names[i] for i in idx])
    return topics


def save_topic_table(path, model_name, topics, dominant_topics):
    counts = pd.Series(dominant_topics).value_counts().to_dict()

    rows = []
    for i, words in enumerate(topics):
        rows.append(
            {
                "model": model_name,
                "topic": i,
                "count": int(counts.get(i, 0)),
                "top_words": ", ".join(words),
                "manual_label": "",
            }
        )

    topic_df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    topic_df.to_csv(path, index=False)
    return topic_df


def run_baselines(n_topics: int = N_TOPICS):
    df = pd.read_csv(PREPARED_CSV, low_memory=False)

    if "id" not in df.columns:
        raise ValueError("Expected column 'id' in prepared dataset.")

    if "clean_text" in df.columns:
        text_col = "clean_text"
    elif "text" in df.columns:
        text_col = "text"
    else:
        raise ValueError("Expected either 'clean_text' or 'text' in prepared dataset.")

    docs = df[text_col].fillna("").astype(str).tolist()

    stopwords = load_stopwords(STOPWORDS_TXT)
    analyzer = make_analyzer(stopwords)

    count_vectorizer = CountVectorizer(
        analyzer=analyzer,
        min_df=5,
        max_df=0.7,
        max_features=8000,
    )

    X_count = count_vectorizer.fit_transform(docs)

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=RANDOM_STATE,
        learning_method="batch",
        max_iter=20,
        n_jobs=-1,
    )

    W_lda = lda.fit_transform(X_count)

    lda_topics = extract_topics(
        lda.components_,
        count_vectorizer.get_feature_names_out(),
        top_n=15,
    )

    lda_dominant = W_lda.argmax(axis=1)
    lda_scores = W_lda.max(axis=1)

    lda_doc_df = df[["id"]].copy()
    lda_doc_df["lda_topic"] = lda_dominant
    lda_doc_df["lda_score"] = lda_scores

    for k in range(n_topics):
        lda_doc_df[f"lda_prob_{k}"] = W_lda[:, k]

    LDA_DOC_TOPICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    lda_doc_df.to_csv(LDA_DOC_TOPICS_CSV, index=False)

    lda_topic_df = save_topic_table(
        LDA_TOPICS_CSV,
        "LDA",
        lda_topics,
        lda_dominant,
    )

    tfidf_vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        min_df=5,
        max_df=0.7,
        max_features=8000,
    )

    X_tfidf = tfidf_vectorizer.fit_transform(docs)

    nmf = NMF(
        n_components=n_topics,
        random_state=RANDOM_STATE,
        init="nndsvda",
        max_iter=400,
    )

    W_nmf = nmf.fit_transform(X_tfidf)

    W_nmf_norm = W_nmf / np.maximum(W_nmf.sum(axis=1, keepdims=True), 1e-12)

    nmf_topics = extract_topics(
        nmf.components_,
        tfidf_vectorizer.get_feature_names_out(),
        top_n=15,
    )

    nmf_dominant = W_nmf_norm.argmax(axis=1)
    nmf_scores = W_nmf_norm.max(axis=1)

    nmf_doc_df = df[["id"]].copy()
    nmf_doc_df["nmf_topic"] = nmf_dominant
    nmf_doc_df["nmf_score"] = nmf_scores

    for k in range(n_topics):
        nmf_doc_df[f"nmf_prob_{k}"] = W_nmf_norm[:, k]

    NMF_DOC_TOPICS_CSV.parent.mkdir(parents=True, exist_ok=True)
    nmf_doc_df.to_csv(NMF_DOC_TOPICS_CSV, index=False)

    nmf_topic_df = save_topic_table(
        NMF_TOPICS_CSV,
        "NMF",
        nmf_topics,
        nmf_dominant,
    )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    tokenized = [analyzer(x) for x in docs]

    eval_rows = [
        evaluate_model_topics("LDA", lda_topics, tokenized, top_n=10),
        evaluate_model_topics("NMF", nmf_topics, tokenized, top_n=10),
    ]

    save_evaluation(eval_rows, TOPIC_EVAL_CSV)

    print(f"Saved LDA topics to {LDA_TOPICS_CSV}")
    print(f"Saved LDA document topics to {LDA_DOC_TOPICS_CSV}")
    print(f"Saved NMF topics to {NMF_TOPICS_CSV}")
    print(f"Saved NMF document topics to {NMF_DOC_TOPICS_CSV}")
    print(f"Saved topic evaluation to {TOPIC_EVAL_CSV}")

    print("\nLDA topic counts:")
    print(lda_topic_df[["model", "topic", "count", "top_words"]])

    print("\nNMF topic counts:")
    print(nmf_topic_df[["model", "topic", "count", "top_words"]])

    return lda_topic_df, nmf_topic_df, lda_doc_df, nmf_doc_df


if __name__ == "__main__":
    run_baselines()