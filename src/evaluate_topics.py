from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd


def topic_diversity(topics: List[List[str]], top_n: int = 10) -> float:
    words = []
    for topic in topics:
        words.extend(topic[:top_n])
    if not words:
        return np.nan
    return len(set(words)) / len(words)


def compute_coherence_scores(topics: List[List[str]], texts: List[List[str]]) -> dict:
    """Compute c_v and c_npmi coherence with gensim if available."""
    try:
        from gensim.corpora import Dictionary
        from gensim.models.coherencemodel import CoherenceModel
    except Exception as e:
        print(f"Gensim coherence unavailable: {e}")
        return {"coherence_c_v": np.nan, "coherence_c_npmi": np.nan}

    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.8)
    clean_topics = [[w for w in topic if w in dictionary.token2id] for topic in topics]
    clean_topics = [t for t in clean_topics if len(t) >= 2]
    if not clean_topics:
        return {"coherence_c_v": np.nan, "coherence_c_npmi": np.nan}

    out = {}
    for metric in ["c_v", "c_npmi"]:
        try:
            cm = CoherenceModel(topics=clean_topics, texts=texts, dictionary=dictionary, coherence=metric)
            out[f"coherence_{metric}"] = float(cm.get_coherence())
        except Exception as e:
            print(f"Could not compute {metric}: {e}")
            out[f"coherence_{metric}"] = np.nan
    return out


def evaluate_model_topics(model_name: str, topics: List[List[str]], tokenized_texts: List[List[str]], top_n: int = 10) -> dict:
    scores = {
        "model": model_name,
        "topic_diversity": topic_diversity(topics, top_n=top_n),
        "n_topics": len(topics),
    }
    scores.update(compute_coherence_scores([t[:top_n] for t in topics], tokenized_texts))
    return scores


def save_evaluation(rows: list[dict], path) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print("Topic model evaluation:")
    print(df)
    return df
