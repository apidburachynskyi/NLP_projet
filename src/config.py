from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"

TOPICS_OUT = OUTPUTS / "topics"
FIGURES_OUT = OUTPUTS / "figures"
METADATA_OUT = OUTPUTS / "metadata"
EVAL_OUT = OUTPUTS / "evaluation"

METADATA_CSV = RAW / "archelect_search.csv"
STOPWORDS_TXT = RAW / "stop_word_fr.txt"
TEXT_ROOT = RAW / "arkindex_archelec" / "text_files"
PREPARED_CSV = PROCESSED / "documents_prepared.csv"

N_TOPICS = 10
RANDOM_STATE = 42
MIN_DOC_CHARS = 300

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
# EMBEDDING_MODEL_NAME = "dangvantuan/sentence-camembert-large"

EMBEDDINGS_NPY = PROCESSED / "embeddings.npy"
EMBEDDINGS_META = PROCESSED / "embeddings_meta.json"
TOPIC_DISTR_NPY = PROCESSED / "topic_distr.npy"

# Separate LDA outputs
LDA_TOPICS_CSV = TOPICS_OUT / "lda_topics.csv"
LDA_DOC_TOPICS_CSV = TOPICS_OUT / "lda_document_topics.csv"

# Separate NMF outputs
NMF_TOPICS_CSV = TOPICS_OUT / "nmf_topics.csv"
NMF_DOC_TOPICS_CSV = TOPICS_OUT / "nmf_document_topics.csv"

# BERTopic outputs
BERTOPIC_TOPICS_CSV = TOPICS_OUT / "bertopic_topics.csv"
BERTOPIC_DOC_TOPICS_CSV = TOPICS_OUT / "bertopic_document_topics.csv"
MANUAL_LABELS_CSV = TOPICS_OUT / "manual_topic_labels.csv"

# Backward-compatible aliases, in case other files still use old names
DOCUMENT_TOPICS_CSV = BERTOPIC_DOC_TOPICS_CSV
MANUAL_TOPIC_LABELS_CSV = MANUAL_LABELS_CSV

# Old baseline aliases, not used by new pipeline
BASELINE_TOPICS_CSV = TOPICS_OUT / "baseline_topics.csv"
BASELINE_DOC_TOPICS_CSV = TOPICS_OUT / "baseline_document_topics.csv"

TOPIC_EVAL_CSV = EVAL_OUT / "topic_model_evaluation.csv"

# Main topic source for metadata heatmaps and logistic regression.
# Options: "nmf", "lda", "bertopic"
MAIN_TOPIC_SOURCE = "nmf"