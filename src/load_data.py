from pathlib import Path
import pandas as pd

from src.config import METADATA_CSV, PREPARED_CSV, STOPWORDS_TXT, TEXT_ROOT, MIN_DOC_CHARS
from src.preprocessing import filter_documents, load_stopwords, prepare_text_columns


def read_text_file(path: Path) -> str:
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return path.read_text(errors="ignore")


def load_ocr_texts(text_root: Path = TEXT_ROOT) -> pd.DataFrame:
    if not text_root.exists():
        raise FileNotFoundError(
            f"Text root not found: {text_root}. Clone Teklia repo into data/raw first."
        )
    rows = []
    for p in text_root.rglob("*.txt"):
        rows.append({"id": p.stem, "text": read_text_file(p), "source_path": str(p)})
    if not rows:
        raise FileNotFoundError(f"No .txt files found under {text_root}")
    return pd.DataFrame(rows)


def load_metadata(metadata_csv: Path = METADATA_CSV) -> pd.DataFrame:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    meta = pd.read_csv(metadata_csv, low_memory=False)
    if "id" not in meta.columns:
        raise ValueError("Metadata CSV must contain an 'id' column")
    return meta


def build_prepared_dataset() -> pd.DataFrame:
    stopwords = load_stopwords(STOPWORDS_TXT)
    texts = load_ocr_texts()
    meta = load_metadata()

    # Keep all OCR files, and attach metadata when the id matches.
    df = texts.merge(meta, on="id", how="left", suffixes=("", "_meta"))
    df = prepare_text_columns(df, stopwords)
    df = filter_documents(df, min_chars=MIN_DOC_CHARS)

    PREPARED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PREPARED_CSV, index=False)
    print(f"Saved prepared dataset to {PREPARED_CSV}")
    print(f"Prepared documents: {len(df)}")
    print(f"Columns: {list(df.columns)[:20]} ...")
    return df


if __name__ == "__main__":
    build_prepared_dataset()
