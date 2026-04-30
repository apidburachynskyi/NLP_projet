import re
import unicodedata
from pathlib import Path
from typing import Iterable, List

import pandas as pd

# Domain-specific boilerplate and OCR artifacts common in Archelec files.
DOMAIN_STOPWORDS = {
    "fonds", "cevipof", "sciences", "po", "arkindex",
    "election", "elections", "élection", "élections",
    "legislative", "legislatives", "législative", "législatives",
    "circonscription", "circonscriptions", "departement", "departements",
    "département", "départements", "scrutin", "scrutins",
    "candidat", "candidate", "candidats", "candidates",
    "suppleant", "suppleante", "suppléant", "suppléante",
    "madame", "mademoiselle", "monsieur", "electeurs", "electrices",
    "électeurs", "électrices", "vote", "votes", "voter", "votez",
    "mars", "juin", "avril", "mai", "tour", "voix",
    "republique", "république", "francaise", "française",
    "liberte", "liberté", "egalite", "fraternite", "fraternité",
    "france", "francais", "français", "francaise", "française",
    "pf", "pdfmasterocr", "bv", "vu", "imp", "imprimerie", "atelier",
    "die", "der", "und", "für", "fur", "den", "das", "sie", "eine", "von",
    "wir", "ist", "mit", "werden", "ich", "ein", "zu", "in","parti communiste", 
    "parti socialiste","lutte ouvriere","sciences po","fonds cevipof","rpr", "udf", "udr", "fn", "pcf", "ps",
    "front", "national", "communiste", "socialiste",
    "gaulliste", "lepen", "pen", "mitterrand", "marchais", "laguiller",
    "pompidou", "giscard"
}

BOILERPLATE_PATTERNS = [
    r"sciences\s+po\s*/?\s*fonds\s+cevipof",
    r"fonds\s+cevipof",
    r"sciences\s+po",
    r"republique\s+francaise\s*-?\s*liberte\s*-?\s*egalite\s*-?\s*fraternite",
    r"république\s+française\s*-?\s*liberté\s*-?\s*egalité\s*-?\s*fraternité",
    r"elections?\s+legislatives?",
    r"élections?\s+législatives?",
    r"vu\s+le\s+candidat",
    r"cette\s+circulaire\s+n.?est\s+pas\s+le\s+bulletin\s+de\s+vote",
        # party names / explicit partisan markers
    r"\brpr\s*[-/]?\s*udf\b",
    r"\budf\s*[-/]?\s*rpr\b",
    r"\bfront\s+national\b",
    r"\bles\s+francais\s+d.?abord\b",
    r"\bparti\s+communiste\s+francais\b",
    r"\bparti\s+communiste\b",
    r"\bparti\s+socialiste\b",
    r"\blutte\s+ouvriere\b",
    r"\bligue\s+communiste\b",
    r"\bunion\s+des\s+republicains\s+de\s+progres\b",
    r"\brassemblement\s+pour\s+la\s+republique\b",
    r"\bunion\s+pour\s+la\s+democratie\s+francaise\b",
    r"\bles\s+verts\b",

    # highly partisan leader names
    r"\bjean\s+marie\s+le\s+pen\b",
    r"\ble\s+pen\b",
    r"\bmitterrand\b",
    r"\bmarchais\b",
    r"\blaguiller\b",
    r"\bpompidou\b",
    r"\bgiscard\b",
]

NON_SUBSTANTIVE_LINE_PATTERNS = [
    # archive / administrative headers
    r"sciences\s+po",
    r"fonds\s+cevipof",
    r"republique\s+francaise",
    r"liberte\s+egalite\s+fraternite",
    r"elections?\s+legislatives?",
    r"\b\d+(ere|eme|e)?\s+circonscription\b",
    r"departement\s+de",

    # candidate identity blocks
    r"^le\s+candidat\b",
    r"^la\s+candidate\b",
    r"^candidat\s*:",
    r"^candidate\s*:",
    r"^suppleant\s*:",
    r"^suppleante\s*:",
    r"^remplacant\s+eventuel",
    r"^president\s*:",
    r"\bdepute\s+sortant\b",
    r"\bconseiller\s+general\b",
    r"\bmaire[-\s]adjoint\b",

    # print / form / useless endings
    r"vu\s+le\s+candidat",
    r"vu\s+les\s+candidats",
    r"\bimprimerie\b",
    r"\bimp\.",
    r"atelier\s+d",
    r"bulletin\s+de\s+vote",
    r"a\s+remplir",
    r"a\s+retourner",
    r"je\s+souhaite\s+recevoir",
    r"documentation",
    r"pour\s+nous\s+ecrire",
    r"^nom\.?$",
    r"^prenom\.?$",
    r"^adresse\.?$",
]

TAIL_CUT_PATTERNS = [
    r"vu\s+le\s+candidat",
    r"vu\s+les\s+candidats",
    r"\bimprimerie\b",
    r"\bimp\.",
    r"atelier\s+d",
    r"a\s+remplir",
    r"a\s+retourner",
    r"je\s+souhaite\s+recevoir",
    r"pour\s+nous\s+ecrire",
]


def remove_non_substantive_lines(text: str) -> str:
    """
    Remove administrative headers, candidate identity lines, printer mentions,
    and reply forms while keeping substantive manifesto content.
    """
    lines = str(text).splitlines()
    kept = []
    cut_tail = False

    for i, line in enumerate(lines):
        raw = line.strip()
        if not raw:
            continue

        norm = strip_accents(raw.lower()).strip()

        # If we reach the printer/form ending, remove this line and everything after.
        if any(re.search(p, norm, flags=re.IGNORECASE) for p in TAIL_CUT_PATTERNS):
            cut_tail = True

        if cut_tail:
            continue

        # Remove common non-substantive lines.
        if any(re.search(p, norm, flags=re.IGNORECASE) for p in NON_SUBSTANTIVE_LINE_PATTERNS):
            continue

        # Remove early candidate biography lines, but only near the beginning.
        # This avoids deleting substantive lines like "retraite à 60 ans".
        if i < 25 and re.search(r"\b\d{2}\s+ans\b", norm):
            continue

        # Remove very short identity-style uppercase lines near the beginning.
        if i < 20 and len(norm.split()) <= 6 and re.search(
            r"\b(maire|conseiller|ingenieur|agriculteur|commercant|enseignant|medecin|avocat|retraite)\b",
            norm,
        ):
            continue

        kept.append(raw)

    return " ".join(kept)

def strip_accents(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text))
    return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")


def load_stopwords(path: Path | str | None = None) -> List[str]:
    words = set(DOMAIN_STOPWORDS)
    if path is not None and Path(path).exists():
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                w = line.strip().lower()
                if w:
                    words.add(w)
                    words.add(strip_accents(w))
    return sorted(words)


def clean_ocr_text(text: str) -> str:
    """Light cleaning for OCR manifesto text."""
    text = remove_non_substantive_lines(text)

    text = str(text)
    text = text.replace("\x0c", " ")
    text = text.replace("☐", " ").replace("☒", " ")
    text = text.replace("□", " ").replace("■", " ")
    text = text.lower()

    # Normalize apostrophes / dashes
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"[‐-‒–—]", "-", text)

    # Remove recurring boilerplate, with and without accents.
    no_acc = strip_accents(text)
    for pat in BOILERPLATE_PATTERNS:
        no_acc = re.sub(pat, " ", no_acc, flags=re.IGNORECASE)
    text = no_acc

    # Remove URLs/emails, numbers, OCR junk, punctuation
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"[^a-zàâçéèêëîïôûùüÿñæœ'\-\s]", " ", text)
    text = re.sub(r"\b[a-z]\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_stopwords(text: str, stopwords: Iterable[str]) -> str:
    stop = set(stopwords)
    stop = stop | {strip_accents(w) for w in stop}

    text = strip_accents(str(text).lower())
    tokens = re.findall(r"[a-z'\-]+", text)

    tokens = [
        t for t in tokens
        if t not in stop
        and len(t) > 2
    ]

    return " ".join(tokens)

def prepare_text_columns(df: pd.DataFrame, stopwords: List[str]) -> pd.DataFrame:
    df = df.copy()
    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column in dataframe")
    df["clean_text"] = df["text"].fillna("").map(clean_ocr_text)
    df["no_stop_text"] = df["clean_text"].map(lambda x: remove_stopwords(x, stopwords))
    # For speed/reproducibility, we use a cleaned no-stop version instead of spaCy lemmatization.
    # If you have a separate lemmatized.csv, it can still be merged before this step.
    if "lemmatized_text" not in df.columns:
        df["lemmatized_text"] = df["no_stop_text"]
    return df


def filter_documents(df: pd.DataFrame, min_chars: int = 300) -> pd.DataFrame:
    df = df.copy()
    df["clean_len"] = df["clean_text"].fillna("").str.len()
    df = df[df["clean_len"] >= min_chars].copy()
    df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)
    return df
