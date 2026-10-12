"""Loaders for the three verified datasets. Each returns raw_text, source_label, label (0/1/2) and extras."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .labels import CMTET_MAPPING, TESENT_MAPPING, map_label, parse_tweeteval_mapping

CMTET_SENT_RE = re.compile(r"^(NEG|NTL|POS):\s?(.*)$")
CMTET_ANY_TAG_RE = re.compile(r"^([A-Z]{2,5}):\s")
CMTET_LANG_TAGS = {"en", "te", "ne", "univ"}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_checksums(folder: Path) -> dict:
    """Compare every file listed in SOURCE.json with its recorded SHA-256."""
    meta = json.loads((Path(folder) / "SOURCE.json").read_text(encoding="utf-8"))
    results = {name: sha256_file(Path(folder) / name) == info["sha256"] for name, info in meta["files"].items()}
    return {"folder": str(folder), "all_match": all(results.values()), "files": results,
            "source": meta.get("source"), "commit": meta.get("commit"), "licence_note": meta.get("licence_note"),
            "citation": meta.get("citation")}


def load_tesent(folder: Path) -> pd.DataFrame:
    df = pd.read_csv(Path(folder) / "Tesent.csv", encoding="utf-8")
    return pd.DataFrame({
        "raw_text": df["Content"], "source_label": df["Label"].astype(str),
        "label": [map_label(v, TESENT_MAPPING) for v in df["Label"]],
        "annotations": df.get("Annotations"), "source_dataset": "TeSent", "language_type": "telugu",
    })


def _lines(path: Path) -> list[str]:
    txt = Path(path).read_text(encoding="utf-8")
    lines = [line.rstrip("\r") for line in txt.split("\n")]
    return lines[:-1] if txt.endswith("\n") else lines


def load_tweeteval(folder: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    folder = Path(folder)
    mapping = parse_tweeteval_mapping((folder / "mapping.txt").read_text(encoding="utf-8"))
    parts = []
    for split in ("train", "val", "test"):
        texts, labels = _lines(folder / f"{split}_text.txt"), _lines(folder / f"{split}_labels.txt")
        if len(texts) != len(labels):
            raise ValueError(f"TweetEval {split}: {len(texts)} texts vs {len(labels)} labels")
        labels = [x.strip() for x in labels]
        parts.append(pd.DataFrame({"raw_text": texts, "source_label": labels,
                                   "label": [map_label(x, mapping) for x in labels], "original_split": split}))
    df = pd.concat(parts, ignore_index=True)
    df["source_dataset"], df["language_type"] = "TweetEval", "english"
    return df, mapping


def parse_cmtet(path: Path) -> tuple[pd.DataFrame, dict]:
    """Parse codemix_sentiment_data.txt: 'NEG|NTL|POS: sentence' followed by a language-tag line."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    recs, issues, unknown = [], Counter(), Counter()
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        m = CMTET_SENT_RE.match(line)
        if not m:
            t = CMTET_ANY_TAG_RE.match(line)
            if t:
                unknown[t.group(1)] += 1
            issues["unparsed_line"] += 1
            i += 1
            continue
        tag, sent = m.group(1), m.group(2).strip()
        j = i + 1
        while j < n and not lines[j].strip():
            j += 1
        if j < n and not CMTET_SENT_RE.match(lines[j]) and set(lines[j].lower().split()) <= CMTET_LANG_TAGS:
            lang_tags, nxt = lines[j].strip().lower(), j + 1
        else:
            lang_tags, nxt = None, i + 1
            issues["missing_language_tag_line"] += 1
        recs.append({"raw_text": sent, "source_label": tag, "language_tags": lang_tags, "line_no": i + 1})
        i = nxt
    df = pd.DataFrame(recs, columns=["raw_text", "source_label", "language_tags", "line_no"])
    audit = {"parsed_sentences": len(df), "issues": dict(issues), "unknown_tags": dict(unknown),
             "tag_counts": df["source_label"].value_counts().to_dict()}
    return df, audit


def load_cmtet(folder: Path) -> tuple[pd.DataFrame, dict]:
    df, audit = parse_cmtet(Path(folder) / "codemix_sentiment_data.txt")
    if audit["unknown_tags"] or audit["issues"].get("unparsed_line"):
        raise ValueError(f"CMTET parser audit failed: {audit}")
    df["label"] = [map_label(t, CMTET_MAPPING) for t in df["source_label"]]
    df["source_dataset"], df["language_type"] = "CMTET", "code_mixed"
    return df, audit
