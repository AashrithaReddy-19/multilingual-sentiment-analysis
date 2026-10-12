"""Target label scheme and the documented source-label mappings (the only mappings this project accepts)."""
from __future__ import annotations

LABEL2ID: dict[str, int] = {"Negative": 0, "Neutral": 1, "Positive": 2}
ID2LABEL: dict[int, str] = {v: k for k, v in LABEL2ID.items()}
CLASS_NAMES: list[str] = [ID2LABEL[i] for i in range(3)]
LANGUAGE_TYPES: tuple[str, ...] = ("telugu", "english", "code_mixed")

# TeSent README: labels are already the strings Positive / Negative / Neutral.
TESENT_MAPPING: dict[str, str] = {"Negative": "Negative", "Neutral": "Neutral", "Positive": "Positive"}
# CMTET README lists the tags; the RANLP 2021 paper (Sec. 3.3) defines the three classes.
CMTET_MAPPING: dict[str, str] = {"NEG": "Negative", "NTL": "Neutral", "POS": "Positive"}

MODEL_C_SKIP = "Skipped — no real conversation_id/message_order available."
MODEL_B_PENDING = "Not trained / pending GPU run."


class LabelMappingError(ValueError):
    """Raised when a source label is not covered by a documented mapping."""


def parse_tweeteval_mapping(text: str) -> dict[str, str]:
    """Parse TweetEval mapping.txt ("0\\tnegative" per line) into {"0": "Negative", ...}.

    The result must cover exactly negative/neutral/positive; anything else is rejected.
    """
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 2 or not parts[0].isdigit():
            raise LabelMappingError(f"unexpected mapping.txt line: {line!r}")
        mapping[parts[0]] = parts[1].strip().capitalize()
    if sorted(mapping.values()) != sorted(CLASS_NAMES):
        raise LabelMappingError(f"mapping.txt does not define exactly {CLASS_NAMES}: {mapping}")
    return mapping


def map_label(source_label: object, mapping: dict[str, str]) -> int:
    """Map one source label to its target id using a documented mapping; unknown labels raise."""
    key = str(source_label).strip()
    if key not in mapping:
        raise LabelMappingError(f"label {source_label!r} not in documented mapping {sorted(mapping)}")
    return LABEL2ID[mapping[key]]


def validate_target_label(value: object) -> int:
    """Accept a target label given as id (0/1/2) or name (Negative/Neutral/Positive, any case)."""
    if isinstance(value, bool):
        raise LabelMappingError(f"invalid label {value!r}")
    if isinstance(value, int) and value in ID2LABEL:
        return value
    name = str(value).strip().capitalize()
    if name in LABEL2ID:
        return LABEL2ID[name]
    raise LabelMappingError(f"invalid label {value!r}; expected one of {CLASS_NAMES} or 0/1/2")
