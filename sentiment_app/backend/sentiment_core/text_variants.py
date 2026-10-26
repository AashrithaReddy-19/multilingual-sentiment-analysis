"""Optional, conservative text variants and label-free signal features for the classical models (Phase 3).

All variants start from cleaned_text (URLs/HTML/control chars/extra whitespace already removed) and never touch
Telugu letters, emojis, negation words or ! and ?. A variant is only used if it improves VALIDATION macro F1.
"""
from __future__ import annotations

import re

import numpy as np

# 3+ repeats of the same Latin letter -> 2 ("goooood" -> "good", "baaaad" -> "baad"); collapsing to 2 (not 1)
# keeps real double letters ("good", "too", "baagundi") intact. Telugu script is never altered.
LATIN_REPEAT_RE = re.compile(r"([A-Za-z])\1{2,}")
PUNCT_REPEAT_RE = re.compile(r"([!?.])\1{2,}")           # "!!!!!" -> "!!!", keeps emphasis but bounds it

EMOTICONS = [  # (pattern, token); matched only as separate tokens so "x:)" style noise is not touched
    (r"(?<!\S)[:;=8]-?[)\]D}]+(?!\S)|(?<!\S)\^_*\^(?!\S)|(?<!\S)<3+(?!\S)", " EMO_POS "),
    (r"(?<!\S)[:;=]-?[(\[{]+(?!\S)|(?<!\S):'-?\((?!\S)|(?<!\S)</3(?!\S)|(?<!\S)-_+-(?!\S)", " EMO_NEG "),
    (r"(?<!\S)[:;=]-?[Pp](?!\S)|(?<!\S);-?\)(?!\S)", " EMO_PLAYFUL "),
]
EMOTICON_RES = [(re.compile(p), t) for p, t in EMOTICONS]
WS_RE = re.compile(r"\s+")


def normalize_repeats(text: str) -> str:
    return PUNCT_REPEAT_RE.sub(r"\1\1\1", LATIN_REPEAT_RE.sub(r"\1\1", text))


def normalize_letter_repeats(text: str) -> str:
    """Repeat normalisation for Latin letters only; punctuation is left exactly as typed."""
    return LATIN_REPEAT_RE.sub(r"", text)


def emoticon_tokens(text: str) -> str:
    for rx, tok in EMOTICON_RES:
        text = rx.sub(tok, text)
    return WS_RE.sub(" ", text).strip()


VARIANTS = {
    "v1_current": lambda t: t,                                   # cleaned_text as used by A/A+
    "v3_repeat_norm": normalize_repeats,
    "v3b_letter_repeat_norm": normalize_letter_repeats,          # less aggressive: punctuation untouched
    "v4_repeat_norm_emoticons": lambda t: emoticon_tokens(normalize_repeats(t)),
    "v4b_emoticons_only": emoticon_tokens,
}


def apply_variant(texts, name: str) -> list[str]:
    fn = VARIANTS[name]
    return [fn(str(t)) for t in texts]


# ---- label-free signal features (fixed hand-written lexicons, never derived from labels) ----
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F1E6-\U0001F1FF]")
POS_EMOJI = set("😀😁😃😄😆😊😍🥰😘🤩😎👍👏🙌🎉💐❤♥💕💖💗💙💚💛💜🤍🥳💯🙏🌹✨⭐👌😇🤗")
NEG_EMOJI = set("😡🤬😠😤😢😭😞😔😟☹🙁💔👎🤮🤢😒🙄😩😫😖😣😱😰🖕")
NEG_WORDS_RE = re.compile(r"\b(not|no|never|nothing|nobody|none|cannot|can't|don't|didn't|isn't|wasn't|won't|"
                          r"ledu|ledhu|kadu|kaadu|kaadhu|vaddu|avvale|avvaledu)\b|లేదు|కాదు|వద్దు|కాలేదు|లేను",
                          re.IGNORECASE)
TELUGU_RE = re.compile("[ఀ-౿]")
LATIN_RE = re.compile("[A-Za-z]")
SIGNAL_NAMES = ["emoji_count", "pos_emoji_count", "neg_emoji_count", "exclamation_count", "question_count",
                "negation_count", "log_char_len", "log_word_len", "latin_ratio", "telugu_ratio"]


def signal_features(texts) -> np.ndarray:
    rows = []
    for t in texts:
        t = str(t)
        n_lat, n_tel = len(LATIN_RE.findall(t)), len(TELUGU_RE.findall(t))   # Telugu vowel signs are not isalpha()
        letters = (n_lat + n_tel) or 1
        rows.append([len(EMOJI_RE.findall(t)), sum(ch in POS_EMOJI for ch in t), sum(ch in NEG_EMOJI for ch in t),
                     t.count("!"), t.count("?"), len(NEG_WORDS_RE.findall(t)), np.log1p(len(t)),
                     np.log1p(len(t.split())), n_lat / letters, n_tel / letters])
    return np.asarray(rows, dtype=np.float64)
