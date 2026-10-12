"""Conservative text cleaning.

Removes only URLs, HTML tags/entities, control characters and redundant whitespace.
Keeps Telugu Unicode (including ZWJ/ZWNJ), Romanized Telugu, English, emojis, negation words
(not, no, never, ledu, kaadu, avvale, vaddu) and punctuation such as ! and ?.
No lowercasing, stemming or stop-word removal.
"""
from __future__ import annotations

import html
import re
import unicodedata

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
HTML_TAG_RE = re.compile(r"</?[A-Za-z][^<>]{0,200}>")
WS_RE = re.compile(r"\s+")
NEGATIONS: tuple[str, ...] = ("not", "no", "never", "ledu", "kaadu", "avvale", "vaddu")


def clean_text(text: object) -> str | None:
    """Return cleaned text, or None when nothing meaningful is left."""
    if text is None:
        return None
    if isinstance(text, float) and text != text:  # NaN
        return None
    t = html.unescape(str(text))
    t = HTML_TAG_RE.sub(" ", t)
    t = URL_RE.sub(" ", t)
    # Category Cc = control characters. Category Cf (ZWJ/ZWNJ) is kept: Telugu conjuncts and emoji need it.
    t = "".join(" " if unicodedata.category(ch) == "Cc" else ch for ch in t).replace("﻿", "")
    t = WS_RE.sub(" ", t).strip()
    return t or None
