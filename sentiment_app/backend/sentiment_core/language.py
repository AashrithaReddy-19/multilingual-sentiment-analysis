"""Heuristic language-type detection for single inputs at prediction time.

The training data has a language_type per dataset (TeSent = telugu, TweetEval = english, CMTET = code_mixed), but a
user-typed message has none, so the app guesses it from the script and a short list of common Romanized Telugu
words. This is a transparent HEURISTIC, shown as such in the UI; it is never used to compute any reported metric.
"""
from __future__ import annotations

import re

TELUGU_RE = re.compile("[ఀ-౿]")
LATIN_RE = re.compile("[A-Za-z]")
WORD_RE = re.compile(r"[A-Za-z]+")
# Distinctive Romanized Telugu words (one is enough) and short particles (two are needed: "ga", "ra" ...).
STRONG = {"ledu", "ledhu", "kadu", "kaadu", "kaadhu", "vaddu", "avvale", "avvaledu", "chala", "chaala", "bagundi",
          "baagundi", "bagunnadi", "bagaledu", "bagoledu", "undi", "unnadi", "unnaru", "enti", "endi", "emi", "nenu",
          "nuvvu", "meeru", "manchi", "chesadu", "chesaru", "chesindi", "cheyyi", "cheppu", "cheppandi", "asalu",
          "kuda", "kooda", "inka", "anna", "annaya", "akka", "babu", "naaku", "maaku", "neeku", "mee", "mana",
          "ippudu", "eppudu", "ekkada", "enduku", "ela", "ayindi", "avutundi", "vachindi", "poyindi"}
WEAK = {"ga", "gaa", "ra", "raa", "lo", "ki", "ni", "ne", "va", "em", "ante", "kani", "kada"}
LANGUAGE_NOTE = "Heuristic guess from script and common Romanized Telugu words (not a trained language classifier)."


def detect_language(text: str) -> str:
    """Return 'telugu', 'english', 'code_mixed' or 'unknown' (no letters, e.g. emoji only)."""
    t = str(text or "")
    has_te, has_lat = bool(TELUGU_RE.search(t)), bool(LATIN_RE.search(t))
    if has_te and has_lat:
        return "code_mixed"
    if has_te:
        return "telugu"
    if not has_lat:
        return "unknown"
    words = [w.lower() for w in WORD_RE.findall(t)]
    strong = sum(w in STRONG for w in words)
    weak = sum(w in WEAK for w in words)
    return "code_mixed" if strong >= 1 or weak >= 2 else "english"
