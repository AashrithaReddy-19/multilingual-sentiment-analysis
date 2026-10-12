"""Context construction from earlier messages of the same conversation only (never future messages)."""
from __future__ import annotations

import pandas as pd

REQUIRED_CONVERSATION_COLUMNS = ("conversation_id", "message_order", "text", "label", "language_type")


def build_context(df: pd.DataFrame, text_col: str = "cleaned_text") -> pd.DataFrame:
    """Add `context_text`:

    [PREVIOUS_2] <t-2> [PREVIOUS_1] <t-1> [CURRENT] <t>  (fewer prefixes for the first messages).
    Rows without conversation_id/message_order get context_text = cleaned text.
    """
    out = df.copy()
    out["context_text"] = out[text_col]
    has_meta = out["conversation_id"].notna() & out["message_order"].notna() if {
        "conversation_id", "message_order"} <= set(out.columns) else pd.Series(False, index=out.index)
    if not has_meta.any():
        return out
    sub = out[has_meta].sort_values(["conversation_id", "message_order"], kind="mergesort")
    grouped = sub.groupby("conversation_id", sort=False)[text_col]
    prev1, prev2 = grouped.shift(1), grouped.shift(2)  # positive shifts look backwards only
    ctx = []
    for cur, p1, p2 in zip(sub[text_col], prev1, prev2):
        parts = []
        if isinstance(p2, str):
            parts.append(f"[PREVIOUS_2] {p2}")
        if isinstance(p1, str):
            parts.append(f"[PREVIOUS_1] {p1}")
        parts.append(f"[CURRENT] {cur}")
        ctx.append(" ".join(parts))
    out.loc[sub.index, "context_text"] = ctx
    return out


def has_real_context(df: pd.DataFrame) -> bool:
    """True only if at least one row received earlier-message context."""
    return bool((df["context_text"] != df["cleaned_text"]).any())
