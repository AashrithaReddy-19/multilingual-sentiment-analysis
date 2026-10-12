"""Reproducible 70/15/15 split, stratified by label x language_type, grouped so that identical
cleaned text (including texts that carry conflicting labels) never appears in more than one split."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPLITS = ("train", "validation", "test")


def split_groups(df: pd.DataFrame) -> pd.Series:
    """Group key: conversation_id when present, otherwise the cleaned text itself."""
    if "conversation_id" in df and df["conversation_id"].notna().any():
        return pd.Series(np.where(df["conversation_id"].notna(), "conv::" + df["conversation_id"].astype(str),
                                  "text::" + df["cleaned_text"]), index=df.index)
    return "text::" + df["cleaned_text"]


def grouped_stratified_split(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    """Return a Series of 'train'/'validation'/'test' (70/15/15).

    Whole groups are assigned to one split, so no group crosses splits. Groups are stratified by the
    label x language_type of their first row (for single-row groups, which are almost all of them,
    that is exact). Falls back to label-only, then unstratified, when a stratum is too small.
    """
    groups = split_groups(df)
    strata = df["label"].astype(str) + "_" + df["language_type"].astype(str)
    g = pd.DataFrame({"group": groups, "stratum": strata}).groupby("group", sort=True)["stratum"].first()
    keys, gstrata = g.index.to_numpy(), g.to_numpy()

    def _strat(values):
        counts = pd.Series(values).value_counts()
        return values if counts.min() >= 7 else None

    strat = _strat(gstrata)
    if strat is None:
        strat = _strat(np.array([s.split("_")[0] for s in gstrata]))
    tr, tmp, _, tmp_strata = train_test_split(keys, gstrata if strat is None else strat, test_size=0.30,
                                              random_state=seed, stratify=strat)
    va, te = train_test_split(tmp, test_size=0.50, random_state=seed,
                              stratify=None if strat is None else tmp_strata)
    assign = {k: "train" for k in tr} | {k: "validation" for k in va} | {k: "test" for k in te}
    return groups.map(assign).rename("split")


def leakage_checks(df: pd.DataFrame, split_col: str = "split") -> dict:
    """Overlap counts of cleaned_text (and conversation_id when present) across the three splits."""
    out = {}
    for col in ("cleaned_text", "conversation_id"):
        if col not in df or df[col].isna().all():
            out[col] = "not applicable"
            continue
        sets = {s: set(df.loc[(df[split_col] == s) & df[col].notna(), col]) for s in SPLITS}
        out[col] = {f"{a}&{b}": len(sets[a] & sets[b])
                    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    out["passed"] = all(v == "not applicable" or sum(v.values()) == 0
                        for k, v in out.items() if k != "passed")
    return out
