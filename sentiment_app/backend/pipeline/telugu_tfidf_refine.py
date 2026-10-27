"""Compact refinement of the Telugu-only TF-IDF baseline on the Telugu-only split (D:/NLP/outputs, read-only):
fit on its train.csv, select on its validation.csv (macro F1), score its test.csv ONCE for the selected config.

    python -m pipeline.telugu_tfidf_refine
Output: artifacts/telugu_only_refine/result.json (+ model.joblib, predictions). The original baseline is unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402
from sentiment_core.experiment_io import full_metrics  # noqa: E402

SRC = paths.NLP_ROOT / "outputs"
OUT = paths.APP_ROOT / "artifacts" / "telugu_only_refine"


def feats(word, char, char_kind):
    parts = []
    if word:
        parts.append(("w", TfidfVectorizer(ngram_range=word, max_features=20000, sublinear_tf=True, token_pattern=r"(?u)\S+")))
    parts.append(("c", TfidfVectorizer(analyzer=char_kind, ngram_range=char, min_df=2, sublinear_tf=True)))
    return FeatureUnion(parts)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rd = lambda s: pd.read_csv(SRC / f"{s}.csv", encoding="utf-8-sig", low_memory=False)
    tr, va, te = rd("train"), rd("validation"), rd("test")
    x = lambda d: d["cleaned_text"].astype(str)
    grid = [((1, 2), (3, 5), "char_wb", C) for C in (0.5, 1, 2, 4)] + \
           [((1, 2), (2, 5), "char_wb", 1), ((1, 1), (2, 6), "char_wb", 1), (None, (2, 5), "char", 1), ((1, 2), (2, 6), "char", 1)]
    rows, best = [], None
    for word, char, kind, C in grid:
        m = Pipeline([("f", feats(word, char, kind)), ("c", LogisticRegression(C=C, class_weight="balanced", max_iter=2000,
                                                                                random_state=42))]).fit(x(tr), tr["label"])
        mm = full_metrics(va["label"].to_numpy(int), m.predict(x(va)))
        cfg = {"word": word, "char": char, "char_kind": kind, "C": C}
        rows.append({**cfg, "val_accuracy": mm["accuracy"], "val_macro_f1": mm["macro_f1"]})
        print(f"{cfg} -> VAL acc {mm['accuracy']:.4f} macro F1 {mm['macro_f1']:.4f}", flush=True)
        if best is None or mm["macro_f1"] > best[0]:
            best = (mm["macro_f1"], cfg, m)
    base = json.loads((SRC / "metrics_baseline.json").read_text(encoding="utf-8"))
    s, cfg, m = best
    t = full_metrics(te["label"].to_numpy(int), m.predict(x(te)))           # the selected config, scored once
    res = {"selected_on_validation": cfg, "validation_macro_f1": s, "search": rows,
           "test": {"accuracy": t["accuracy"], "macro_f1": t["macro_f1"], "weighted_f1": t["weighted_f1"],
                    "per_class_f1": {k: v["f1"] for k, v in t["per_class"].items()}},
           "original_baseline_test": {"accuracy": base["accuracy"], "macro_f1": base["macro_f1"]}}
    (OUT / "result.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    joblib.dump(m, OUT / "model.joblib")
    print("SELECTED", cfg, "| TEST acc %.4f macro F1 %.4f (original %.4f / %.4f)" % (
        t["accuracy"], t["macro_f1"], base["accuracy"], base["macro_f1"]))


if __name__ == "__main__":
    main()
