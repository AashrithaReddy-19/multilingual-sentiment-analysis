"""Controlled CPU experiments for multilingual TF-IDF models.

Train on train.csv, score on validation.csv ONLY (test.csv is never read here). Results go to
artifacts/multilingual/experiments/validation_results.csv. No model is selected or tested in this script.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402

OUT = paths.MULTI / "experiments"
WORD_TOKEN = r"(?u)\S+"       # whitespace tokens: keeps Telugu words, emojis, ! and ? attached as written


def word_vec(ngram, sublinear, min_df, max_features):
    return TfidfVectorizer(analyzer="word", token_pattern=WORD_TOKEN, lowercase=True, ngram_range=ngram,
                           sublinear_tf=sublinear, min_df=min_df, max_features=max_features)


def char_vec(ngram):
    # char_wb works on the raw cleaned text, so Telugu letters/vowel signs, emojis, ! and ? are all features
    return TfidfVectorizer(analyzer="char_wb", ngram_range=ngram, sublinear_tf=True, lowercase=True, min_df=2)


def build(cfg: dict) -> Pipeline:
    fam = cfg["family"]
    if fam == "word_lr":
        feats = word_vec(tuple(cfg["ngram"]), cfg["sublinear_tf"], cfg["min_df"], cfg["max_features"])
        clf = LogisticRegression(C=cfg["C"], class_weight="balanced", max_iter=1000, random_state=42)
    elif fam == "char_svc":
        feats = char_vec(tuple(cfg["char_ngram"]))
        clf = LinearSVC(C=cfg["C"], class_weight="balanced", random_state=42, max_iter=5000)
    else:  # word+char union
        feats = FeatureUnion([("word", word_vec((1, 2), True, 1, 20000)), ("char", char_vec((3, 5)))])
        clf = (LinearSVC(C=cfg["C"], class_weight="balanced", random_state=42, max_iter=5000) if fam == "union_svc"
               else LogisticRegression(C=cfg["C"], class_weight="balanced", max_iter=1000, random_state=42))
    return Pipeline([("features", feats), ("clf", clf)])


def configs() -> list[dict]:
    out = []
    for ngram, sub, mdf, mf, C in itertools.product([(1, 2), (1, 3)], [False, True], [1, 2, 3], [20000, 50000], [0.5, 1.0, 2.0]):
        out.append({"family": "word_lr", "ngram": list(ngram), "sublinear_tf": sub, "min_df": mdf, "max_features": mf, "C": C})
    for ng in [(3, 5), (3, 6)]:
        out.append({"family": "char_svc", "char_ngram": list(ng), "C": 1.0})
    for C in [0.5, 1.0]:
        out.append({"family": "union_svc", "C": C})
    for C in [0.5, 1.0, 2.0]:
        out.append({"family": "union_lr", "C": C})
    for i, c in enumerate(out):
        c["id"] = f"{c['family']}_{i:03d}"
    return out


def run_one(cfg, Xtr, ytr, Xva, yva) -> dict:
    t0 = time.time()
    model = build(cfg).fit(Xtr, ytr)
    pred = model.predict(Xva)
    return {**cfg, "val_macro_f1": float(f1_score(yva, pred, average="macro")),
            "val_accuracy": float(accuracy_score(yva, pred)),
            "val_weighted_f1": float(f1_score(yva, pred, average="weighted")),
            "n_features": int(sum(len(v.vocabulary_) for v in _vectorizers(model))),
            "fit_seconds": round(time.time() - t0, 1)}


def _vectorizers(model):
    f = model.named_steps["features"]
    return [t for _, t in f.transformer_list] if isinstance(f, FeatureUnion) else [f]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res_path = OUT / "validation_results.csv"
    if res_path.exists():
        raise SystemExit(f"{res_path} exists; refusing to overwrite.")
    read = lambda s: pd.read_csv(paths.DATA / f"{s}.csv", encoding="utf-8-sig", low_memory=False)  # noqa: E731
    train, val = read("train"), read("validation")
    Xtr, ytr, Xva, yva = train["cleaned_text"].astype(str), train["label"], val["cleaned_text"].astype(str), val["label"]
    cfgs = configs()
    (OUT / "experiment_grid.json").write_text(json.dumps(cfgs, indent=1), encoding="utf-8")
    word = [c for c in cfgs if c["family"] == "word_lr"]
    other = [c for c in cfgs if c["family"] != "word_lr"]
    print(f"{len(word)} word configs (3 parallel jobs), {len(other)} char/union configs (sequential)", flush=True)
    t0 = time.time()
    rows = Parallel(n_jobs=3, verbose=5)(delayed(run_one)(c, Xtr, ytr, Xva, yva) for c in word)
    for c in other:
        r = run_one(c, Xtr, ytr, Xva, yva)
        print(f"{r['id']}: val macro F1 {r['val_macro_f1']:.4f} acc {r['val_accuracy']:.4f} ({r['fit_seconds']}s, {r['n_features']} feats)", flush=True)
        rows.append(r)
    df = pd.DataFrame(rows)
    df["ngram"] = df["ngram"].astype(str)
    df["char_ngram"] = df.get("char_ngram", pd.Series(dtype=object)).astype(str)
    df = df.sort_values(["val_macro_f1", "val_accuracy"], ascending=False).reset_index(drop=True)
    df.to_csv(res_path, index=False)
    print(f"done in {round(time.time() - t0)}s; saved {res_path}")
    print(df.head(12)[["id", "family", "ngram", "sublinear_tf", "min_df", "max_features", "char_ngram", "C",
                       "val_macro_f1", "val_accuracy", "fit_seconds"]].to_string(index=False))


if __name__ == "__main__":
    main()
