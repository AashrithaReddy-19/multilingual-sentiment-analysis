"""Steps 4, 5, 6 and 13: staged VALIDATION-only search for stronger classical (TF-IDF + linear) models.

test.csv is never read here. Every configuration is fitted on train.csv and scored on validation.csv; each stage
keeps the best configuration by validation macro F1 (ties -> validation accuracy) and passes it to the next stage:

  S0  reproduce A+ on validation (sanity check: must match the recorded 0.6962)
  S1  word TF-IDF grid          (probe classifier: LogisticRegression C=1, balanced)
  S2  char_wb TF-IDF grid       (same probe)
  S3  word + char FeatureUnion  (top word x top char, the suggested 40k/50k pair, and larger variants)
  S4  classifier comparison on the best representation (LR / LinearSVC / SGD; ladders stop early when clearly weak)
  S5  preprocessing ablation    (casing, repeated-character normalisation, emoticon tokens)
  S6  label-free signal features appended to TF-IDF

Each experiment is saved to experiments/<experiment_id>/ and upserted into experiment_leaderboard.csv. Re-running
skips experiments that already have validation_metrics.json.

    python -m pipeline.classical_search [--stages S0,S1,...] [--jobs 4]
"""
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.exp_common import (EXP, full_metrics, leaderboard_row_from, read_split, save_eval_files,  # noqa: E402
                                 stage_banner, upsert_leaderboard)
from sentiment_core.text_variants import apply_variant, signal_features  # noqa: E402

WORD_TOKEN = r"(?u)\S+"
A_PLUS_RECORDED_VAL_F1 = 0.6962040371748555
BASE = {"prep": "v1_current", "lowercase": True, "signals": 0.0,
        "clf": {"type": "lr", "C": 1.0, "class_weight": "balanced"}}


# ---------------------------------------------------------------- model construction
def word_vec(w: dict, lowercase: bool) -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="word", token_pattern=WORD_TOKEN, lowercase=lowercase, strip_accents=None,
                           ngram_range=tuple(w["ngram"]), min_df=w["min_df"], max_df=w.get("max_df", 1.0),
                           max_features=w.get("max_features"), sublinear_tf=w.get("sublinear_tf", True),
                           dtype=np.float32)


def char_vec(c: dict, lowercase: bool) -> TfidfVectorizer:
    return TfidfVectorizer(analyzer="char_wb", lowercase=lowercase, strip_accents=None, ngram_range=tuple(c["ngram"]),
                           min_df=c["min_df"], max_features=c.get("max_features"), sublinear_tf=True,
                           dtype=np.float32)


def make_clf(c: dict):
    cw = c.get("class_weight")
    if c["type"] == "lr":
        return LogisticRegression(C=c["C"], class_weight=cw, max_iter=2000 if c["C"] >= 5 else 1000, random_state=42)
    if c["type"] == "svc":
        return LinearSVC(C=c["C"], class_weight=cw, max_iter=5000, random_state=42)
    if c["type"] == "sgd":
        return SGDClassifier(loss=c["loss"], alpha=c["alpha"], class_weight=cw, max_iter=50, tol=1e-4,
                             random_state=42, n_jobs=1)
    raise ValueError(c["type"])


def build_features(cfg: dict) -> Pipeline:
    parts, weights = [], {}
    if cfg.get("word"):
        parts.append(("word", word_vec(cfg["word"], cfg["lowercase"])))
    if cfg.get("char"):
        parts.append(("char", char_vec(cfg["char"], cfg["lowercase"])))
    if cfg.get("signals"):
        parts.append(("signals", Pipeline([("extract", FunctionTransformer(signal_features)),
                                           ("scale", StandardScaler())])))
        weights["signals"] = cfg["signals"]
    union = FeatureUnion(parts, transformer_weights=weights or None)
    return Pipeline([("prep", FunctionTransformer(functools.partial(apply_variant, name=cfg["prep"]))),
                     ("features", union)])


def build_pipeline(cfg: dict) -> Pipeline:
    feats = build_features(cfg)
    return Pipeline(feats.steps + [("clf", make_clf(cfg["clf"]))])


def scores_of(clf, X) -> tuple[np.ndarray, str]:
    if hasattr(clf, "predict_proba") and not (isinstance(clf, SGDClassifier) and clf.loss == "hinge"):
        return clf.predict_proba(X), "predict_proba"
    return softmax(clf.decision_function(X), axis=1), "softmax of decision_function (uncalibrated pseudo-probability)"


# ---------------------------------------------------------------- experiment bookkeeping
def describe(cfg: dict) -> tuple[str, str, str, str]:
    feats = {k: cfg[k] for k in ("word", "char") if cfg.get(k)}
    if cfg.get("signals"):
        feats["signals"] = {"weight": cfg["signals"]}
    prep = f"{cfg['prep']}; lowercase={cfg['lowercase']}"
    model = {"lr": "LogisticRegression", "svc": "LinearSVC", "sgd": "SGDClassifier"}[cfg["clf"]["type"]]
    return model, json.dumps(feats), prep, json.dumps(cfg["clf"])


def done(exp_id: str) -> dict | None:
    f = EXP / exp_id / "validation_metrics.json"
    if not f.exists():
        return None
    p = pd.read_csv(EXP / exp_id / "predictions.csv", encoding="utf-8-sig")
    return {"id": exp_id, "cfg": json.loads((EXP / exp_id / "config.json").read_text(encoding="utf-8"))["config"],
            "val": json.loads(f.read_text(encoding="utf-8")),
            "probs": p[["prob_Negative", "prob_Neutral", "prob_Positive"]].to_numpy()}


def fit_eval(exp_id: str, cfg: dict, tr_text, ytr, va_text, yva, X=None) -> dict:
    """Fit on train, score validation. If X=(Xtr, Xva) is given, only the classifier is fitted (S4 cache)."""
    t0 = time.time()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        if X is None:
            feats = build_features(cfg)
            Xtr, Xva = feats.fit_transform(tr_text), feats.transform(va_text)
            n_feat = Xtr.shape[1]
        else:
            Xtr, Xva = X
            n_feat = Xtr.shape[1]
        clf = make_clf(cfg["clf"]).fit(Xtr, ytr)
        probs, prob_src = scores_of(clf, Xva)
    converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    # predictions come from argmax of the scores; for every family here this equals clf.predict
    m = full_metrics(yva, probs.argmax(1))
    return {"id": exp_id, "cfg": cfg, "probs": probs, "val": m, "n_features": int(n_feat),
            "fit_seconds": round(time.time() - t0, 1), "converged": converged, "prob_source": prob_src}


def persist(r: dict, va: pd.DataFrame, stage: str, notes: str = "") -> dict:
    folder = EXP / r["id"]
    if (folder / "validation_metrics.json").exists():          # loaded from disk (resume)
        return r
    model, feats, prep, hyper = describe(r["cfg"])
    (folder).mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(
        {"experiment_id": r["id"], "stage": stage, "config": r["cfg"], "n_features": r["n_features"],
         "fit_seconds": r["fit_seconds"], "converged": r["converged"], "probability_source": r["prob_source"],
         "trained_on": "train.csv", "scored_on": "validation.csv", "test_used": False}, indent=2), encoding="utf-8")
    m = save_eval_files(folder, va["label"], r["probs"], va, "validation", f"{r['id']} ({model})")
    r["val"] = m
    note = notes + ("" if r["converged"] else " [ConvergenceWarning]")
    upsert_leaderboard(leaderboard_row_from(r["id"], model, feats, prep, hyper, m, note.strip()))
    return r


def key(r: dict):
    return (r["val"]["macro_f1"], r["val"]["accuracy"])


def run_many(stage, items, tr, va, jobs, notes="", X=None) -> list[dict]:
    """items: list of (exp_id, cfg). Already-finished experiments are loaded from disk."""
    out, todo = [], []
    for eid, cfg in items:
        prev = done(eid)
        (out.append(prev) if prev else todo.append((eid, cfg)))
    if todo:
        print(f"[{stage}] fitting {len(todo)} configs ({len(out)} loaded from disk), jobs={jobs}", flush=True)
        res = Parallel(n_jobs=min(jobs, len(todo)), verbose=0)(
            delayed(fit_eval)(eid, cfg, tr["text"], tr["label"], va["text"], va["label"], X) for eid, cfg in todo)
        for r in res:
            persist(r, va, stage, notes)
            print(f"  {r['id']}: val macroF1 {r['val']['macro_f1']:.4f} acc {r['val']['accuracy']:.4f} "
                  f"P {r['val']['macro_precision']:.4f} R {r['val']['macro_recall']:.4f} "
                  f"({r['fit_seconds']}s, {r['n_features']} feats{'' if r['converged'] else ', NOT converged'})",
                  flush=True)
        out += res
    return sorted(out, key=key, reverse=True)


def texts(df: pd.DataFrame) -> dict:
    return {"text": df["cleaned_text"].astype(str).tolist(), "label": df["label"].to_numpy(dtype=int)}


def show_top(stage: str, res: list[dict], k: int = 5) -> None:
    print(f"\n[{stage}] top {min(k, len(res))} by validation macro F1:")
    for r in res[:k]:
        v = r["val"]
        print(f"  {r['id']:<34} macroF1 {v['macro_f1']:.4f}  acc {v['accuracy']:.4f}  P {v['macro_precision']:.4f}"
              f"  R {v['macro_recall']:.4f}  F1(neg/neu/pos) {v['per_class']['Negative']['f1']:.3f}/"
              f"{v['per_class']['Neutral']['f1']:.3f}/{v['per_class']['Positive']['f1']:.3f}")


# ---------------------------------------------------------------- stages
def fmt_ng(ng) -> str:
    return f"{ng[0]}{ng[1]}"


def s1_items():
    items = []
    for ng in ([1, 1], [1, 2], [1, 3]):
        for mdf in (2, 3, 5):
            for mx in (0.90, 0.95, 1.0):
                for mf in (20000, 40000, 60000, 100000):
                    w = {"ngram": ng, "min_df": mdf, "max_df": mx, "max_features": mf, "sublinear_tf": True}
                    items.append((f"s1_word_ng{fmt_ng(ng)}_md{mdf}_mx{int(mx * 100)}_f{mf // 1000}k",
                                  {**BASE, "word": w, "char": None}))
    return items


def s2_items():
    items = []
    for ng in ([2, 5], [3, 5], [3, 6], [3, 7], [4, 7]):
        for mf in (30000, 50000, 80000):
            for mdf in (2, 3):
                c = {"ngram": ng, "min_df": mdf, "max_features": mf}
                items.append((f"s2_char_ng{fmt_ng(ng)}_md{mdf}_f{mf // 1000}k", {**BASE, "word": None, "char": c}))
    return items


def s3_items(top_w: list[dict], top_c: list[dict]):
    items = []
    for i, w in enumerate(top_w):
        for j, c in enumerate(top_c):
            items.append((f"s3_union_w{i + 1}_c{j + 1}", {**BASE, "word": w["cfg"]["word"], "char": c["cfg"]["char"]}))
    items.append(("s3_union_suggested_w12_40k_c36_50k",
                  {**BASE, "word": {"ngram": [1, 2], "min_df": 2, "max_df": 1.0, "max_features": 40000, "sublinear_tf": True},
                   "char": {"ngram": [3, 6], "min_df": 2, "max_features": 50000}}))
    bw, bc = top_w[0]["cfg"]["word"], top_c[0]["cfg"]["char"]
    items.append(("s3_union_large_w100k_c150k", {**BASE, "word": {**bw, "max_features": 100000},
                                                  "char": {**bc, "max_features": 150000}}))
    items.append(("s3_union_large_w100k_cunlimited", {**BASE, "word": {**bw, "max_features": 100000},
                                                       "char": {**bc, "max_features": None}}))
    return items


def s4_ladders(feat_cfg: dict) -> list[list[tuple[str, dict]]]:
    ladders = []
    for cw in (None, "balanced"):
        tag = "bal" if cw else "none"
        ladders.append([(f"s4_lr_{tag}_C{C}", {**feat_cfg, "clf": {"type": "lr", "C": C, "class_weight": cw}})
                        for C in (0.25, 0.5, 1, 2, 3, 5, 10)])
        ladders.append([(f"s4_svc_{tag}_C{C}", {**feat_cfg, "clf": {"type": "svc", "C": C, "class_weight": cw}})
                        for C in (0.25, 0.5, 1, 2, 3)])
    for loss in ("modified_huber", "log_loss"):
        ladders.append([(f"s4_sgd_{loss}_a{a:g}", {**feat_cfg, "clf": {"type": "sgd", "loss": loss, "alpha": a,
                                                                       "class_weight": "balanced"}})
                        for a in (1e-5, 1e-4, 1e-3)])
    return ladders


def run_ladder(ladder, Xtr, ytr, Xva, yva) -> list[dict]:
    """Fit a C/alpha ladder in order; stop once two consecutive points are >0.005 below the ladder's best."""
    out, best, bad = [], -1.0, 0
    for eid, cfg in ladder:
        prev = done(eid)
        r = prev or fit_eval(eid, cfg, None, ytr, None, yva, X=(Xtr, Xva))
        r["_new"] = prev is None
        out.append(r)
        f = r["val"]["macro_f1"]
        if f > best:
            best, bad = f, 0
        elif f < best - 0.005:
            bad += 1
            if bad >= 2:
                break
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="S0,S1,S2,S3,S4,S5,S6")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args(argv)
    stages = set(args.stages.split(","))
    tr, va_df = texts(read_split("train")), read_split("validation")
    va = texts(va_df)
    best_overall = None
    t_start = time.time()

    if "S0" in stages:
        a_plus = {**BASE, "word": {"ngram": [1, 2], "min_df": 1, "max_df": 1.0, "max_features": 20000, "sublinear_tf": True},
                  "char": {"ngram": [3, 5], "min_df": 2, "max_features": None}}
        r = run_many("S0", [("s0_a_plus_reproduced", a_plus)], tr, va_df.assign(text=va["text"]), 1,
                     notes="Reproduction of Model A+ (existing model untouched)")[0]
        diff = abs(r["val"]["macro_f1"] - A_PLUS_RECORDED_VAL_F1)
        print(f"A+ reproduced: val macro F1 {r['val']['macro_f1']:.4f} (recorded {A_PLUS_RECORDED_VAL_F1:.4f}, diff {diff:.4f})")
        if diff > 2e-3:
            raise SystemExit("A+ does not reproduce on validation; stopping so the comparison stays valid.")
        stage_banner("S0 A+ reproduced on validation", "S1 word TF-IDF grid")

    V = va_df.assign(text=va["text"])
    load = lambda prefix: sorted([d for d in (done(p.name) for p in EXP.glob(f"{prefix}*")) if d], key=key, reverse=True)  # noqa: E731

    if "S1" in stages:
        res1 = run_many("S1", s1_items(), tr, V, args.jobs, notes="S1 word grid, probe LR C=1 balanced")
        show_top("S1", res1)
        stage_banner("S1 word TF-IDF grid (108 configs)", "S2 char TF-IDF grid")
    if "S2" in stages:
        res2 = run_many("S2", s2_items(), tr, V, max(1, args.jobs - 1), notes="S2 char_wb grid, probe LR C=1 balanced")
        show_top("S2", res2)
        stage_banner("S2 char_wb TF-IDF grid (30 configs)", "S3 word+char FeatureUnion")
    if "S3" in stages:
        top_w, top_c = load("s1_")[:2], load("s2_")[:2]
        res3 = run_many("S3", s3_items(top_w, top_c), tr, V, 2, notes="S3 word+char union, probe LR C=1 balanced")
        show_top("S3", res3, 8)
        stage_banner("S3 word+char FeatureUnion", "S4 classifier comparison")
    if "S4" in stages:
        rep = (load("s3_") + load("s0_"))
        rep = sorted(rep, key=key, reverse=True)[0]
        feat_cfg = {k: v for k, v in rep["cfg"].items() if k != "clf"}
        print(f"[S4] representation from {rep['id']}: {json.dumps(feat_cfg)}")
        feats = build_features(feat_cfg)
        t0 = time.time()
        Xtr, Xva = feats.fit_transform(tr["text"]), feats.transform(va["text"])
        print(f"[S4] features cached: {Xtr.shape} in {time.time() - t0:.0f}s", flush=True)
        ladders = s4_ladders(feat_cfg)
        res = Parallel(n_jobs=min(args.jobs, len(ladders)), verbose=0)(
            delayed(run_ladder)(ld, Xtr, tr["label"], Xva, va["label"]) for ld in ladders)
        res4 = []
        for ld_res, ld in zip(res, ladders):
            for r in ld_res:
                r["n_features"] = Xtr.shape[1]
                if r.pop("_new", False):
                    persist(r, V, "S4", notes=f"S4 classifier on {rep['id']} features")
                print(f"  {r['id']}: val macroF1 {r['val']['macro_f1']:.4f} acc {r['val']['accuracy']:.4f}", flush=True)
                res4.append(r)
            skipped = [eid for eid, _ in ld][len(ld_res):]
            if skipped:
                print(f"  ladder stopped early (clearly weaker); not run: {skipped}")
        res4.sort(key=key, reverse=True)
        show_top("S4", res4, 8)
        del Xtr, Xva
        stage_banner("S4 classifier comparison (LR / LinearSVC / SGD)", "S5 preprocessing ablation")
    if "S5" in stages:
        best = sorted(load("s4_") + load("s3_") + load("s0_"), key=key, reverse=True)[0]
        cfg = best["cfg"]
        print(f"[S5] base config from {best['id']} (this is variant 1 = current preprocessing, lowercase=True)")
        items = [("s5_v2_minimal_keep_case", {**cfg, "prep": "v1_current", "lowercase": False}),
                 ("s5_v3_repeat_norm", {**cfg, "prep": "v3_repeat_norm"}),
                 ("s5_v4_repeat_norm_emoticons", {**cfg, "prep": "v4_repeat_norm_emoticons"}),
                 ("s5_v4b_emoticons_only", {**cfg, "prep": "v4b_emoticons_only"})]
        res5 = run_many("S5", items, tr, V, min(args.jobs, 2),
                        notes=f"S5 preprocessing ablation vs {best['id']} (variant 1)")
        show_top("S5", sorted(res5 + [best], key=key, reverse=True), 6)
        stage_banner("S5 preprocessing ablation", "S6 sentiment-signal features")
    if "S6" in stages:
        best = sorted(load("s5_") + load("s4_") + load("s3_") + load("s0_"), key=key, reverse=True)[0]
        items = [(f"s6_signals_w{w:g}", {**best["cfg"], "signals": w}) for w in (0.3, 1.0)]
        res6 = run_many("S6", items, tr, V, 2, notes=f"S6 signal features added to {best['id']}")
        show_top("S6", sorted(res6 + [best], key=key, reverse=True), 3)
        stage_banner("S6 sentiment-signal features", "select best classical model on validation, then test once")
    print(f"total {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
