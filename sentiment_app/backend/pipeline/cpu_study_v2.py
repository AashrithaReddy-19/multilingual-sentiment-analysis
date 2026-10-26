"""CPU improvement study, round 2 (VALIDATION only; test.csv is never read).

    python -m pipeline.cpu_study_v2 [--extra NAME=RUN_DIR ...]

Members: A, A+, old E5 (production member), new E5 (e5_lr3e-05 winner) and any extra validation probability files.
Every method is scored by the same 5-fold CV inside validation (parameters fitted on 4 folds, scored on the held-out
fold): fine-grid weights (0.025 for pairs, 0.05 for triples), class-aware and language-aware weights over A+/old E5/
new E5, stacking (logistic regression / gradient boosting) on log-probabilities + detected language + length + top-1
probability + top1-top2 margin, confidence routing (keep the production prediction when its confidence >= t, else a
backup model decides) and a disagreement resolver (only rows where A+, old E5 and new E5 disagree are re-decided by
a meta-classifier). A method is a test candidate only if mean fold gain >= 0.005 macro F1 and better in >= 4/5 folds.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import cpu_ensemble_study as v1  # noqa: E402
from pipeline import paths  # noqa: E402
from sentiment_core.language import detect_language  # noqa: E402

OUT = paths.MULTI / "experiments" / "cpu_study"
LANGS = ("english", "telugu", "code_mixed")
NEW_E5 = "e5_lr3e-05_ce_ep3p2_bs32_len128_cpu"


def f1(y, p):
    return v1.macro_f1(y, p)


def blend(M, names, w, idx):
    return sum(wi * M[n][idx] for wi, n in zip(w, names))


def simplex(k, step):
    vals = np.round(np.arange(step, 1.0, step), 4)
    if k == 2:
        return [(a, round(1 - a, 4)) for a in vals]
    return [(a, b, round(1 - a - b, 4)) for a in vals for b in vals if 1 - a - b >= step - 1e-9]


def m_weights(names, step):
    grid = simplex(len(names), step)

    def f(M, y, tr, te, **k):
        w = max(grid, key=lambda w: f1(y[tr], blend(M, names, w, tr).argmax(1)))
        return blend(M, names, w, te).argmax(1)
    return f


def m_classaware3(M, y, tr, te, **k):
    """Per class column: weights over (A+, E5, E5new) from a coarse 0.25 simplex (regularised)."""
    names = ("A+", "E5", "E5new")
    grid = [w for w in itertools.product((0, 0.25, 0.5, 0.75, 1.0), repeat=3) if abs(sum(w) - 1) < 1e-9]
    S = np.stack([M[n] for n in names])                              # (3 models, n, 3 classes)
    best, bw = -1, None
    for wc in itertools.product(grid, repeat=3):                     # one weight vector per class
        W = np.array(wc).T                                           # (models, classes)
        s = f1(y[tr], np.einsum("mnc,mc->nc", S[:, tr], W).argmax(1))
        if s > best:
            best, bw = s, W
    return np.einsum("mnc,mc->nc", S[:, te], bw).argmax(1)


def m_langaware3(route):
    names = ("A+", "E5", "E5new")
    grid = simplex(3, 0.1)

    def f(M, y, tr, te, langs, dlangs, **k):
        lg = dlangs if route == "detected" else langs
        out = np.zeros(len(te), int)
        for L in LANGS:
            trL, teL = tr[lg[tr] == L], te[lg[te] == L]
            if not len(teL):
                continue
            w = max(grid, key=lambda w: f1(y[trL], blend(M, names, w, trL).argmax(1)))
            out[np.searchsorted(te, teL)] = blend(M, names, w, teL).argmax(1)
        return out
    return f


def feats(M, val, dl):
    names = sorted(M)
    cols = [np.log(np.clip(M[n], 1e-6, 1)) for n in names]
    for n in names:
        s = np.sort(M[n], 1)
        cols += [s[:, -1:], (s[:, -1] - s[:, -2])[:, None]]
    cols.append(np.stack([(dl == x).astype(float) for x in LANGS], 1))
    cols.append(np.log1p(val["cleaned_text"].astype(str).str.split().str.len().to_numpy(float))[:, None])
    return np.hstack(cols)


def m_stack(kind):
    def f(M, y, tr, te, X, **k):
        if kind == "lr":
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(C=0.5, max_iter=4000)
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                               l2_regularization=1.0, random_state=42)
        return m.fit(X[tr], y[tr]).predict(X[te])
    return f


def m_conf_route(backup):
    def f(M, y, tr, te, **k):
        prod = 0.5 * M["A+"] + 0.5 * M["E5"]
        alt = M[backup] if backup in M else (M["A+"] + M["E5"] + M["E5new"]) / 3
        conf = prod.max(1)
        ts = np.round(np.arange(0.35, 0.96, 0.05), 2)
        t = max(ts, key=lambda t: f1(y[tr], np.where(conf[tr] >= t, prod[tr].argmax(1), alt[tr].argmax(1))))
        return np.where(conf[te] >= t, prod[te].argmax(1), alt[te].argmax(1))
    return f


def m_disagree(M, y, tr, te, X, **k):
    from sklearn.linear_model import LogisticRegression
    preds = np.stack([M[n].argmax(1) for n in ("A+", "E5", "E5new")], 1)
    dis = ~((preds[:, 0] == preds[:, 1]) & (preds[:, 1] == preds[:, 2]))
    base = (0.5 * M["A+"] + 0.5 * M["E5"]).argmax(1)
    trd = tr[dis[tr]]
    clf = LogisticRegression(C=0.5, max_iter=4000).fit(X[trd], y[trd])
    out = base[te].copy()
    ted = dis[te]
    if ted.any():
        out[ted] = clf.predict(X[te][ted])
    return out


def detail(y, p, langs):
    from sentiment_core.experiment_io import full_metrics
    m = full_metrics(y, p, langs)
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
            **{f"{c}_f1": m["per_class"][c]["f1"] for c in ("Negative", "Neutral", "Positive")},
            **{f"{lg}_macro_f1": m["per_language"][lg]["macro_f1"] for lg in LANGS}}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra", nargs="*", default=[], help="NAME=path/to/validation/predictions.csv")
    args = ap.parse_args(argv)
    val = pd.read_csv(paths.MULTI / "data" / "validation.csv", encoding="utf-8-sig", low_memory=False)
    y, langs = val["label"].to_numpy(int), val["language_type"].to_numpy()
    dl = val["cleaned_text"].astype(str).map(detect_language).to_numpy()
    M = v1.load_members(val, None)
    newp = paths.MULTI / "experiments" / "transformers_cpu" / NEW_E5 / "predictions.csv"
    d = pd.read_csv(newp, encoding="utf-8-sig"); assert (d["row_id"].to_numpy() == val["row_id"].to_numpy()).all()
    M["E5new"] = d[v1.P].to_numpy(float)
    for spec in args.extra:
        name, path = spec.split("=", 1)
        d = pd.read_csv(path, encoding="utf-8-sig"); assert (d["row_id"].to_numpy() == val["row_id"].to_numpy()).all()
        M[name] = d[v1.P].to_numpy(float)
    X = feats(M, val, dl)
    methods = {"production A+.5/E5.5": v1.m_fixed,
               "pair A+,E5new (0.025)": m_weights(("A+", "E5new"), 0.025),
               "pair A+,E5 (0.025)": m_weights(("A+", "E5"), 0.025),
               "pair E5,E5new (0.025)": m_weights(("E5", "E5new"), 0.025),
               "triple A+,E5,E5new (0.05)": m_weights(("A+", "E5", "E5new"), 0.05),
               "triple A,A+,E5new (0.05)": m_weights(("A", "A+", "E5new"), 0.05),
               "class-aware A+,E5,E5new": m_classaware3,
               "lang-aware detected A+,E5,E5new": m_langaware3("detected"),
               "lang-aware dataset A+,E5,E5new": m_langaware3("dataset"),
               "stacking LR (all members)": m_stack("lr"), "stacking HGB (all members)": m_stack("hgb"),
               "confidence routing -> E5new": m_conf_route("E5new"),
               "confidence routing -> mean3": m_conf_route("mean3"),
               "disagreement resolver (LR)": m_disagree}
    for name in [n for n in M if n not in ("A", "A+", "E5", "E5new")]:
        methods[f"pair A+,{name} (0.025)"] = m_weights(("A+", name), 0.025)
        methods[f"triple A+,E5new,{name} (0.05)"] = m_weights(("A+", "E5new", name), 0.05)
    from sklearn.model_selection import StratifiedKFold
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    rows, oof = [], {}
    for name, f in methods.items():
        sc, pred = [], np.zeros(len(y), int)
        for tr, te in folds:
            p = f(M, y, tr, te, langs=langs, dlangs=dl, X=X)
            pred[te] = p
            sc.append(f1(y[te], p))
        oof[name] = pred
        rows.append({"method": name, "folds": sc, "cv_macro_f1": float(np.mean(sc)), "cv_std": float(np.std(sc)),
                     **{f"oof_{k}": v for k, v in detail(y, pred, langs).items()}})
        print(f"{name:<36} CV macro F1 {np.mean(sc):.4f} (+/-{np.std(sc):.4f})", flush=True)
    inc = np.array(rows[0]["folds"])
    for r in rows:
        dd = np.array(r["folds"]) - inc
        r.update({"mean_gain": float(dd.mean()), "folds_better": int((dd > 0).sum()),
                  "candidate": bool(r is not rows[0] and dd.mean() >= 0.005 and (dd > 0).sum() >= 4)})
    df = pd.DataFrame(rows).drop(columns="folds").sort_values("cv_macro_f1", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "cv_results_v2.csv", index=False)
    show = ["method", "cv_macro_f1", "cv_std", "mean_gain", "folds_better", "candidate", "oof_accuracy",
            "oof_Neutral_f1", "oof_english_macro_f1", "oof_telugu_macro_f1", "oof_code_mixed_macro_f1"]
    print("\n" + df[show].round(4).to_string(index=False))
    cands = df[df["candidate"]]["method"].tolist()
    (OUT / "candidates_v2.json").write_text(json.dumps({"candidates": cands}, indent=2), encoding="utf-8")
    print("\nTest candidates:", cands or "NONE")


if __name__ == "__main__":
    main()
