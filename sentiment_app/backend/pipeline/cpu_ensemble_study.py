"""CPU-only improvement study on VALIDATION (test.csv is never read here).

    python -m pipeline.cpu_ensemble_study [--extra-e5 RUN_ID]

Every method is scored by 5-fold cross-validation INSIDE validation.csv (parameters fitted on 4 folds, scored on the
held-out fold; same folds for every method), so methods with fitted parameters are compared fairly with the fixed
incumbent (A+ 0.50 / E5 0.50). Methods:
  global weights (pairs / triples of A, A+, E5[, E5b]), class-aware weights (one A+/E5 weight per class),
  language-aware weights (routing by the app's detected language AND by dataset language), class-bias offsets on
  the incumbent (Neutral boundary), temperature scaling (calibration only), stacking (logistic regression and
  gradient boosting on member log-probabilities + detected language + length + confidence).
A method becomes a test candidate only if its mean fold gain over the incumbent is >= 0.005 macro F1 AND it beats the
incumbent in >= 4 of 5 folds. Candidates are then refitted on all of validation and frozen in
experiments/cpu_study/candidates.json; testing them is a separate, single step.
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
from pipeline import paths  # noqa: E402
from sentiment_core.language import detect_language  # noqa: E402

EXP = paths.MULTI / "experiments"
OUT = EXP / "cpu_study"
P = ["prob_Negative", "prob_Neutral", "prob_Positive"]
LANGS = ("english", "telugu", "code_mixed")
GRID = np.round(np.arange(0.1, 0.91, 0.1), 2)


def macro_f1(y, pred) -> float:
    f = []
    for k in range(3):
        tp = np.sum((pred == k) & (y == k)); fp = np.sum((pred == k) & (y != k)); fn = np.sum((pred != k) & (y == k))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def load_members(val: pd.DataFrame, extra_e5: str | None) -> dict[str, np.ndarray]:
    def csv(p):
        d = pd.read_csv(p, encoding="utf-8-sig")
        if "row_id" in d:
            assert (d["row_id"].to_numpy() == val["row_id"].to_numpy()).all(), p
        return d[P].to_numpy(float)
    import joblib
    a = joblib.load(paths.MULTI / "model_a" / "tfidf_logreg_multilingual.joblib")
    pa = a.predict_proba(val["cleaned_text"].astype(str))[:, np.argsort(a.classes_)]
    m = {"A": pa, "A+": csv(EXP / "s0_a_plus_reproduced" / "predictions.csv"),
         "E5": csv(EXP / "transformers" / "e5_lr5e-05_ce_ep3p2_bs32_len128_cpu" / "predictions.csv")}
    if extra_e5:
        m["E5b"] = csv(EXP / "transformers_cpu" / extra_e5 / "predictions.csv")
    return m


# ---------------------------------------------------------------- methods: fit(train idx) -> predict(test idx)
def m_fixed(M, y, tr, te, **k):
    return (0.5 * M["A+"][te] + 0.5 * M["E5"][te]).argmax(1)


def m_global(names):
    def f(M, y, tr, te, **k):
        grid = [w for w in itertools.product(GRID, repeat=len(names) - 1) if sum(w) < 1 - 1e-9]
        best = max(grid, key=lambda w: macro_f1(y[tr], sum(wi * M[n][tr] for wi, n in zip(list(w) + [1 - sum(w)], names)).argmax(1)))
        ws = list(best) + [1 - sum(best)]
        return sum(wi * M[n][te] for wi, n in zip(ws, names)).argmax(1)
    return f


def m_classaware(M, y, tr, te, **k):
    a, e = M["A+"], M["E5"]
    best, bw = -1, None
    for w in itertools.product(GRID, repeat=3):          # one A+ weight per class column
        w = np.array(w)
        s = macro_f1(y[tr], (w * a[tr] + (1 - w) * e[tr]).argmax(1))
        if s > best:
            best, bw = s, w
    return (bw * a[te] + (1 - bw) * e[te]).argmax(1)


def m_langaware(key):
    def f(M, y, tr, te, langs, dlangs, **k):
        lg = dlangs if key == "detected" else langs
        a, e = M["A+"], M["E5"]
        out = np.zeros(len(te), int)
        for L in LANGS:
            trL, teL = tr[lg[tr] == L], te[lg[te] == L]
            if not len(teL):
                continue
            w = max(GRID, key=lambda w: macro_f1(y[trL], (w * a[trL] + (1 - w) * e[trL]).argmax(1)) if len(trL) else 0)
            out[np.searchsorted(te, teL)] = (w * a[teL] + (1 - w) * e[teL]).argmax(1)
        return out
    return f


def m_bias(M, y, tr, te, **k):
    lp = np.log(np.clip(0.5 * M["A+"] + 0.5 * M["E5"], 1e-9, 1))
    steps = np.round(np.arange(-0.3, 0.301, 0.05), 2)
    bn, bu = max(itertools.product(steps, steps), key=lambda b: macro_f1(y[tr], (lp[tr] + [b[0], b[1], 0]).argmax(1)))
    return (lp[te] + [bn, bu, 0]).argmax(1)


def features(M, val, dlangs):
    lg = [np.log(np.clip(M[n], 1e-6, 1)) for n in sorted(M)]
    L = np.stack([(dlangs == x).astype(float) for x in LANGS], 1)
    n_tok = np.log1p(val["cleaned_text"].astype(str).str.split().str.len().to_numpy(float))[:, None]
    conf = np.stack([M[n].max(1) for n in sorted(M)], 1)
    return np.hstack(lg + [L, n_tok, conf])


def m_stack_lr(M, y, tr, te, X, **k):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=1.0, max_iter=3000).fit(X[tr], y[tr]).predict(X[te])


def m_stack_hgb(M, y, tr, te, X, **k):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                          random_state=42).fit(X[tr], y[tr]).predict(X[te])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extra-e5", default=None, help="run id of a second E5 (validation predictions)")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(paths.MULTI / "data" / "validation.csv", encoding="utf-8-sig", low_memory=False)
    y = val["label"].to_numpy(int)
    langs = val["language_type"].to_numpy()
    dlangs = val["cleaned_text"].astype(str).map(detect_language).to_numpy()
    M = load_members(val, args.extra_e5)
    X = features(M, val, dlangs)
    methods = {"incumbent A+.5/E5.5": m_fixed, "global A+,E5": m_global(["A+", "E5"]),
               "global A,E5": m_global(["A", "E5"]), "global A,A+,E5": m_global(["A", "A+", "E5"]),
               "class-aware A+/E5": m_classaware, "lang-aware (detected) A+/E5": m_langaware("detected"),
               "lang-aware (dataset) A+/E5": m_langaware("dataset"), "class-bias on incumbent": m_bias,
               "stacking LR": m_stack_lr, "stacking HGB": m_stack_hgb}
    if "E5b" in M:
        methods.update({"global E5,E5b": m_global(["E5", "E5b"]), "global A+,E5,E5b": m_global(["A+", "E5", "E5b"]),
                        "global A+,E5b": m_global(["A+", "E5b"])})
    from sklearn.model_selection import StratifiedKFold
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    rows = []
    for name, f in methods.items():
        sc = []
        for tr, te in folds:
            pred = f(M, y, tr, te, langs=langs, dlangs=dlangs, X=X)
            sc.append(macro_f1(y[te], pred))
        rows.append({"method": name, "folds": sc, "cv_macro_f1": float(np.mean(sc)), "cv_std": float(np.std(sc))})
        print(f"{name:<32} CV macro F1 {np.mean(sc):.4f} (+/-{np.std(sc):.4f})", flush=True)
    inc = np.array(rows[0]["folds"])
    for r in rows:
        d = np.array(r["folds"]) - inc
        r.update({"mean_gain_vs_incumbent": float(d.mean()), "folds_better": int((d > 0).sum()),
                  "candidate": bool(r["method"] != rows[0]["method"] and d.mean() >= 0.005 and (d > 0).sum() >= 4)})
    df = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    df.drop(columns="folds").to_csv(OUT / "cv_results.csv", index=False)
    # temperature scaling (calibration only, labels unchanged): fitted on validation
    from scipy.optimize import minimize_scalar
    p = 0.5 * M["A+"] + 0.5 * M["E5"]
    lp = np.log(np.clip(p, 1e-12, 1))
    nll = lambda t: -np.mean((lp / t)[np.arange(len(y)), y] - np.log(np.exp(lp / t).sum(1)))
    T = float(minimize_scalar(nll, bounds=(0.25, 5), method="bounded").x)
    cand = df[df["candidate"]]["method"].tolist()
    (OUT / "candidates.json").write_text(json.dumps({"candidates": cand, "incumbent_cv_macro_f1": float(inc.mean()),
                                                     "temperature_incumbent_fitted_on_validation": T,
                                                     "rule": "mean fold gain >= 0.005 and better in >= 4/5 folds"},
                                                    indent=2), encoding="utf-8")
    print("\n" + df.drop(columns="folds").round(4).to_string(index=False))
    print("\nTest candidates (validation rule):", cand or "NONE - keep production")
    print(f"Temperature for the incumbent (calibration only, fitted on validation): T = {T:.3f}")


if __name__ == "__main__":
    main()
