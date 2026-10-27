"""Language-aware ensemble, round 2: selection on VALIDATION with the language router in the loop.

    python -m pipeline.lang_aware_v2 select     # validation only (test is not read)
    python -m pipeline.lang_aware_v2 test       # scores the FROZEN choice once on test + statistics vs production

Members (saved probabilities): A+, old E5 (production member), new E5 (e5_lr3e-05 family winner).
Routers: the app's heuristic detect_language, and a learned char TF-IDF + LR language detector trained on TRAIN
language labels (with its probability for confidence routing).
Candidates (each refitted inside every CV fold for the stability check):
  production                fixed A+ 0.5 / old E5 0.5
  lang2_<router>            per-language E5 weight in 0.35..0.70 (0.025 steps), A+ the rest; coordinate search
  lang3_<router>            per-language weights over A+ / old E5 / new E5 (0.05 simplex); coordinate search
  + neutral bias            Neutral probability multiplier 0.97..1.05 on top of the best language-aware candidate
  blend P/E-lang            production and lang3 probabilities mixed 0.25 / 0.5 / 0.75
  confroute_<t>             learned router: lang3 weights where detector confidence >= t, production otherwise
Decision rule (set by the user for this round): higher validation macro F1 than production, better in >= 3/5
validation CV folds, no language or class F1 drop > 0.01. The chosen configuration is frozen to
experiments/lang_aware_v2/frozen.json before test is read.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402
from sentiment_core.experiment_io import full_metrics  # noqa: E402
from sentiment_core.language import detect_language  # noqa: E402

EXP = paths.MULTI / "experiments"
OUT = EXP / "lang_aware_v2"
P = ["prob_Negative", "prob_Neutral", "prob_Positive"]
LANGS = ("english", "telugu", "code_mixed")
OLD, NEW = "e5_lr5e-05_ce_ep3p2_bs32_len128_cpu", "e5_lr3e-05_ce_ep3p2_bs32_len128_cpu"


def f1(y, p):
    f = []
    for k in range(3):
        tp = np.sum((p == k) & (y == k)); fp = np.sum((p == k) & (y != k)); fn = np.sum((p != k) & (y == k))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def probs(split):
    def rd(path):
        return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)[P].to_numpy(float)
    if split == "validation":
        ap = rd(EXP / "s0_a_plus_reproduced" / "predictions.csv")
        old = rd(EXP / "transformers" / OLD / "predictions.csv"); new = rd(EXP / "transformers" / NEW / "predictions.csv")
    else:
        ap = rd(paths.MULTI / "model_a_plus" / "predictions_model_a_plus.csv")
        old = rd(EXP / "transformers" / OLD / "predictions_test.csv"); new = rd(EXP / "transformers" / NEW / "predictions_test.csv")
    return {"A+": ap, "E5": old, "E5new": new}


def learned_detector():
    import joblib
    path = OUT / "lang_detector.joblib"
    if path.exists():
        return joblib.load(path)
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    tr = pd.read_csv(paths.MULTI / "data" / "train.csv", encoding="utf-8-sig", low_memory=False)
    m = Pipeline([("f", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=3, max_features=200000,
                                        sublinear_tf=True)),
                  ("c", LogisticRegression(C=4.0, max_iter=2000))]).fit(tr["cleaned_text"].astype(str),
                                                                       tr["language_type"].astype(str))
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(m, path)
    return m


def routers(texts):
    heur = np.array([detect_language(t) for t in texts])
    heur = np.where(np.isin(heur, LANGS), heur, "english")            # app: unknown -> English weights
    m = learned_detector()
    pr = m.predict_proba(texts)
    learned = np.array(m.classes_)[pr.argmax(1)]
    return {"heuristic": heur, "learned": learned}, pr.max(1)


def blend(M, names, w, idx):
    return sum(wi * M[n][idx] for wi, n in zip(w, names))


def lang_fit(M, y, route, idx, names, grid):
    """coordinate search of per-language weights maximising macro F1 over idx (2 rounds)."""
    w = {L: grid[len(grid) // 2] for L in LANGS}
    for _ in range(2):
        for L in LANGS:
            def score(g):
                ww = {**w, L: g}
                p = np.zeros((len(idx), 3))
                for LL in LANGS:
                    m = route[idx] == LL
                    p[m] = blend(M, names, ww[LL], idx[m])
                return f1(y[idx], p.argmax(1))
            w[L] = max(grid, key=score)
    return w


def lang_prob(M, route, idx, names, w):
    p = np.zeros((len(idx), 3))
    for L in LANGS:
        m = route[idx] == L
        p[m] = blend(M, names, w[L], idx[m])
    return p


GRID2 = [(round(1 - e, 3), e) for e in np.round(np.arange(0.35, 0.7001, 0.025), 3)]
GRID3 = [(a, b, round(1 - a - b, 2)) for a in np.round(np.arange(0.1, 0.91, 0.05), 2)
         for b in np.round(np.arange(0.0, 0.91, 0.05), 2) if 1 - a - b >= -1e-9]
NEU = (0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04, 1.05)


def neutral(p, m):
    q = p.copy(); q[:, 1] *= m
    return q / q.sum(1, keepdims=True)


def candidates(M, y, R, conf):
    """name -> (fit(idx) -> params, prob(idx, params))"""
    C = {"production": (lambda idx: None, lambda idx, _: 0.5 * M["A+"][idx] + 0.5 * M["E5"][idx])}
    for rn in ("heuristic", "learned"):
        r = R[rn]
        C[f"lang2_{rn}"] = (lambda idx, r=r: lang_fit(M, y, r, idx, ("A+", "E5"), GRID2),
                            lambda idx, w, r=r: lang_prob(M, r, idx, ("A+", "E5"), w))
        C[f"lang3_{rn}"] = (lambda idx, r=r: lang_fit(M, y, r, idx, ("A+", "E5", "E5new"), GRID3),
                            lambda idx, w, r=r: lang_prob(M, r, idx, ("A+", "E5", "E5new"), w))
    for base in ("lang2_heuristic", "lang3_heuristic"):
        bf, bp = C[base]

        def fit_n(idx, bf=bf, bp=bp):
            w = bf(idx)
            p = bp(idx, w)
            return w, max(NEU, key=lambda m: f1(y[idx], neutral(p, m).argmax(1)))
        C[f"{base}+neutral"] = (fit_n, lambda idx, prm, bp=bp: neutral(bp(idx, prm[0]), prm[1]))
    bf3, bp3 = C["lang3_heuristic"]

    def fit_b(idx):
        w = bf3(idx)
        prod, la = 0.5 * M["A+"][idx] + 0.5 * M["E5"][idx], bp3(idx, w)
        return w, max((0.25, 0.5, 0.75), key=lambda a: f1(y[idx], (a * prod + (1 - a) * la).argmax(1)))
    C["blend P/E-lang"] = (fit_b, lambda idx, prm: prm[1] * (0.5 * M["A+"][idx] + 0.5 * M["E5"][idx])
                           + (1 - prm[1]) * bp3(idx, prm[0]))
    bfl, bpl = C["lang3_learned"]

    def fit_c(idx):
        w = bfl(idx)
        la, prod = bpl(idx, w), 0.5 * M["A+"][idx] + 0.5 * M["E5"][idx]
        t = max((0.6, 0.7, 0.8, 0.9), key=lambda t: f1(y[idx], np.where((conf[idx] >= t)[:, None], la, prod).argmax(1)))
        return w, t
    C["confroute learned"] = (fit_c, lambda idx, prm: np.where((conf[idx] >= prm[1])[:, None], bpl(idx, prm[0]),
                                                               0.5 * M["A+"][idx] + 0.5 * M["E5"][idx]))
    return C


def detail(y, p, langs):
    m = full_metrics(y, p, langs)
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"], "weighted_f1": m["weighted_f1"],
            **{f"{c}_f1": m["per_class"][c]["f1"] for c in ("Negative", "Neutral", "Positive")},
            **{f"{l}_macro_f1": m["per_language"][l]["macro_f1"] for l in LANGS}}


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o.item() if hasattr(o, "item") else o


def select():
    from sklearn.model_selection import StratifiedKFold
    OUT.mkdir(parents=True, exist_ok=True)
    va = pd.read_csv(paths.MULTI / "data" / "validation.csv", encoding="utf-8-sig", low_memory=False)
    y, langs, texts = va["label"].to_numpy(int), va["language_type"].to_numpy(), va["cleaned_text"].astype(str).tolist()
    M = probs("validation")
    R, conf = routers(texts)
    acc = {k: float((v == langs).mean()) for k, v in R.items()}
    print("language-detection accuracy on validation (vs dataset labels):", {k: round(v, 4) for k, v in acc.items()})
    C = candidates(M, y, R, conf)
    allidx = np.arange(len(y))
    rows, fitted = [], {}
    prod_d = detail(y, C["production"][1](allidx, None).argmax(1), langs)
    for name, (fit, prob) in C.items():
        prm = fit(allidx)
        fitted[name] = prm
        d = detail(y, prob(allidx, prm).argmax(1), langs)
        rows.append({"candidate": name, **d})
        print(f"{name:<28} VAL acc {d['accuracy']:.4f} F1 {d['macro_f1']:.4f} | N/U/P {d['Negative_f1']:.3f}/"
              f"{d['Neutral_f1']:.3f}/{d['Positive_f1']:.3f} | EN/TE/CM {d['english_macro_f1']:.3f}/"
              f"{d['telugu_macro_f1']:.3f}/{d['code_mixed_macro_f1']:.3f}", flush=True)
    df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    top = [c for c in df["candidate"] if c != "production"][:3]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y))
    stab = {}
    for name in ["production"] + top:
        fit, prob = C[name]
        sc = [f1(y[te], prob(te, fit(tr)).argmax(1)) for tr, te in folds]
        stab[name] = sc
    print("\n5-fold CV inside validation (fit on 4 folds, score the 5th):")
    res = []
    for name in top:
        d = np.array(stab[name]) - np.array(stab["production"])
        full = df.set_index("candidate").loc[name]
        drops = {k: float(full[k] - prod_d[k]) for k in prod_d if k.endswith("_f1") and k != "macro_f1"
                 and k != "weighted_f1"}
        ok = bool(full["macro_f1"] > prod_d["macro_f1"] and (d > 0).sum() >= 3 and min(drops.values()) > -0.01)
        res.append({"candidate": name, "fold_scores": stab[name], "prod_fold_scores": stab["production"],
                    "cv_mean": float(np.mean(stab[name])), "cv_std": float(np.std(stab[name])),
                    "mean_gain": float(d.mean()), "folds_better": int((d > 0).sum()),
                    "largest_class_or_language_drop": min(drops.values()), "passes_rule": ok})
        print(f"  {name:<28} CV {np.mean(stab[name]):.4f} +/-{np.std(stab[name]):.4f} gain {d.mean():+.4f} "
              f"better {int((d > 0).sum())}/5 | largest drop {min(drops.values()):+.4f} -> {'PASS' if ok else 'fail'}")
    df.to_csv(OUT / "validation_candidates.csv", index=False)
    passing = [r for r in res if r["passes_rule"]]
    # simplicity tie-break (decided before test): among passing candidates within 0.001 validation macro F1 of the
    # best, take the least complex (lang2 < lang3 < lang3 + neutral bias / blend / confidence routing)
    cost = lambda n: 0 if n.startswith("lang2") and "+" not in n else 1 if n.startswith("lang3") and "+" not in n else 2
    vf = lambda r: df.set_index("candidate").loc[r["candidate"], "macro_f1"]
    choice = None
    if passing:
        best = max(vf(r) for r in passing)
        choice = min([r for r in passing if best - vf(r) < 0.001], key=lambda r: (cost(r["candidate"]), -vf(r)))
    frozen = {"rule": "val macro F1 > production, better in >= 3/5 validation CV folds, no class/language F1 drop > 0.01",
              "language_detection_accuracy_validation": acc, "stability": res,
              "choice": choice["candidate"] if choice else None,
              "params": jsonable(fitted[choice["candidate"]]) if choice else None,
              "validation": jsonable(df.set_index("candidate").loc[choice["candidate"]].to_dict()) if choice else None,
              "production_validation": prod_d}
    (OUT / "frozen.json").write_text(json.dumps(jsonable(frozen), indent=2), encoding="utf-8")
    print("\nFROZEN CHOICE:", frozen["choice"], frozen["params"])


def test():
    from pipeline.finalize_phase3 import bootstrap, mcnemar
    fr = json.loads((OUT / "frozen.json").read_text(encoding="utf-8"))
    if not fr["choice"]:
        raise SystemExit("No validation-selected candidate; test not read.")
    te = pd.read_csv(paths.MULTI / "data" / "test.csv", encoding="utf-8-sig", low_memory=False)
    y, langs, texts = te["label"].to_numpy(int), te["language_type"].to_numpy(), te["cleaned_text"].astype(str).tolist()
    M = probs("test")
    R, conf = routers(texts)
    C = candidates(M, y, R, conf)
    name, prm = fr["choice"], fr["params"]
    if isinstance(prm, list) and prm and isinstance(prm[0], dict):
        prm = (prm[0], prm[1])
    idx = np.arange(len(y))
    p = C[name][1](idx, prm)
    prod = C["production"][1](idx, None)
    d, dp = detail(y, p.argmax(1), langs), detail(y, prod.argmax(1), langs)
    st = {"mcnemar": mcnemar(y, prod.argmax(1), p.argmax(1)), "bootstrap": bootstrap(y, prod.argmax(1), p.argmax(1))}
    out = {"candidate": name, "params": prm, "test": d, "production_test": dp, "vs_production": st}
    (OUT / "test_result.json").write_text(json.dumps(jsonable(out), indent=2), encoding="utf-8")
    pd.DataFrame({"row_id": te["row_id"], **{c: p[:, i] for i, c in enumerate(P)}}).to_csv(
        OUT / "predictions_test.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(jsonable({"test": d, "mcnemar_p": st["mcnemar"]["p_value"],
                               "acc_diff_ci": st["bootstrap"]["accuracy_diff_ci95"],
                               "f1_diff_ci": st["bootstrap"]["macro_f1_diff_ci95"]}), indent=1))


if __name__ == "__main__":
    globals()[sys.argv[1]]()
