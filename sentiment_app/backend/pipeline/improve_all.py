"""Per-model improvement round (CPU only). Selection on VALIDATION; test scored once per frozen new variant.

    python -m pipeline.improve_all inventory     # every trained model: validation / test metrics, gaps to 75%
    python -m pipeline.improve_all postproc      # temperature / Neutral multiplier / class bias per probability model
    python -m pipeline.improve_all dual_e5       # old vs new E5: disagreement, oracle, fine blend
    python -m pipeline.improve_all prod_refine   # local refinement of production weights (language and language x class)
    python -m pipeline.improve_all test_frozen   # scores frozen variants once (only those that passed validation)

Rules: a post-processing / blend variant becomes a NEW named model only if it beats its source on validation by
>= 0.003 macro F1 (mean of 5 validation CV folds, parameters refitted per fold) AND is better in >= 4/5 folds.
Production challengers additionally need no class/language F1 drop > 0.01 and a significant test comparison.
Outputs: experiments/improve_all/*.json|csv. Original models and their metrics are never modified.
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
from pipeline import lang_aware_v2 as la  # noqa: E402
from sentiment_core.experiment_io import full_metrics  # noqa: E402

EXP = paths.MULTI / "experiments"
OUT = EXP / "improve_all"
P = ["prob_Negative", "prob_Neutral", "prob_Positive"]
LANGS = ("english", "telugu", "code_mixed")
TEL = paths.NLP_ROOT / "outputs"
f1 = la.f1


def rd(p):
    return pd.read_csv(p, encoding="utf-8-sig", low_memory=False)


def joblib_probs(model_path, texts):
    import joblib
    m = joblib.load(model_path)
    return m.predict_proba(texts)[:, np.argsort(m.classes_)]


def multi_members(split):
    """validation / test probabilities of the multilingual single models (saved, or recomputed for A)."""
    d = rd(paths.MULTI / "data" / f"{split}.csv")
    M = la.probs(split)                                                        # A+, E5 (old), E5new
    cache = OUT / f"A_{split}.npy"
    if not cache.exists():
        OUT.mkdir(parents=True, exist_ok=True)
        np.save(cache, joblib_probs(paths.MULTI / "model_a" / "tfidf_logreg_multilingual.joblib",
                                    d["cleaned_text"].astype(str)))
    M["A"] = np.load(cache)
    return d, M


def telugu_members(split):
    d = rd(TEL / f"{split}.csv")
    out = {}
    if split == "validation":
        out["TelB"] = rd(paths.APP_ROOT / "artifacts" / "telugu_only_bias" / "validation_probs_xlmr_telugu.csv")[P].to_numpy()
    else:
        out["TelB"] = rd(TEL / "predictions_no_context.csv")[P].to_numpy()
    out["TelA"] = joblib_probs(TEL / "baseline_tfidf_logreg.joblib", d["cleaned_text"].astype(str))
    out["TelA+"] = joblib_probs(paths.APP_ROOT / "artifacts" / "telugu_only_refine" / "model.joblib", d["cleaned_text"].astype(str))
    return d, out


def summary(y, p, langs=None):
    m = full_metrics(y, p.argmax(1), langs)
    r = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
         "macro_precision": m["macro_precision"], "macro_recall": m["macro_recall"],
         **{f"{c}_f1": m["per_class"][c]["f1"] for c in ("Negative", "Neutral", "Positive")}}
    if langs is not None:
        r.update({f"{l}_macro_f1": m["per_language"][l]["macro_f1"] for l in LANGS if l in m["per_language"]})
    return r


def ladder(v):
    return " ".join(f"{t}:{'Y' if v >= t / 100 else '-'}" for t in (70, 72, 73, 74, 75, 76, 77))


# ------------------------------------------------------------------ inventory
def inventory():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    va, Mv = multi_members("validation")
    te, Mt = multi_members("test")
    yv, yt = va["label"].to_numpy(int), te["label"].to_numpy(int)
    Rv, _ = la.routers(va["cleaned_text"].astype(str).tolist()); Rt, _ = la.routers(te["cleaned_text"].astype(str).tolist())
    w = json.loads((EXP / "lang_aware_v2" / "config.json").read_text())["weights_by_language"]
    ens = json.loads((EXP / "ensemble_validation_tuned" / "config.json").read_text())
    ew = dict(zip(ens["member_ids"], ens["weights"]))
    comb = {
        "Previous Production (A+ .5 / E5 .5)": lambda M, R: 0.5 * M["A+"] + 0.5 * M["E5"],
        "Optimized Ensemble": lambda M, R: ew["model_a_plus"] * M["A+"] + ew[la.OLD] * M["E5"] + ew[la.NEW] * M["E5new"],
        "Production (lang_aware_v2)": lambda M, R: la.lang_prob(M, R["learned"], np.arange(len(M["A+"])),
                                                               ("A+", "E5", "E5new"), w),
    }
    for name, key, arch in (("TF-IDF Baseline (A)", "A", "word TF-IDF + LR"), ("Enhanced TF-IDF (A+)", "A+", "word+char TF-IDF + LR"),
                            ("Multilingual E5 (earlier run)", "E5", "fine-tuned e5-small"), ("Multilingual E5", "E5new", "fine-tuned e5-small")):
        rows.append({"model": name, "architecture": arch, "split": "multilingual", "val_probs": True, "test_probs": True,
                     **{f"val_{k}": v for k, v in summary(yv, Mv[key], va["language_type"]).items()},
                     **{f"test_{k}": v for k, v in summary(yt, Mt[key], te["language_type"]).items()}})
    for name, fn in comb.items():
        rows.append({"model": name, "architecture": "ensemble", "split": "multilingual", "val_probs": True, "test_probs": True,
                     **{f"val_{k}": v for k, v in summary(yv, fn(Mv, Rv), va["language_type"]).items()},
                     **{f"test_{k}": v for k, v in summary(yt, fn(Mt, Rt), te["language_type"]).items()}})
    tv, Tv = telugu_members("validation"); tt, Tt = telugu_members("test")
    for name, key, arch in (("Telugu Model A", "TelA", "TF-IDF + LR"), ("Telugu Model A+", "TelA+", "char TF-IDF + LR"),
                            ("Telugu Model B", "TelB", "fine-tuned XLM-R")):
        rows.append({"model": name, "architecture": arch, "split": "telugu_only", "val_probs": True, "test_probs": True,
                     **{f"val_{k}": v for k, v in summary(tv["label"].to_numpy(int), Tv[key]).items()},
                     **{f"test_{k}": v for k, v in summary(tt["label"].to_numpy(int), Tt[key]).items()}})
    df = pd.DataFrame(rows)
    for k in ("accuracy", "macro_f1", "Neutral_f1"):
        df[f"gap_to_75_{k}"] = 0.75 - df[f"test_{k}"]
    df.to_csv(OUT / "inventory.csv", index=False)
    pd.set_option("display.width", 250)
    print(df[["model", "val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1", "test_Neutral_f1",
              "gap_to_75_accuracy", "gap_to_75_macro_f1"]].round(4).to_string(index=False))
    # oracle analysis on validation (is 75% reachable by any selection among the multilingual members?)
    keys = ["A", "A+", "E5", "E5new"]
    correct = np.stack([Mv[k].argmax(1) == yv for k in keys], 1)
    print("\nVALIDATION oracle accuracy (some member right):",
          {"+".join(c): round(float(correct[:, [keys.index(x) for x in c]].any(1).mean()), 4)
           for r in (2, 3, 4) for c in itertools.combinations(keys, r) if "E5" in c or "E5new" in c})
    tc = np.stack([Tv[k].argmax(1) == tv["label"].to_numpy(int) for k in ("TelA", "TelA+", "TelB")], 1)
    print("Telugu-only VALIDATION oracle accuracy (A, A+, B):", round(float(tc.any(1).mean()), 4),
          "| B + A+:", round(float(tc[:, [1, 2]].any(1).mean()), 4))


# ------------------------------------------------------------------ CV machinery
def folds(y, k=5):
    from sklearn.model_selection import StratifiedKFold
    return list(StratifiedKFold(k, shuffle=True, random_state=42).split(np.zeros(len(y)), y))


def cv_gain(y, source, fit, apply):
    """fit(idx) -> params ; apply(idx, params) -> probs. Returns mean fold gain vs source argmax, wins, per-fold."""
    g = []
    for tr, ho in folds(y):
        prm = fit(tr)
        g.append(f1(y[ho], apply(ho, prm).argmax(1)) - f1(y[ho], source[ho].argmax(1)))
    return float(np.mean(g)), int(sum(x > 0 for x in g)), [round(x, 4) for x in g]


def passes(gain, wins):
    return gain >= 0.003 and wins >= 4


def bias_fit(y, p):
    from pipeline.finalize_phase3 import fit_class_bias
    return fit_class_bias(y, p)[0]


def bias_apply(p, b):
    return np.log(np.clip(p, 1e-12, 1)) + np.asarray(b)


def softmax_rows(z):
    z = np.exp(z - z.max(1, keepdims=True)); return z / z.sum(1, keepdims=True)


def ece(y, p, bins=15):
    c, pr = p.max(1), p.argmax(1)
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        m = (c > lo) & (c <= lo + 1 / bins)
        if m.any():
            e += m.mean() * abs((pr[m] == y[m]).mean() - c[m].mean())
    return float(e)


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(la.jsonable(obj), indent=2), encoding="utf-8")


# ------------------------------------------------------------------ post-processing per probability model
def postproc():
    """class log-bias (covers the Neutral multiplier 0.67-1.49) per single model; temperature for calibration only
    (temperature never changes argmax of a single model, so it cannot change accuracy / F1)."""
    from pipeline.finalize_phase3 import fit_temperature, apply_temperature
    va, Mv = multi_members("validation"); yv = va["label"].to_numpy(int)
    tv, Tv = telugu_members("validation"); ytv = tv["label"].to_numpy(int)
    res = {}
    for key, M, y in [(k, Mv, yv) for k in ("A", "A+", "E5", "E5new")] + [(k, Tv, ytv) for k in ("TelA", "TelA+", "TelB")]:
        p = M[key]
        gain, wins, per = cv_gain(y, p, lambda i: bias_fit(y[i], p[i]), lambda i, b: bias_apply(p[i], b))
        T = fit_temperature(y, p)
        res[key] = {"val_macro_f1": f1(y, p.argmax(1)), "bias_full_val": bias_fit(y, p), "bias_cv_gain": gain,
                    "bias_cv_wins": wins, "bias_cv_folds": per, "bias_passes": passes(gain, wins),
                    "temperature": T, "ece_raw": ece(y, p), "ece_temperature": ece(y, apply_temperature(p, T))}
        print(f"{key:6} val F1 {res[key]['val_macro_f1']:.4f} | bias CV gain {gain:+.4f} wins {wins}/5 -> "
              f"{'PASS' if res[key]['bias_passes'] else 'no'} | T {T:.3f} ECE {res[key]['ece_raw']:.4f}->{res[key]['ece_temperature']:.4f}")
    save("postproc", res)


# ------------------------------------------------------------------ Dual E5
def dual_e5():
    va, M = multi_members("validation"); y = va["label"].to_numpy(int)
    a, b = M["E5"], M["E5new"]
    pa, pb = a.argmax(1), b.argmax(1)
    agree = pa == pb
    print(f"agreement {agree.mean():.4f} | acc when agree {(pa[agree] == y[agree]).mean():.4f} | "
          f"disagree: old right {(pa[~agree] == y[~agree]).mean():.4f}, new right {(pb[~agree] == y[~agree]).mean():.4f} | "
          f"oracle {((pa == y) | (pb == y)).mean():.4f}")
    W = np.round(np.arange(0, 1.0001, 0.025), 3)

    def fitw(i):
        return max(W, key=lambda w: f1(y[i], (w * a[i] + (1 - w) * b[i]).argmax(1)))
    full_w = fitw(np.arange(len(y)))
    g1 = cv_gain(y, b, fitw, lambda i, w: w * a[i] + (1 - w) * b[i])
    g2 = cv_gain(y, b, lambda i: (lambda w: (w, bias_fit(y[i], w * a[i] + (1 - w) * b[i])))(fitw(i)),
                 lambda i, prm: bias_apply(prm[0] * a[i] + (1 - prm[0]) * b[i], prm[1]))
    p = full_w * a + (1 - full_w) * b
    res = {"weight_old_e5": float(full_w), "val_macro_f1": f1(y, p.argmax(1)), "vs_new_e5_cv": g1, "with_bias_cv": g2,
           "passes": passes(*g1[:2]), "passes_with_bias": passes(*g2[:2])}
    print(f"Dual-E5 w_old {full_w} val F1 {res['val_macro_f1']:.4f} (new E5 {f1(y, pb):.4f}) | CV vs new E5 {g1[0]:+.4f} "
          f"{g1[1]}/5 | +bias {g2[0]:+.4f} {g2[1]}/5")
    save("dual_e5", res)


# ------------------------------------------------------------------ production local refinement
def prod_refine():
    va, M = multi_members("validation"); y = va["label"].to_numpy(int)
    R, _ = la.routers(va["cleaned_text"].astype(str).tolist()); route = R["learned"]
    names = ("A+", "E5", "E5new")
    w0 = json.loads((EXP / "lang_aware_v2" / "config.json").read_text())["weights_by_language"]
    allidx = np.arange(len(y))
    prod = la.lang_prob(M, route, allidx, names, w0)
    out = {}

    # (1) finer weights: 0.025 simplex within +-0.1 of production weights, per language
    def fine_w(i):
        w = {}
        for L in LANGS:
            j = i[route[i] == L]; base = np.array(w0[L])
            cands = [base]
            for d1 in np.arange(-0.1, 0.1001, 0.025):
                for d2 in np.arange(-0.1, 0.1001, 0.025):
                    v = base + np.array([d1, d2, -d1 - d2])
                    if (v >= -1e-9).all():
                        cands.append(np.clip(v, 0, 1))
            w[L] = list(max(cands, key=lambda v: f1(y[j], la.blend(M, names, v, j).argmax(1))))
        return w
    out["fine_weights"] = cv_gain(y, prod, fine_w, lambda i, w: la.lang_prob(M, route, i, names, w))

    # (2) language x class bias with shrinkage toward the global bias
    def lc_fit(i, lam):
        g = np.array(bias_fit(y[i], prod[i]))
        return {L: list(lam * np.array(bias_fit(y[i[route[i] == L]], prod[i[route[i] == L]])) + (1 - lam) * g) for L in LANGS}

    def lc_apply(i, b):
        z = np.log(np.clip(prod[i], 1e-12, 1))
        for L in LANGS:
            m = route[i] == L; z[m] += np.array(b[L])
        return z
    for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
        out[f"lang_class_bias_lam{lam}"] = cv_gain(y, prod, lambda i, lam=lam: lc_fit(i, lam), lc_apply)

    # (3) Neutral multiplier only (0.94-1.08)
    NM = np.round(np.arange(0.94, 1.0801, 0.01), 2)
    out["neutral_mult"] = cv_gain(y, prod, lambda i: max(NM, key=lambda m: f1(y[i], la.neutral(prod[i], m).argmax(1))),
                                  lambda i, m: la.neutral(prod[i], m))
    # (4) adding A (TF-IDF baseline) as a 4th member, per-language weight 0..0.2
    def add_a(i):
        return {L: max(np.arange(0, 0.2001, 0.025), key=lambda wa: f1(y[i[route[i] == L]],
                ((1 - wa) * prod[i[route[i] == L]] + wa * M["A"][i[route[i] == L]]).argmax(1))) for L in LANGS}

    def add_a_apply(i, wa):
        p = prod[i].copy()
        for L in LANGS:
            m = route[i] == L; p[m] = (1 - wa[L]) * p[m] + wa[L] * M["A"][i][m]
        return p
    out["add_model_A"] = cv_gain(y, prod, add_a, add_a_apply)
    for k, (g, w, per) in out.items():
        print(f"{k:24} CV gain {g:+.4f} wins {w}/5 {per} -> {'PASS' if passes(g, w) else 'no'}")
    save("prod_refine", {k: {"gain": v[0], "wins": v[1], "folds": v[2], "passes": passes(v[0], v[1])} for k, v in out.items()})


# ------------------------------------------------------------------ Telugu-only
def tel_leakage():
    """overlap of Telugu-only validation/test texts with the MULTILINGUAL training split (decides whether multilingual
    models may be evaluated / mixed on the Telugu-only split)."""
    mtr = set(rd(paths.MULTI / "data" / "train.csv")["cleaned_text"].astype(str).str.strip())
    res = {}
    for s in ("validation", "test"):
        t = rd(TEL / f"{s}.csv")["cleaned_text"].astype(str).str.strip()
        res[s] = {"rows": int(len(t)), "in_multilingual_train": int(t.isin(mtr).sum()), "share": float(t.isin(mtr).mean())}
    print(res); save("telugu_leakage", res)
    return res


def tel_ensembles():
    tv, V = telugu_members("validation"); y = tv["label"].to_numpy(int)
    W = np.round(np.arange(0, 1.0001, 0.05), 2)
    out = {}
    for other in ("TelA", "TelA+"):
        a, b = V[other], V["TelB"]
        fit = lambda i, a=a, b=b: max(W, key=lambda w: f1(y[i], (w * a[i] + (1 - w) * b[i]).argmax(1)))
        g = cv_gain(y, b, fit, lambda i, w, a=a, b=b: w * a[i] + (1 - w) * b[i])
        wf = fit(np.arange(len(y)))
        out[f"TelB+{other}"] = {"weight_other": float(wf), "val_macro_f1": f1(y, (wf * a + (1 - wf) * b).argmax(1)),
                                "cv": g, "passes": passes(*g[:2])}
        g2 = cv_gain(y, b, lambda i, a=a, b=b: (lambda w: (w, bias_fit(y[i], w * a[i] + (1 - w) * b[i])))(fit(i)),
                     lambda i, prm, a=a, b=b: bias_apply(prm[0] * a[i] + (1 - prm[0]) * b[i], prm[1]))
        out[f"TelB+{other}+bias"] = {"cv": g2, "passes": passes(*g2[:2])}
    for k, v in out.items():
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()})
    save("telugu_ensembles", out)


def tel_embed():
    """frozen (not fine-tuned) multilingual-e5-small mean-pooled embeddings of the Telugu-only split."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.set_num_threads(10)
    d = OUT / "tel_embed"; d.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    model = AutoModel.from_pretrained("intfloat/multilingual-e5-small").eval()
    for s in ("train", "validation", "test"):
        if (d / f"{s}.npy").exists():
            continue
        texts = rd(TEL / f"{s}.csv")["cleaned_text"].astype(str).tolist()
        order = np.argsort([len(t) for t in texts]); out = np.zeros((len(texts), 384), np.float32)
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                idx = order[i:i + 64]
                enc = tok(["query: " + texts[j] for j in idx], truncation=True, max_length=128, padding=True, return_tensors="pt")
                h = model(**enc).last_hidden_state; m = enc["attention_mask"].unsqueeze(-1).float()
                out[idx] = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).numpy()
        np.save(d / f"{s}.npy", out); print(s, "embedded", out.shape, flush=True)


def tel_models():
    """new Telugu-only architectures, trained on the Telugu-only TRAIN split, selected on its VALIDATION split."""
    from scipy.sparse import hstack, csr_matrix
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import FeatureUnion
    from sklearn.svm import LinearSVC
    tr, va = rd(TEL / "train.csv"), rd(TEL / "validation.csv")
    Xt, Xv = tr["cleaned_text"].astype(str), va["cleaned_text"].astype(str)
    yt, yv = tr["label"].to_numpy(int), va["label"].to_numpy(int)
    E = {s: np.load(OUT / "tel_embed" / f"{s}.npy") for s in ("train", "validation")}
    feats = {
        "char25": lambda: TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
        "char_wb25": lambda: TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
        "char36": lambda: TfidfVectorizer(analyzer="char", ngram_range=(3, 6), sublinear_tf=True, min_df=2, max_features=400000),
        "word12+char25": lambda: FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
                                                ("c", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True, min_df=2))]),
    }
    res, probs = {}, {}
    cache = {}
    for fn, mk in feats.items():
        v = mk(); cache[fn] = (v.fit_transform(Xt), v.transform(Xv))
    clfs = {"lr": [(f"C{c}", lambda c=c: LogisticRegression(C=c, max_iter=3000, class_weight="balanced")) for c in (1, 2, 4, 8)],
            "svc": [(f"C{c}", lambda c=c: CalibratedClassifierCV(LinearSVC(C=c, class_weight="balanced"), cv=5)) for c in (0.05, 0.1, 0.3)],
            "sgd": [(f"a{a}", lambda a=a: CalibratedClassifierCV(SGDClassifier(loss="modified_huber", alpha=a, class_weight="balanced",
                                                                             random_state=42), cv=5)) for a in (1e-5, 3e-5)]}
    for fn, (A, B) in cache.items():
        for cn, grid in clfs.items():
            for pn, mk in grid:
                name = f"{fn}|{cn}|{pn}"
                p = mk().fit(A, yt).predict_proba(B); probs[name] = p
                res[name] = f1(yv, p.argmax(1)); print(f"{name:32} val F1 {res[name]:.4f}", flush=True)
    for c in (1, 4, 16):
        name = f"e5emb|lr|C{c}"; p = LogisticRegression(C=c, max_iter=3000, class_weight="balanced").fit(E["train"], yt).predict_proba(E["validation"])
        probs[name] = p; res[name] = f1(yv, p.argmax(1)); print(f"{name:32} val F1 {res[name]:.4f}", flush=True)
    for h in (256, 512):
        name = f"e5emb|mlp|h{h}"
        p = MLPClassifier((h,), alpha=1e-3, early_stopping=True, max_iter=200, random_state=42).fit(E["train"], yt).predict_proba(E["validation"])
        probs[name] = p; res[name] = f1(yv, p.argmax(1)); print(f"{name:32} val F1 {res[name]:.4f}", flush=True)
    A, B = cache["char25"]
    for s in (1.0, 2.0):
        name = f"hybrid char25+e5emb x{s}|lr|C2"
        p = LogisticRegression(C=2, max_iter=3000, class_weight="balanced").fit(
            hstack([A, csr_matrix(E["train"] * s)]).tocsr(), yt).predict_proba(hstack([B, csr_matrix(E["validation"] * s)]).tocsr())
        probs[name] = p; res[name] = f1(yv, p.argmax(1)); print(f"{name:32} val F1 {res[name]:.4f}", flush=True)
    d = OUT / "tel_models"; d.mkdir(parents=True, exist_ok=True)
    for k, p in probs.items():
        np.save(d / (k.replace("|", "__").replace(" ", "_").replace("+", "p") + ".npy"), p)
    best = sorted(res.items(), key=lambda kv: -kv[1])
    print("\nTOP 5:", [(k, round(v, 4)) for k, v in best[:5]], "| reference Telugu A+ 0.6684, Telugu B 0.7126")
    # complementarity with Telugu B (CV-checked blend)
    _, V = telugu_members("validation"); b = V["TelB"]
    W = np.round(np.arange(0, 1.0001, 0.05), 2); blends = {}
    for k, _ in best[:6]:
        a = probs[k]
        fit = lambda i, a=a: max(W, key=lambda w: f1(yv[i], (w * a[i] + (1 - w) * b[i]).argmax(1)))
        g = cv_gain(yv, b, fit, lambda i, w, a=a: w * a[i] + (1 - w) * b[i])
        wf = fit(np.arange(len(yv)))
        blends[k] = {"w_other": float(wf), "val_macro_f1": f1(yv, (wf * a + (1 - wf) * b).argmax(1)), "cv_vs_B": g, "passes": passes(*g[:2])}
        print(f"B + {k:30} w {wf} val {blends[k]['val_macro_f1']:.4f} CV {g[0]:+.4f} {g[1]}/5 -> {'PASS' if blends[k]['passes'] else 'no'}")
    save("telugu_models", {"val_macro_f1": res, "blends_with_B": blends})


TEL_OUT = paths.APP_ROOT / "artifacts" / "telugu_only_improve"


def tel_test_frozen():
    """freeze the validation-selected Telugu-only finalists, then score each ONCE on the Telugu-only test split."""
    import joblib
    from scipy.sparse import hstack, csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from pipeline.finalize_phase3 import bootstrap, mcnemar
    from sentiment_core.evaluation import evaluate_and_save
    TEL_OUT.mkdir(parents=True, exist_ok=True)
    sel = json.loads((OUT / "telugu_models.json").read_text(encoding="utf-8"))
    frozen = {"hybrid": {"name": "Telugu Hybrid (char TF-IDF + E5 embeddings)", "features": "char(2,5) TF-IDF + frozen e5-small x1.0",
                         "clf": "LogisticRegression C=2 balanced", "val_macro_f1": sel["val_macro_f1"]["hybrid char25+e5emb x1.0|lr|C2"]},
              "ensemble": {"name": "Telugu Ensemble (Model B + char TF-IDF)", "member": "char(2,5) TF-IDF LR C=2",
                           "w_char_tfidf": sel["blends_with_B"]["char25|lr|C2"]["w_other"],
                           "val_macro_f1": sel["blends_with_B"]["char25|lr|C2"]["val_macro_f1"],
                           "cv_vs_B": sel["blends_with_B"]["char25|lr|C2"]["cv_vs_B"]}}
    (TEL_OUT / "frozen.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    tr, te = rd(TEL / "train.csv"), rd(TEL / "test.csv")
    Et, Ee = np.load(OUT / "tel_embed" / "train.npy"), np.load(OUT / "tel_embed" / "test.npy")
    yt, ye = tr["label"].to_numpy(int), te["label"].to_numpy(int)
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True, min_df=2)
    A, B = vec.fit_transform(tr["cleaned_text"].astype(str)), vec.transform(te["cleaned_text"].astype(str))
    hyb = LogisticRegression(C=2, max_iter=3000, class_weight="balanced").fit(hstack([A, csr_matrix(Et)]).tocsr(), yt)
    p_h = hyb.predict_proba(hstack([B, csr_matrix(Ee)]).tocsr())
    ch = LogisticRegression(C=2, max_iter=3000, class_weight="balanced").fit(A, yt)
    _, T = telugu_members("test")
    w = frozen["ensemble"]["w_char_tfidf"]
    p_e = w * ch.predict_proba(B) + (1 - w) * T["TelB"]
    joblib.dump({"vectorizer": vec, "hybrid": hyb, "char_lr": ch}, TEL_OUT / "models.joblib")
    out = {}
    for key, p, ref, tag in (("hybrid", p_h, "TelA+", "telugu_hybrid"), ("ensemble", p_e, "TelB", "telugu_ensemble")):
        m = evaluate_and_save(tag=tag, model_name=frozen[key]["name"], input_col="cleaned_text", test_df=te, probs=p,
                              out_dir=TEL_OUT / key, best_val_macro_f1=frozen[key]["val_macro_f1"])
        out[key] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "reference": ref,
                    "reference_macro_f1": f1(ye, T[ref].argmax(1)),
                    "mcnemar": mcnemar(ye, T[ref].argmax(1), p.argmax(1)), "bootstrap": bootstrap(ye, T[ref].argmax(1), p.argmax(1))}
        print(f"{frozen[key]['name']}: TEST acc {m['accuracy']:.4f} F1 {m['macro_f1']:.4f} | vs {ref} {out[key]['reference_macro_f1']:.4f} "
              f"McNemar p {out[key]['mcnemar']['p_value']:.3g} CI {np.round(out[key]['bootstrap']['macro_f1_diff_ci95'], 4)}")
    (TEL_OUT / "test_result.json").write_text(json.dumps(la.jsonable(out), indent=2), encoding="utf-8")


if __name__ == "__main__":
    globals()[sys.argv[1]]()
