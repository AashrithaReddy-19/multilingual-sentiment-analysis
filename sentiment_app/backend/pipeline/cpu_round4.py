"""CPU round 4: frozen-embedding, hybrid and specialist models (fit on train.csv; selection on VALIDATION only;
test.csv is never read here).

    python -m pipeline.cpu_round4 [features] [models]

features  caches (once): A+ word+char TF-IDF -> TruncatedSVD(512) fitted on train; numeric linguistic features
          (lengths, punctuation, emoji, uppercase ratio, Latin/Telugu-script ratios, repetition, negation count/
          position, token after negation hashed); detected-language one-hot (the app's detector, for train AND
          validation, as served). Frozen base-E5 embeddings come from cpu_frozen_embeddings.
models    E  = embeddings only, H = SVD(k) + embeddings + numeric + language one-hot (multi-view)
          emb_lr / emb_svc        LR (C grid) / LinearSVC (C grid, sigmoid-calibrated best) on E
          hyb_lr_k{128,256,512}   LR on H with k SVD dims
          hyb_mlp                 torch MLP on H (128/256/512 hidden, dropout, CE / weighted CE / focal), early stop
                                  on a 10% split of TRAIN (validation is only used to compare finished models)
          neutral_spec            Neutral-vs-polar LR on H -> 3-class with A+ polarity split
          two_stage_hyb           Neutral-vs-polar LR on H, then Negative-vs-Positive LR on H (polar train rows)
          en_spec / te_spec       hybrid LR trained on one language's train rows, applied to rows the detector
                                  assigns to that language; other rows keep A+ (full 3-class validation output)
          distill                 MLP student on H trained with 0.7 * CE(hard) + 0.3 * KL(soft A+ / E5 targets on
                                  TRAIN rows); soft targets come from models that saw these rows in training, so
                                  this is reported as an experiment, not a leakage-free teacher
Outputs: experiments/cpu_round4/NAME/predictions.csv (+ model state) and a summary CSV.
"""
from __future__ import annotations

import re
import sys
import time
import zlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402
from sentiment_core.experiment_io import full_metrics  # noqa: E402
from sentiment_core.language import detect_language  # noqa: E402

EXP = paths.MULTI / "experiments"
OUT = EXP / "cpu_round4"
CACHE = OUT / "_cache"
EMB = EXP / "cpu_embed"
P = ["prob_Negative", "prob_Neutral", "prob_Positive"]
LANGS = ("english", "telugu", "code_mixed")
NEG = re.compile(r"\b(not|no|never|nothing|nobody|ledu|kaadu|vaddu|kadu|ledhu)\b|n't", re.I)
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
TEL = re.compile("[ఀ-౿]")
ROWS = []


def load(split):
    return pd.read_csv(paths.MULTI / "data" / f"{split}.csv", encoding="utf-8-sig", low_memory=False)


def numeric(texts):
    out = []
    for t in texts:
        letters = [c for c in t if c.isalpha()]
        nl = max(1, len(letters))
        words = t.split()
        m = NEG.search(t)
        after = ""
        if m:
            rest = t[m.end():].split()
            after = rest[0].lower() if rest else ""
        h = np.zeros(16)
        if after:
            h[zlib.crc32(after.encode()) % 16] = 1.0
        out.append(np.concatenate([[np.log1p(len(t)), np.log1p(len(words)), t.count("!"), t.count("?"),
                                    len(EMOJI.findall(t)), sum(c.isupper() for c in letters) / nl,
                                    sum(c.isascii() for c in letters) / nl, len(TEL.findall(t)) / nl,
                                    len(re.findall(r"(.)\1\1", t)), len(NEG.findall(t)),
                                    (m.start() / max(1, len(t))) if m else -1.0], h]))
    return np.array(out, np.float32)


def lang_onehot(texts):
    d = np.array([detect_language(t) for t in texts])
    return np.stack([(d == x).astype(np.float32) for x in LANGS], 1), d


def features():
    from sklearn.decomposition import TruncatedSVD
    from pipeline.cpu_round3 import a_plus_feats
    CACHE.mkdir(parents=True, exist_ok=True)
    tr, va = load("train"), load("validation")
    t0 = time.time()
    fe = a_plus_feats().fit(tr["cleaned_text"].astype(str))
    Xt, Xv = fe.transform(tr["cleaned_text"].astype(str)), fe.transform(va["cleaned_text"].astype(str))
    svd = TruncatedSVD(512, random_state=42).fit(Xt)
    joblib.dump({"fe": fe, "svd": svd}, CACHE / "tfidf_svd.joblib")
    for split, X, df in (("train", Xt, tr), ("validation", Xv, va)):
        np.save(CACHE / f"svd512_{split}.npy", svd.transform(X).astype(np.float32))
        np.save(CACHE / f"num_{split}.npy", numeric(df["cleaned_text"].astype(str)))
        oh, d = lang_onehot(df["cleaned_text"].astype(str))
        np.save(CACHE / f"lang_{split}.npy", oh)
        np.save(CACHE / f"det_{split}.npy", d)
    print(f"features cached in {time.time() - t0:.0f}s; SVD explained variance {svd.explained_variance_ratio_.sum():.3f}")


class Hyb:
    """Standardised multi-view matrix [SVD(k) | embeddings | numeric | language one-hot]; stats from train only."""
    def __init__(self, k):
        self.k = k

    def raw(self, split):
        return np.hstack([np.load(CACHE / f"svd512_{split}.npy")[:, :self.k], np.load(EMB / f"{split}.npy"),
                          np.load(CACHE / f"num_{split}.npy"), np.load(CACHE / f"lang_{split}.npy")])

    def fit(self, split="train"):
        R = self.raw(split)
        self.mu, self.sd = R.mean(0), R.std(0) + 1e-6
        return (R - self.mu) / self.sd

    def tf(self, split):
        return (self.raw(split) - self.mu) / self.sd


def record(name, va, pv, note=""):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": va["row_id"], **{c: pv[:, i] for i, c in enumerate(P)}}).to_csv(
        d / "predictions.csv", index=False, encoding="utf-8-sig")
    m = full_metrics(va["label"].to_numpy(int), pv.argmax(1), va["language_type"])
    row = {"model": name, "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"],
           **{f"{c}_f1": m["per_class"][c]["f1"] for c in ("Negative", "Neutral", "Positive")},
           **{f"{l}_macro_f1": m["per_language"][l]["macro_f1"] for l in LANGS}, "note": note}
    ROWS.append(row)
    print(f"{name:<22} VAL acc {row['val_accuracy']:.4f} macro F1 {row['val_macro_f1']:.4f} | N/U/P "
          f"{row['Negative_f1']:.3f}/{row['Neutral_f1']:.3f}/{row['Positive_f1']:.3f} | EN/TE/CM "
          f"{row['english_macro_f1']:.3f}/{row['telugu_macro_f1']:.3f}/{row['code_mixed_macro_f1']:.3f} {note}",
          flush=True)


def best_lr(Xt, yt, Xv, yv, Cs, **kw):
    from sklearn.linear_model import LogisticRegression
    best = None
    for C in Cs:
        m = LogisticRegression(C=C, max_iter=4000, **kw).fit(Xt, yt)
        s = full_metrics(yv, m.predict(Xv))["macro_f1"]
        if best is None or s > best[0]:
            best = (s, m)
    return best[1]


def torch_mlp(Xt, yt, Xv, hidden, dropout, loss, epochs=40, soft=None, alpha=1.0, seed=42):
    """Small MLP; early stopping on a 10% TRAIN hold-out (macro F1). Returns validation probabilities."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(yt)); hold, fit = idx[: len(idx) // 10], idx[len(idx) // 10:]
    Xf, yf = torch.tensor(Xt[fit]), torch.tensor(yt[fit])
    Sf = torch.tensor(soft[fit]) if soft is not None else None
    net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
                              torch.nn.Linear(hidden, 3))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    cnt = np.bincount(yt, minlength=3); w = torch.tensor((cnt.sum() / (3 * cnt)).astype(np.float32))
    best, state, bad = -1, None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(yf))
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            logits = net(Xf[b])
            if loss == "focal":
                lp = torch.log_softmax(logits, -1).gather(1, yf[b][:, None]).squeeze(1)
                l = (-(1 - lp.exp()) ** 1.5 * lp).mean()
            else:
                l = torch.nn.functional.cross_entropy(logits, yf[b], weight=w if loss == "wce" else None)
            if Sf is not None:
                kl = torch.nn.functional.kl_div(torch.log_softmax(logits, -1), Sf[b], reduction="batchmean")
                l = alpha * l + (1 - alpha) * kl
            opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            s = full_metrics(yt[hold], net(torch.tensor(Xt[hold])).argmax(1).numpy())["macro_f1"]
        if s > best:
            best, state, bad = s, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 5:
                break
    net.load_state_dict(state)
    with torch.no_grad():
        return torch.softmax(net(torch.tensor(Xv)), -1).numpy(), best


def models():
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    tr, va = load("train"), load("validation")
    yt, yv = tr["label"].to_numpy(int), va["label"].to_numpy(int)
    det_v = np.load(CACHE / "det_validation.npy", allow_pickle=True)
    det_t = np.load(CACHE / "det_train.npy", allow_pickle=True)
    Et, Ev = np.load(EMB / "train.npy"), np.load(EMB / "validation.npy")
    a_plus_v = pd.read_csv(EXP / "s0_a_plus_reproduced" / "predictions.csv", encoding="utf-8-sig")[P].to_numpy()

    m = best_lr(Et, yt, Ev, yv, (0.25, 1, 4, 16), class_weight="balanced")
    record("emb_lr", va, m.predict_proba(Ev), f"C={m.C}")
    best = None
    for C in (0.1, 0.5, 1, 3):
        s = full_metrics(yv, LinearSVC(C=C, class_weight="balanced").fit(Et, yt).decision_function(Ev).argmax(1))["macro_f1"]
        best = max(best or (s, C), (s, C))
    sv = CalibratedClassifierCV(LinearSVC(C=best[1], class_weight="balanced"), cv=3).fit(Et, yt)
    record("emb_svc", va, sv.predict_proba(Ev), f"C={best[1]}")

    hyb = {}
    for k in (128, 256, 512):
        h = Hyb(k); Ht = h.fit(); Hv = h.tf("validation")
        m = best_lr(Ht, yt, Hv, yv, (0.05, 0.25, 1), class_weight="balanced")
        record(f"hyb_lr_k{k}", va, m.predict_proba(Hv), f"C={m.C}")
        hyb[k] = (h, Ht, Hv, full_metrics(yv, m.predict(Hv))["macro_f1"])
    k = max(hyb, key=lambda k: hyb[k][3]); h, Ht, Hv, _ = hyb[k]
    best = None
    for hidden in (128, 256, 512):
        for dropout in (0.1, 0.3):
            for loss in ("ce", "wce", "focal"):
                pv, hold = torch_mlp(Ht, yt, Hv, hidden, dropout, loss)
                if best is None or hold > best[0]:              # chosen on the TRAIN hold-out, not validation
                    best = (hold, pv, f"k={k} h={hidden} d={dropout} {loss}")
    record("hyb_mlp", va, best[1], best[2])

    # neutral specialist and two-stage on the best hybrid view
    nm = LogisticRegression(C=0.25, max_iter=4000, class_weight="balanced").fit(Ht, (yt == 1).astype(int))
    pn = nm.predict_proba(Hv)[:, 1]
    pol = a_plus_v[:, [0, 2]] / a_plus_v[:, [0, 2]].sum(1, keepdims=True)
    record("neutral_spec", va, np.stack([(1 - pn) * pol[:, 0], pn, (1 - pn) * pol[:, 1]], 1),
           f"neutral-vs-polar val F1 {full_metrics((yv == 1).astype(int), (pn > 0.5).astype(int))['macro_f1']:.4f}")
    pm = yt != 1
    sm = LogisticRegression(C=0.25, max_iter=4000, class_weight="balanced").fit(Ht[pm], (yt[pm] == 2).astype(int))
    q = sm.predict_proba(Hv)[:, 1]
    record("two_stage_hyb", va, np.stack([(1 - pn) * (1 - q), pn, (1 - pn) * q], 1))

    # language specialists (detector routing as served; other rows keep A+)
    for L, name in (("english", "en_spec"), ("telugu", "te_spec")):
        mt = det_t == L
        ml = best_lr(Ht[mt], yt[mt], Hv[det_v == L], yv[det_v == L], (0.05, 0.25, 1), class_weight="balanced")
        pv = a_plus_v.copy()
        pv[det_v == L] = ml.predict_proba(Hv[det_v == L])
        record(name, va, pv, f"C={ml.C}, rows routed {int((det_v == L).sum())}")

    # distillation (soft targets of A+ on TRAIN rows, from the A+ model that was trained on them)
    ap = joblib.load(paths.MULTI / "model_a_plus" / "model_a_plus.joblib")
    soft = ap.predict_proba(tr["cleaned_text"].astype(str))[:, np.argsort(ap.classes_)].astype(np.float32)
    pv, hold = torch_mlp(Ht, yt, Hv, 256, 0.3, "ce", soft=soft, alpha=0.7)
    record("distill", va, pv, "alpha 0.7, A+ soft targets")
    pd.DataFrame(ROWS).sort_values("val_macro_f1", ascending=False).to_csv(OUT / "summary.csv", index=False)


def main(argv=None) -> None:
    steps = (argv if argv is not None else sys.argv[1:]) or ["features", "models"]
    OUT.mkdir(parents=True, exist_ok=True)
    for s in steps:
        globals()[s]()


if __name__ == "__main__":
    main()
