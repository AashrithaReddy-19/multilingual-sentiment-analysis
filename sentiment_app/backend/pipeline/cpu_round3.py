"""CPU round 3: new model types (train.csv only for fitting; selection on VALIDATION only; test never read here).

    python -m pipeline.cpu_round3 NAME [NAME ...]
NAMES
  svc_search   compact LinearSVC search over NEW feature ranges (word (1,3), char (2,6), char analyzer, binary tf);
               the best config by validation macro F1 is refitted with sigmoid calibration -> probabilities
  hard_aplus   A+ features/classifier with training-only hard-example weights: 3-fold CV on train gives each
               training row's out-of-fold probability of its own label p; weight = 1 + (1 - p)
  emb_lr       logistic regression on frozen base-E5 embeddings (C selected on validation)
  emb_mlp      small MLP (256 or 512 hidden, ReLU, early stopping on a 10% TRAIN split) on frozen embeddings
  svd_concat   A+ TF-IDF -> TruncatedSVD(256, fitted on train) concatenated with frozen embeddings -> LR
Each writes experiments/cpu_round3/NAME/predictions.csv (validation probabilities) and model.joblib.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402
from sentiment_core.experiment_io import full_metrics  # noqa: E402

OUT = paths.MULTI / "experiments" / "cpu_round3"
EMB = paths.MULTI / "experiments" / "cpu_embed"
P = ["prob_Negative", "prob_Neutral", "prob_Positive"]
LANGS = ("english", "telugu", "code_mixed")


def load(split):
    return pd.read_csv(paths.MULTI / "data" / f"{split}.csv", encoding="utf-8-sig", low_memory=False)


def a_plus_feats():
    return FeatureUnion([("word", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True,
                                                  token_pattern=r"(?u)\S+")),
                         ("char", TfidfVectorizer(analyzer="char_wb", min_df=2, ngram_range=(3, 5), sublinear_tf=True))])


def report(name, va, pv, extra=""):
    m = full_metrics(va["label"].to_numpy(int), pv.argmax(1), va["language_type"])
    print(f"{name:<28} VAL acc {m['accuracy']:.4f} macro F1 {m['macro_f1']:.4f} | N/U/P "
          + "/".join(f"{m['per_class'][c]['f1']:.3f}" for c in ("Negative", "Neutral", "Positive"))
          + " | EN/TE/CM " + "/".join(f"{m['per_language'][l]['macro_f1']:.3f}" for l in LANGS) + f" {extra}", flush=True)
    return m["macro_f1"]


def save(name, model, va, pv):
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, d / "model.joblib")
    pd.DataFrame({"row_id": va["row_id"], **{c: pv[:, i] for i, c in enumerate(P)}}).to_csv(
        d / "predictions.csv", index=False, encoding="utf-8-sig")


def svc_search(tr, va):
    xt, yt, xv, yv = tr["cleaned_text"].astype(str), tr["label"].to_numpy(int), va["cleaned_text"].astype(str), \
        va["label"].to_numpy(int)
    configs = {
        "w13_cwb26": lambda: FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 3), max_features=60000, min_df=2,
                                                                 sublinear_tf=True, token_pattern=r"(?u)\S+")),
                                           ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 6), min_df=3,
                                                                 max_features=300000, sublinear_tf=True))]),
        "w12_c25": lambda: FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=2,
                                                               sublinear_tf=True, token_pattern=r"(?u)\S+")),
                                         ("c", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=3,
                                                               max_features=300000, sublinear_tf=True))]),
        "w12_cwb36_bin": lambda: FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), max_features=40000, min_df=2,
                                                                     binary=True, token_pattern=r"(?u)\S+")),
                                               ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=3,
                                                                     max_features=300000, binary=True))]),
    }
    best = (-1, None, None)
    for fname, fb in configs.items():
        t0 = time.time()
        fe = fb()
        Xt, Xv = fe.fit_transform(xt), None
        Xv = fe.transform(xv)
        for C in (0.1, 0.25, 0.5, 1.0):
            clf = LinearSVC(C=C, class_weight="balanced", random_state=42).fit(Xt, yt)
            s = full_metrics(yv, clf.decision_function(Xv).argmax(1))["macro_f1"]
            print(f"  svc {fname} C={C}: VAL macro F1 {s:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if s > best[0]:
                best = (s, fname, C)
    s, fname, C = best
    model = Pipeline([("f", configs[fname]()), ("c", CalibratedClassifierCV(
        LinearSVC(C=C, class_weight="balanced", random_state=42), cv=3, method="sigmoid"))]).fit(xt, yt)
    pv = model.predict_proba(xv)
    report("svc_search (calibrated best)", va, pv, f"[{fname}, C={C}]")
    save("svc_search", model, va, pv)


def hard_aplus(tr, va):
    from sklearn.model_selection import StratifiedKFold
    xt, yt = tr["cleaned_text"].astype(str).to_numpy(), tr["label"].to_numpy(int)
    p_true = np.zeros(len(yt))
    for k, (a, b) in enumerate(StratifiedKFold(3, shuffle=True, random_state=42).split(xt, yt)):
        m = Pipeline([("f", a_plus_feats()), ("c", LogisticRegression(class_weight="balanced", max_iter=1000,
                                                                       random_state=42))]).fit(xt[a], yt[a])
        p_true[b] = m.predict_proba(xt[b])[np.arange(len(b)), yt[b]]
        print(f"  hard_aplus fold {k + 1}/3 done", flush=True)
    w = 1.0 + (1.0 - p_true)
    model = Pipeline([("f", a_plus_feats()), ("c", LogisticRegression(class_weight="balanced", max_iter=1000,
                                                                       random_state=42))])
    model.fit(xt, yt, c__sample_weight=w)
    pv = model.predict_proba(va["cleaned_text"].astype(str))
    report("hard_aplus", va, pv, f"[mean weight {w.mean():.3f}]")
    save("hard_aplus", model, va, pv)


def emb(split):
    return np.load(EMB / f"{split}.npy")


def emb_lr(tr, va):
    Et, Ev, yt = emb("train"), emb("validation"), tr["label"].to_numpy(int)
    best = (-1, None)
    for C in (0.5, 2.0, 8.0, 32.0):
        m = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Et, yt)
        s = full_metrics(va["label"].to_numpy(int), m.predict(Ev))["macro_f1"]
        print(f"  emb_lr C={C}: VAL macro F1 {s:.4f}", flush=True)
        if s > best[0]:
            best = (s, m)
    pv = best[1].predict_proba(Ev)
    report("emb_lr", va, pv, f"[C={best[1].C}]")
    save("emb_lr", best[1], va, pv)


def emb_mlp(tr, va):
    from sklearn.neural_network import MLPClassifier
    Et, Ev, yt = emb("train"), emb("validation"), tr["label"].to_numpy(int)
    best = (-1, None)
    for h in (256, 512):
        for alpha in (1e-4, 1e-3):
            m = MLPClassifier(hidden_layer_sizes=(h,), alpha=alpha, early_stopping=True, validation_fraction=0.1,
                              max_iter=200, n_iter_no_change=8, random_state=42).fit(Et, yt)
            s = full_metrics(va["label"].to_numpy(int), m.predict(Ev))["macro_f1"]
            print(f"  emb_mlp h={h} alpha={alpha}: VAL macro F1 {s:.4f} (epochs {m.n_iter_})", flush=True)
            if s > best[0]:
                best = (s, m)
    pv = best[1].predict_proba(Ev)
    report("emb_mlp", va, pv, f"[{best[1].hidden_layer_sizes}, alpha={best[1].alpha}]")
    save("emb_mlp", best[1], va, pv)


class SvdConcat:
    def fit(self, texts, E, y, dim=256):
        from sklearn.decomposition import TruncatedSVD
        self.fe = a_plus_feats().fit(texts)
        self.svd = TruncatedSVD(dim, random_state=42).fit(self.fe.transform(texts))
        Z = self._z(texts, E, fit=True)
        self.clf = LogisticRegression(C=2.0, max_iter=4000, class_weight="balanced").fit(Z, y)
        return self

    def _z(self, texts, E, fit=False):
        S = self.svd.transform(self.fe.transform(texts))
        if fit:
            self.mu, self.sd = S.mean(0), S.std(0) + 1e-6
        return np.hstack([(S - self.mu) / self.sd, E * np.sqrt(E.shape[1])])

    def predict_proba(self, texts, E):
        return self.clf.predict_proba(self._z(texts, E))


def svd_concat(tr, va):
    m = SvdConcat().fit(tr["cleaned_text"].astype(str), emb("train"), tr["label"].to_numpy(int))
    pv = m.predict_proba(va["cleaned_text"].astype(str), emb("validation"))
    report("svd_concat (256 + emb)", va, pv)
    save("svd_concat", m, va, pv)


def main(argv=None) -> None:
    names = argv if argv is not None else sys.argv[1:]
    tr, va = load("train"), load("validation")
    for n in names:
        globals()[n](tr, va)


if __name__ == "__main__":
    main()
