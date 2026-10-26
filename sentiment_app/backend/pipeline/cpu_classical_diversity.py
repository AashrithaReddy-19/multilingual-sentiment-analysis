"""CPU-trained classical models for ensemble diversity (train.csv only; outputs VALIDATION probabilities only).

    python -m pipeline.cpu_classical_diversity [name ...]

Models (A+ features: word (1,2) 20k + char_wb (3,5) TF-IDF, sublinear):
  two_stage      Neutral-vs-polar LR, then Negative-vs-Positive LR on polar training rows;
                 P(neu)=p1, P(neg)=(1-p1)*q, P(pos)=(1-p1)*(1-q)
  lang_specific  one A+-style LR per training language; at inference each text is routed by the app's language
                 detector (detect_language), exactly as the web app could do it
  linsvc_cal     LinearSVC (C=0.5, balanced) calibrated with 3-fold sigmoid calibration on train
Fitted models are saved (joblib) so a model is scored on test only if a validation-selected ensemble uses it.
"""
from __future__ import annotations

import sys
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
from sentiment_core.language import detect_language  # noqa: E402

OUT = paths.MULTI / "experiments" / "cpu_classical"
P = ["prob_Negative", "prob_Neutral", "prob_Positive"]
LANGS = ("english", "telugu", "code_mixed")


def feats():
    return FeatureUnion([("word", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), sublinear_tf=True,
                                                  token_pattern=r"(?u)\S+")),
                         ("char", TfidfVectorizer(analyzer="char_wb", min_df=2, ngram_range=(3, 5),
                                                  sublinear_tf=True))])


def lr():
    return LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)


class TwoStage:
    def fit(self, x, y):
        self.s1 = Pipeline([("f", feats()), ("c", lr())]).fit(x, (y == 1).astype(int))
        pol = y != 1
        self.s2 = Pipeline([("f", feats()), ("c", lr())]).fit(x[pol], (y[pol] == 2).astype(int))
        return self

    def predict_proba(self, x):
        p1 = self.s1.predict_proba(x)[:, 1]
        q = self.s2.predict_proba(x)[:, 0]                 # P(Negative | polar)
        return np.stack([(1 - p1) * q, p1, (1 - p1) * (1 - q)], 1)


class LangSpecific:
    def fit(self, x, y, lang):
        self.m = {L: Pipeline([("f", feats()), ("c", lr())]).fit(x[lang == L], y[lang == L]) for L in LANGS}
        return self

    def predict_proba(self, x):
        det = np.array([detect_language(t) for t in x])
        out = np.zeros((len(x), 3))
        for L in LANGS:
            m = det == L if L != "english" else ~np.isin(det, ("telugu", "code_mixed"))   # unknown -> English
            if m.any():
                out[m] = self.m[L].predict_proba(x[m])
        return out


def main(argv=None) -> None:
    names = (argv if argv is not None else sys.argv[1:]) or ["two_stage", "lang_specific", "linsvc_cal"]
    tr = pd.read_csv(paths.MULTI / "data" / "train.csv", encoding="utf-8-sig", low_memory=False)
    va = pd.read_csv(paths.MULTI / "data" / "validation.csv", encoding="utf-8-sig", low_memory=False)
    xt, yt, lt = tr["cleaned_text"].astype(str).to_numpy(), tr["label"].to_numpy(int), tr["language_type"].to_numpy()
    xv, yv = va["cleaned_text"].astype(str).to_numpy(), va["label"].to_numpy(int)
    for name in names:
        if name == "two_stage":
            m = TwoStage().fit(xt, yt)
        elif name == "lang_specific":
            m = LangSpecific().fit(xt, yt, lt)
        elif name == "linsvc_cal":
            m = Pipeline([("f", feats()), ("c", CalibratedClassifierCV(
                LinearSVC(C=0.5, class_weight="balanced", random_state=42), cv=3, method="sigmoid"))]).fit(xt, yt)
        else:
            raise SystemExit(name)
        pv = m.predict_proba(xv)
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(m, d / "model.joblib")
        pd.DataFrame({"row_id": va["row_id"], **{c: pv[:, i] for i, c in enumerate(P)}}).to_csv(
            d / "predictions.csv", index=False, encoding="utf-8-sig")
        mm = full_metrics(yv, pv.argmax(1), va["language_type"])
        print(f"{name:<14} VALIDATION acc {mm['accuracy']:.4f} macro F1 {mm['macro_f1']:.4f} | Neu F1 "
              f"{mm['per_class']['Neutral']['f1']:.4f} | EN/TE/CM "
              + "/".join(f"{mm['per_language'][l]['macro_f1']:.3f}" for l in LANGS), flush=True)


if __name__ == "__main__":
    main()
