"""Exact metric computation and per-split evaluation files, shared by the local pipeline and the Colab package."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from .evaluation import save_confusion_png
from .labels import CLASS_NAMES, ID2LABEL, LANGUAGE_TYPES

LABELS = [0, 1, 2]


def full_metrics(y, pred, lang: pd.Series | np.ndarray | None = None) -> dict:
    """Exact metrics from predictions: accuracy, macro/weighted P/R/F1, per-class P/R/F1/support, per-language."""
    y, pred = np.asarray(y, dtype=int), np.asarray(pred, dtype=int)
    out = {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred))}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y, pred, labels=LABELS, average=avg, zero_division=0)
        out.update({f"{avg}_precision": float(p), f"{avg}_recall": float(r), f"{avg}_f1": float(f)})
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    out["per_class"] = {ID2LABEL[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]),
                                      "support": int(s[i])} for i in LABELS}
    if lang is not None:
        lang = np.asarray(lang)
        out["per_language"] = {}
        for lt in LANGUAGE_TYPES:
            m = lang == lt
            if m.any():
                mp, mr, mf, _ = precision_recall_fscore_support(y[m], pred[m], labels=LABELS, average="macro",
                                                                zero_division=0)
                cp, cr, cf, _ = precision_recall_fscore_support(y[m], pred[m], labels=LABELS, zero_division=0)
                out["per_language"][lt] = {"n": int(m.sum()), "accuracy": float(accuracy_score(y[m], pred[m])),
                                           "macro_precision": float(mp), "macro_recall": float(mr),
                                           "macro_f1": float(mf),
                                           "per_class": {ID2LABEL[i]: {"precision": float(cp[i]), "recall": float(cr[i]),
                                                                       "f1": float(cf[i])} for i in LABELS}}
    return out


def save_eval_files(folder: Path, y, probs: np.ndarray, df: pd.DataFrame, split: str, title: str) -> dict:
    """predictions.csv / classification_report.json / confusion_matrix.csv/png for one split."""
    folder.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=int)
    pred = probs.argmax(1)
    sfx = "" if split == "validation" else f"_{split}"
    pr = pd.DataFrame({"row_id": df["row_id"].to_numpy(), "language_type": df["language_type"].to_numpy(),
                       "actual_label": [ID2LABEL[v] for v in y], "predicted_label": [ID2LABEL[v] for v in pred],
                       "confidence": probs.max(1)})
    for i, c in enumerate(CLASS_NAMES):
        pr[f"prob_{c}"] = probs[:, i]
    pr.to_csv(folder / f"predictions{sfx}.csv", index=False, encoding="utf-8-sig")
    (folder / f"classification_report{sfx}.json").write_text(json.dumps(
        classification_report(y, pred, labels=LABELS, target_names=CLASS_NAMES, zero_division=0, output_dict=True),
        indent=2), encoding="utf-8")
    cm = confusion_matrix(y, pred, labels=LABELS)
    pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_NAMES], columns=[f"pred_{c}" for c in CLASS_NAMES]) \
        .to_csv(folder / f"confusion_matrix{sfx}.csv", encoding="utf-8-sig")
    save_confusion_png(cm, f"{title}\n{split} confusion matrix", folder / f"confusion_matrix{sfx}.png")
    m = full_metrics(y, pred, df["language_type"])
    m["confusion_matrix"] = cm.tolist()
    (folder / f"{split}_metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    return m
