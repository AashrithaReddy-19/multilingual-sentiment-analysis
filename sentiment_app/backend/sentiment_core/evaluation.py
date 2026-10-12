"""Evaluation shared by every trained model: metrics, per-language results, confusion matrices,
predictions with confidence, and misclassified examples tagged with HEURISTIC error reasons."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from .cleaning import NEGATIONS
from .labels import CLASS_NAMES, ID2LABEL, LANGUAGE_TYPES

HEURISTIC_NOTE = "HEURISTIC (rule-based guess, not ground truth)"
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿\U0001F1E6-\U0001F1FF]")
TELUGU_RE = re.compile("[ఀ-౿]")
LATIN_RE = re.compile("[A-Za-z]")
NEG_RE = re.compile(r"\b(" + "|".join(NEGATIONS) + r")\b|లేదు|కాదు|వద్దు|కాలేదు|లేను", re.IGNORECASE)
LABELS = [0, 1, 2]


def heuristic_error_reason(text: str, language_type: str, actual: int, predicted: int, confidence: float,
                           annotations: object = None, has_context: bool = False) -> str:
    reasons = []
    if confidence < 0.5:
        reasons.append("low confidence prediction")
    if NEG_RE.search(text):
        reasons.append("negation handling")
    if EMOJI_RE.search(text):
        reasons.append("emoji ambiguity")
    if language_type == "code_mixed" or (TELUGU_RE.search(text) and LATIN_RE.search(text)):
        reasons.append("mixed-language ambiguity")
    if len(text.split()) <= 3 and not has_context:
        reasons.append("insufficient conversational context")
    if isinstance(annotations, str) and len({a.split("-")[0].strip() for a in annotations.split("|")}) > 1:
        reasons.append("annotation ambiguity")
    if {actual, predicted} == {0, 2} and not reasons:
        reasons.append("sarcasm/implicit sentiment")
    return "; ".join(reasons) if reasons else "class overlap"


def _scores(y, p) -> dict:
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, p)),
            "macro_f1": float(f1_score(y, p, average="macro", labels=LABELS, zero_division=0)),
            "weighted_f1": float(f1_score(y, p, average="weighted", labels=LABELS, zero_division=0))}


def per_class(y, p) -> dict:
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=LABELS, zero_division=0)
    return {ID2LABEL[i]: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]),
                          "support": int(sup[i])} for i in LABELS}


def save_confusion_png(cm: np.ndarray, title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def evaluate_and_save(*, tag: str, model_name: str, input_col: str, test_df: pd.DataFrame, probs: np.ndarray,
                      out_dir: Path, best_val_macro_f1: float | None, n_errors: int = 30,
                      extra: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y = test_df["label"].to_numpy()
    pred = probs.argmax(1)
    conf = probs.max(1)

    m = {"model": model_name, "tag": tag, "input": input_col, **_scores(y, pred),
         "per_class": per_class(y, pred),
         "classification_report": classification_report(y, pred, labels=LABELS, target_names=CLASS_NAMES,
                                                         zero_division=0, output_dict=True),
         "confidence": {"mean": float(conf.mean()),
                        "mean_when_correct": float(conf[pred == y].mean()) if (pred == y).any() else None,
                        "mean_when_wrong": float(conf[pred != y].mean()) if (pred != y).any() else None},
         "best_validation_macro_f1": best_val_macro_f1, **(extra or {})}

    per_lang = {}
    for lang in LANGUAGE_TYPES:
        mask = (test_df["language_type"] == lang).to_numpy()
        if not mask.any():
            per_lang[lang] = {"status": "not in test set", "n": 0, "accuracy": None, "macro_f1": None, "weighted_f1": None}
            continue
        per_lang[lang] = {"status": "evaluated", **_scores(y[mask], pred[mask]), "per_class": per_class(y[mask], pred[mask])}
    m["per_language"] = per_lang

    cm = confusion_matrix(y, pred, labels=LABELS)
    pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_NAMES], columns=[f"pred_{c}" for c in CLASS_NAMES]) \
        .to_csv(out_dir / f"confusion_matrix_{tag}.csv", encoding="utf-8-sig")
    save_confusion_png(cm, f"{model_name}\nTest confusion matrix", out_dir / f"confusion_matrix_{tag}.png")
    m["confusion_matrix"] = cm.tolist()
    for lang in LANGUAGE_TYPES:
        mask = (test_df["language_type"] == lang).to_numpy()
        if mask.any():
            pd.DataFrame(confusion_matrix(y[mask], pred[mask], labels=LABELS),
                         index=[f"true_{c}" for c in CLASS_NAMES], columns=[f"pred_{c}" for c in CLASS_NAMES]) \
                .to_csv(out_dir / f"confusion_matrix_{tag}_{lang}.csv", encoding="utf-8-sig")

    preds = test_df[[c for c in ("raw_text", "cleaned_text", "context_text", "language_type", "source_dataset",
                                 "source_label", "annotations") if c in test_df]].copy()
    preds["actual_label"] = [ID2LABEL[int(v)] for v in y]
    preds["predicted_label"] = [ID2LABEL[int(v)] for v in pred]
    preds["confidence"] = conf
    for i, c in enumerate(CLASS_NAMES):
        preds[f"prob_{c}"] = probs[:, i]
    preds.to_csv(out_dir / f"predictions_{tag}.csv", index=False, encoding="utf-8-sig")

    wrong = preds[preds["actual_label"] != preds["predicted_label"]].copy()
    wrong["likely_error_reason"] = [
        heuristic_error_reason(str(r.cleaned_text), r.language_type, int(y[i]), int(pred[i]), float(r.confidence),
                               r.annotations if "annotations" in wrong else None, has_context=input_col == "context_text")
        for i, r in zip(np.flatnonzero(pred != y), wrong.itertuples())]
    wrong["reason_type"] = HEURISTIC_NOTE
    wrong = wrong.sort_values("confidence", ascending=False)
    k = min(n_errors, len(wrong))
    picked = pd.concat([wrong.head((k + 1) // 2), wrong.tail(k // 2)])
    picked = picked[~picked.index.duplicated()]
    if len(picked) < k:
        picked = pd.concat([picked, wrong.drop(picked.index).head(k - len(picked))])
    cols = ["actual_label", "predicted_label", "confidence", "raw_text", "cleaned_text", "language_type",
            "source_dataset", "likely_error_reason", "reason_type"]
    picked[cols].to_csv(out_dir / f"misclassified_{tag}.csv", index=False, encoding="utf-8-sig")
    m["n_misclassified_total"] = int(len(wrong))
    m["n_misclassified_saved"] = int(len(picked))
    m["heuristic_reason_counts_all_mistakes"] = wrong["likely_error_reason"].str.split("; ").explode().value_counts().to_dict()
    m["heuristic_note"] = HEURISTIC_NOTE

    (out_dir / f"metrics_{tag}.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    return m
