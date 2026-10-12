"""Error analysis of the existing multilingual Model A on the VALIDATION split (test split untouched).

Loads the saved model (no retraining), predicts validation, and measures error rates by class, language,
label-conflict groups and text patterns (short text, emoji, negation, Romanized Telugu, code-mixed script).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402
from sentiment_core.evaluation import EMOJI_RE, LATIN_RE, NEG_RE, TELUGU_RE  # noqa: E402
from sentiment_core.labels import CLASS_NAMES  # noqa: E402

OUT = paths.MULTI / "experiments" / "error_analysis_model_a"


def main() -> None:
    if (OUT / "error_analysis.json").exists():
        raise SystemExit(f"{OUT} already exists; refusing to overwrite.")
    OUT.mkdir(parents=True)
    val = pd.read_csv(paths.DATA / "validation.csv", encoding="utf-8-sig", low_memory=False)
    model = joblib.load(paths.MODEL_A / "tfidf_logreg_multilingual.joblib")
    probs = model.predict_proba(val["cleaned_text"])[:, np.argsort(model.classes_)]
    val["pred"] = probs.argmax(1)
    val["conf"] = probs.max(1)
    val["wrong"] = val["pred"] != val["label"]
    t = val["cleaned_text"].astype(str)
    tags = val["language_tags"].fillna("").astype(str)
    patterns = {
        "short_text_<=3_tokens": t.str.split().str.len() <= 3,
        "contains_emoji": t.str.contains(EMOJI_RE),
        "contains_negation": t.str.contains(NEG_RE),
        "telugu_script_present": t.str.contains(TELUGU_RE),
        "romanized_telugu (CMTET te-tags, no Telugu script)": (val["source_dataset"] == "CMTET")
            & tags.str.contains(r"\bte\b") & ~t.str.contains(TELUGU_RE),
        "code_mixed_tags (CMTET has both en and te tags)": tags.str.contains(r"\ben\b") & tags.str.contains(r"\bte\b"),
        "telugu_and_latin_script_mixed": t.str.contains(TELUGU_RE) & t.str.contains(LATIN_RE),
        "label_conflict_group": val["conflicting_label_group"].astype(str).str.lower() == "true",
        "low_confidence_<0.5": val["conf"] < 0.5,
    }
    overall_err = float(val["wrong"].mean())
    pat = []
    for name, mask in patterns.items():
        n = int(mask.sum())
        pat.append({"pattern": name, "rows": n, "share_of_validation": round(n / len(val), 4),
                    "error_rate": round(float(val.loc[mask, "wrong"].mean()), 4) if n else None,
                    "error_rate_rest": round(float(val.loc[~mask, "wrong"].mean()), 4),
                    "macro_f1": round(float(f1_score(val.loc[mask, "label"], val.loc[mask, "pred"], average="macro",
                                                     labels=[0, 1, 2], zero_division=0)), 4) if n else None})
    pat_df = pd.DataFrame(pat)
    pat_df.to_csv(OUT / "pattern_error_rates.csv", index=False)

    lang = []
    for (lng), g in val.groupby("language_type"):
        cm = confusion_matrix(g["label"], g["pred"], labels=[0, 1, 2])
        lang.append({"language_type": lng, "rows": len(g), "error_rate": round(float(g["wrong"].mean()), 4),
                     "macro_f1": round(float(f1_score(g["label"], g["pred"], average="macro")), 4),
                     "confusion_rows_actual_cols_pred": cm.tolist()})
    cm_all = confusion_matrix(val["label"], val["pred"], labels=[0, 1, 2])
    pairs = []
    for i, a in enumerate(CLASS_NAMES):
        for j, p in enumerate(CLASS_NAMES):
            if i != j:
                pairs.append({"actual": a, "predicted": p, "count": int(cm_all[i, j]),
                              "share_of_errors": round(float(cm_all[i, j] / val["wrong"].sum()), 4)})
    pairs = sorted(pairs, key=lambda d: -d["count"])
    report = {"split": "validation", "rows": len(val), "macro_f1": round(float(f1_score(val["label"], val["pred"], average="macro")), 4),
              "accuracy": round(1 - overall_err, 4), "error_rate": round(overall_err, 4),
              "confusion_matrix_rows_actual_cols_pred": cm_all.tolist(), "error_pairs": pairs,
              "per_language": lang, "patterns": pat}
    (OUT / "error_analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("rows", "macro_f1", "accuracy")}, indent=1))
    print("error pairs:", [(d["actual"], d["predicted"], d["count"], d["share_of_errors"]) for d in pairs])
    print(pd.DataFrame(lang).drop(columns="confusion_rows_actual_cols_pred").to_string(index=False))
    print(pat_df.to_string(index=False))


if __name__ == "__main__":
    main()
