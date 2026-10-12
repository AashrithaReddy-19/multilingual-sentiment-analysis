"""Step 2: recompute exact metrics (incl. every F1) from the saved TEST prediction files of the four completed
models. Nothing is retrained and no file of those runs is modified.

Multilingual A / A+ share one test split (15,124 rows); Telugu A / B share another (3,168 rows). The two groups are
NOT directly comparable. Output: experiments/recomputed_metrics_comparison.csv (+ .json).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402
from pipeline.exp_common import EXP, full_metrics, labels_from_names  # noqa: E402

RUNS = [
    ("Multilingual A", "multilingual test (15,124 rows)", paths.MULTI / "model_a" / "predictions_model_a.csv"),
    ("Multilingual A+", "multilingual test (15,124 rows)", paths.MULTI / "model_a_plus" / "predictions_model_a_plus.csv"),
    ("Telugu A", "Telugu-only test (3,168 rows)", paths.TELUGU_OUTPUTS / "predictions_baseline.csv"),
    ("Telugu B", "Telugu-only test (3,168 rows)", paths.TELUGU_OUTPUTS / "predictions_no_context.csv"),
]


def flat(name: str, test_set: str, src: Path, m: dict) -> dict:
    row = {"model": name, "test_set": test_set, "source_file": str(src), "n": m["n"], "accuracy": m["accuracy"]}
    for avg in ("macro", "weighted"):
        for k in ("precision", "recall", "f1"):
            row[f"{avg}_{k}"] = m[f"{avg}_{k}"]
    for c, v in m["per_class"].items():
        for k in ("precision", "recall", "f1", "support"):
            row[f"{c}_{k}"] = v[k]
    for lt, v in m.get("per_language", {}).items():
        row[f"{lt}_accuracy"], row[f"{lt}_macro_f1"] = v["accuracy"], v["macro_f1"]
    return row


def main() -> list[dict]:
    rows, full = [], {}
    for name, test_set, src in RUNS:
        p = pd.read_csv(src, encoding="utf-8-sig", low_memory=False)
        m = full_metrics(labels_from_names(p["actual_label"]), labels_from_names(p["predicted_label"]),
                         p["language_type"])
        rows.append(flat(name, test_set, src, m))
        full[name] = m
    df = pd.DataFrame(rows)
    EXP.mkdir(parents=True, exist_ok=True)
    df.to_csv(EXP / "recomputed_metrics_comparison.csv", index=False, encoding="utf-8-sig")
    (EXP / "recomputed_metrics_comparison.json").write_text(json.dumps(full, indent=2), encoding="utf-8")
    show = ["model", "n", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_precision",
            "weighted_recall", "weighted_f1", "Negative_f1", "Neutral_f1", "Positive_f1"]
    print(df[show].round(4).to_string(index=False))
    print("\nPer class (precision / recall / F1 / support):")
    for r in rows:
        print(f"  {r['model']}: " + " | ".join(
            f"{c} {r[c + '_precision']:.4f}/{r[c + '_recall']:.4f}/{r[c + '_f1']:.4f}/{r[c + '_support']}"
            for c in ("Negative", "Neutral", "Positive")))
    print("\nNOTE: Multilingual and Telugu-only rows use different test sets; do not compare them directly.")
    return rows


if __name__ == "__main__":
    main()
