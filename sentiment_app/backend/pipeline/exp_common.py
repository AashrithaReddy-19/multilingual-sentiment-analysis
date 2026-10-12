"""Shared helpers for the improvement experiments (Phase 3).

Every experiment gets its own folder under artifacts/multilingual/experiments/<experiment_id>/ with
config.json, validation_metrics.json, predictions.csv (validation), classification_report.json,
confusion_matrix.csv/png. test_metrics.json is written only for configurations selected on validation.
experiment_leaderboard.csv is upserted by experiment_id, so re-running one experiment never touches others.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402
from sentiment_core.experiment_io import LABELS, full_metrics, save_eval_files  # noqa: E402,F401
from sentiment_core.labels import LABEL2ID  # noqa: E402

EXP = paths.MULTI / "experiments"
LEADERBOARD = EXP / "experiment_leaderboard.csv"
LEADERBOARD_COLS = ["experiment_id", "model", "features", "preprocessing", "hyperparameters",
                    "validation_accuracy", "validation_macro_precision", "validation_macro_recall",
                    "validation_macro_f1", "validation_weighted_f1", "test_accuracy", "test_macro_f1", "notes"]


def read_split(name: str) -> pd.DataFrame:
    return pd.read_csv(paths.DATA / f"{name}.csv", encoding="utf-8-sig", keep_default_na=False, na_values=[""],
                       low_memory=False)


def upsert_leaderboard(row: dict) -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    row = {k: row.get(k, "") for k in LEADERBOARD_COLS}
    for k in ("features", "preprocessing", "hyperparameters"):
        if not isinstance(row[k], str):
            row[k] = json.dumps(row[k], ensure_ascii=False)
    lb = pd.read_csv(LEADERBOARD, encoding="utf-8-sig") if LEADERBOARD.exists() else pd.DataFrame(columns=LEADERBOARD_COLS)
    lb = lb[lb["experiment_id"] != row["experiment_id"]]
    lb = pd.concat([lb, pd.DataFrame([row])], ignore_index=True)
    lb = lb.sort_values("validation_macro_f1", ascending=False, na_position="last")
    lb.to_csv(LEADERBOARD, index=False, encoding="utf-8-sig")


def leaderboard_row_from(exp_id: str, model: str, features, preprocessing, hyper, val: dict, notes: str = "",
                         test: dict | None = None) -> dict:
    return {"experiment_id": exp_id, "model": model, "features": features, "preprocessing": preprocessing,
            "hyperparameters": hyper, "validation_accuracy": val["accuracy"],
            "validation_macro_precision": val["macro_precision"], "validation_macro_recall": val["macro_recall"],
            "validation_macro_f1": val["macro_f1"], "validation_weighted_f1": val["weighted_f1"],
            "test_accuracy": test["accuracy"] if test else "", "test_macro_f1": test["macro_f1"] if test else "",
            "notes": notes}


def labels_from_names(names) -> np.ndarray:
    return np.array([LABEL2ID[str(n)] for n in names], dtype=int)


def stage_banner(done: str, next_step: str) -> None:
    """Print DONE / CURRENT BEST VALIDATION MACRO F1 / CURRENT BEST MODEL / NEXT STEP from the leaderboard."""
    best_f1, best_model = "n/a", "n/a"
    if LEADERBOARD.exists():
        lb = pd.read_csv(LEADERBOARD, encoding="utf-8-sig")
        lb = lb[pd.to_numeric(lb["validation_macro_f1"], errors="coerce").notna()]
        if len(lb):
            top = lb.sort_values("validation_macro_f1", ascending=False).iloc[0]
            best_f1, best_model = f"{float(top['validation_macro_f1']):.4f}", f"{top['experiment_id']} ({top['model']})"
    print("\n" + "=" * 70)
    print(f"DONE: {done}")
    print(f"CURRENT BEST VALIDATION MACRO F1: {best_f1}")
    print(f"CURRENT BEST MODEL: {best_model}")
    print(f"NEXT STEP: {next_step}")
    print("=" * 70, flush=True)
