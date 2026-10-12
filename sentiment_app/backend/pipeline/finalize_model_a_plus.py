"""Select Model A+ from validation results, refit it on train, verify its validation score, then evaluate it
ONCE on the untouched test split. Writes artifacts/multilingual/model_a_plus and experiments/experiment_comparison.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402
from pipeline.run_experiments import OUT as EXP, build  # noqa: E402
from sentiment_core.evaluation import evaluate_and_save  # noqa: E402

PLUS = paths.MULTI / "model_a_plus"


def main() -> None:
    if (PLUS / "metrics_model_a_plus.json").exists():
        raise SystemExit(f"{PLUS} already has results; refusing to overwrite (the test split is evaluated only once).")
    res = pd.read_csv(EXP / "validation_results.csv").sort_values(["val_macro_f1", "val_accuracy"], ascending=False)
    best = res.iloc[0]
    grid = {c["id"]: c for c in json.loads((EXP / "experiment_grid.json").read_text(encoding="utf-8"))}
    cfg = grid[best["id"]]
    print("selected by validation macro F1, then accuracy:", cfg)

    read = lambda s: pd.read_csv(paths.DATA / f"{s}.csv", encoding="utf-8-sig", low_memory=False)  # noqa: E731
    train, val = read("train"), read("validation")
    model = build(cfg).fit(train["cleaned_text"].astype(str), train["label"])
    vpred = model.predict(val["cleaned_text"].astype(str))
    val_f1, val_acc = float(f1_score(val["label"], vpred, average="macro")), float(accuracy_score(val["label"], vpred))
    print(f"refit validation macro F1 {val_f1:.4f} (grid {best['val_macro_f1']:.4f}), accuracy {val_acc:.4f}")
    if abs(val_f1 - best["val_macro_f1"]) > 1e-3:
        raise SystemExit("Refit does not reproduce the validation score; stopping before touching the test split.")

    PLUS.mkdir(parents=True)
    joblib.dump(model, PLUS / "model_a_plus.joblib")
    test = read("test")                                   # first and only use of the test split for Model A+
    probs = model.predict_proba(test["cleaned_text"].astype(str))[:, np.argsort(model.classes_)]
    m = evaluate_and_save(tag="model_a_plus", model_name="Model A+: word + char TF-IDF + Logistic Regression (multilingual)",
                          input_col="cleaned_text", test_df=test, probs=probs, out_dir=PLUS, best_val_macro_f1=val_f1,
                          extra={"config": cfg, "validation_accuracy": val_acc, "selection_rule":
                                 "highest validation macro F1, then validation accuracy, over "
                                 f"{len(res)} candidates; test split used once, after selection",
                                 "features": "FeatureUnion(word TF-IDF (1,2) sublinear max_features=20000, "
                                             "char_wb TF-IDF (3,5) sublinear min_df=2)",
                                 "confidence_source": "Logistic Regression predict_proba (native probabilities)"})

    base = json.loads((paths.MODEL_A / "metrics_model_a.json").read_text(encoding="utf-8"))
    res = res.assign(selected=res["id"] == best["id"], test_macro_f1=np.nan, test_accuracy=np.nan, test_weighted_f1=np.nan)
    sel = res["id"] == best["id"]
    res.loc[sel, ["test_macro_f1", "test_accuracy", "test_weighted_f1"]] = [m["macro_f1"], m["accuracy"], m["weighted_f1"]]
    baseline_row = {"id": "model_a_original", "family": "word_lr (original Model A, saved model)", "ngram": "[1, 2]",
                    "sublinear_tf": False, "min_df": 1, "max_features": 20000, "C": 1.0,
                    "val_macro_f1": base["best_validation_macro_f1"], "test_macro_f1": base["macro_f1"],
                    "test_accuracy": base["accuracy"], "test_weighted_f1": base["weighted_f1"], "selected": False}
    comp = pd.concat([pd.DataFrame([baseline_row]), res], ignore_index=True)
    comp.to_csv(EXP / "experiment_comparison.csv", index=False)

    beats = m["macro_f1"] > base["macro_f1"]
    summary = {"model_a_original": {k: base[k] for k in ("accuracy", "macro_f1", "weighted_f1")},
               "model_a_plus": {k: m[k] for k in ("accuracy", "macro_f1", "weighted_f1")},
               "per_language_macro_f1": {lang: {"original": base["per_language"][lang]["macro_f1"],
                                                "plus": m["per_language"][lang]["macro_f1"]} for lang in m["per_language"]},
               "model_a_plus_beats_original_on_test_macro_f1": beats,
               "ui_updated": beats}
    (PLUS / "comparison_with_original.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
