"""Phase 2A: TF-IDF + Logistic Regression on the multilingual split (CPU; trains in minutes)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402
from sentiment_core.evaluation import evaluate_and_save  # noqa: E402

PARAMS = {"analyzer": "word", "ngram_range": [1, 2], "max_features": 20000,
          "token_pattern": r"(?u)\S+", "lowercase": True,
          "class_weight": "balanced", "max_iter": 1000, "random_state": 42}


def main() -> None:
    out = paths.MODEL_A
    if (out / "metrics_model_a.json").exists():
        raise SystemExit(f"{out} already has Model A results; refusing to overwrite.")
    read = lambda s: pd.read_csv(paths.DATA / f"{s}.csv", encoding="utf-8-sig", keep_default_na=False,  # noqa: E731
                                 na_values=[""])
    train, val, test = read("train"), read("validation"), read("test")

    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=20000,
                                  token_pattern=r"(?u)\S+", lowercase=True)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ])
    t0 = time.time()
    model.fit(train["cleaned_text"], train["label"])
    train_seconds = round(time.time() - t0, 1)
    val_f1 = float(f1_score(val["label"], model.predict(val["cleaned_text"]), average="macro"))
    print(f"trained in {train_seconds}s | validation macro F1 {val_f1:.4f}")

    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / "tfidf_logreg_multilingual.joblib")
    probs = model.predict_proba(test["cleaned_text"])[:, np.argsort(model.classes_)]
    m = evaluate_and_save(tag="model_a", model_name="TF-IDF + Logistic Regression (multilingual)",
                          input_col="cleaned_text", test_df=test, probs=probs, out_dir=out,
                          best_val_macro_f1=val_f1,
                          extra={"params": PARAMS, "train_rows": len(train), "train_seconds": train_seconds,
                                 "token_pattern_note": "\\S+ keeps Telugu words with vowel signs and emojis intact"})
    print(json.dumps({k: m[k] for k in ("accuracy", "macro_f1", "weighted_f1", "best_validation_macro_f1")}, indent=1))
    print({k: (round(v["macro_f1"], 4) if v["macro_f1"] is not None else None) for k, v in m["per_language"].items()})
    print("saved to", out)


if __name__ == "__main__":
    main()
