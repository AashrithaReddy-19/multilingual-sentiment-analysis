"""(Re)build and execute D:\\NLP\\context_aware_sentiment_analysis.ipynb as the CURRENT-STATE notebook.

The notebook only reads saved artifacts, displays executed metrics, runs demo predictions with the served model and
documents the method. It trains nothing and writes nothing outside its own outputs. The original Phase-1
(Telugu-only, 2026-09-30) notebook is preserved unchanged in D:\\NLP\\archive\\.

    python -m pipeline.build_status_notebook [--no-execute]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402

NB = paths.NLP_ROOT / "context_aware_sentiment_analysis.ipynb"
KERNEL = "nlp-sentiment"

CELLS = [
    ("md", """# Sentiment Analysis for Telugu, English and Telugu-English Code-Mixed Text — current state

**Classes:** `0 = Negative`, `1 = Neutral`, `2 = Positive`

This notebook **reads the project's saved, executed results** and demonstrates the served model. It does not train
anything. The authoritative pipeline lives in `D:\\NLP\\sentiment_app\\backend\\pipeline` (see `sentiment_app/README.md`).

- The original Phase-1 notebook (Telugu-only run, 30 Sep 2026) is preserved unchanged in
  `D:\\NLP\\archive\\context_aware_sentiment_analysis_phase1_telugu_2026-09-30.ipynb`. Its results are in `D:\\NLP\\outputs` (read-only).
- **Model C (context-aware): Skipped — no real conversation_id/message_order available.** No dataset contains genuine
  conversation metadata, so genuine context-aware training was not performed. Every served prediction uses the current message only.
- Kernel: **Python (NLP Sentiment)** = Python 3.11 at `C:\\Users\\sresh\\AppData\\Local\\Programs\\Python\\Python311\\python.exe`."""),
    ("md", "## 1. Environment verification"),
    ("code", """import sys
print(sys.executable)

import joblib
import sklearn
import torch
import transformers
import datasets
import evaluate

print("joblib:", joblib.__version__)
print("sklearn:", sklearn.__version__)
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("evaluate:", evaluate.__version__)
print("CUDA:", torch.cuda.is_available())

import pandas, numpy, matplotlib, seaborn, scipy, accelerate, sentencepiece, tokenizers, openpyxl
for m in (pandas, numpy, matplotlib, seaborn, scipy, accelerate, sentencepiece, tokenizers, openpyxl):
    print(f"{m.__name__}: {getattr(m, '__version__', 'ok')}")"""),
    ("code", """import json, hashlib
from pathlib import Path
import pandas as pd
from IPython.display import Image, Markdown, display

NLP = Path(r"D:\\NLP")
APP = NLP / "sentiment_app"
sys.path.insert(0, str(APP / "backend"))
MULTI = APP / "artifacts" / "multilingual"
DATA, EXP = MULTI / "data", MULTI / "experiments"
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 80)"""),
    ("md", """## 2. Data and split
TeSent (Telugu) + TweetEval sentiment (English) + CMTET (Telugu-English code-mixed). Label mappings are taken only from
each dataset's documentation (`label_mapping.json`). Hate-speech files in `Tel_Eng_dataset` are **not** used.
The split is a group-level stratified 70/15/15 split (identical cleaned texts never cross splits); the **test split is frozen**."""),
    ("code", """split = {s: pd.read_csv(DATA / f"{s}.csv", encoding="utf-8-sig", low_memory=False) for s in ("train", "validation", "test")}
print(json.dumps(json.loads((DATA / "label_mapping.json").read_text(encoding="utf-8"))["documented_mappings"], indent=1))
for s, d in split.items():
    print(f"\\n{s}: {len(d):,} rows")
    display(pd.crosstab(d["language_type"], d["label"].map({0: "Negative", 1: "Neutral", 2: "Positive"}), margins=True))
pkg = APP / "artifacts" / "colab_package_phase3" / "MANIFEST.json"
if pkg.exists():
    man = json.loads(pkg.read_text())
    for s in split:
        ok = man[f"data/{s}.csv"] == hashlib.sha256((DATA / f"{s}.csv").read_bytes()).hexdigest()
        print(f"{s}.csv matches the Colab package fingerprint: {ok}")"""),
    ("md", """## 3. Executed results (multilingual test split, n = 15,124)
Only models chosen on **validation** are scored on test. The Telugu-only run uses a different test split (n = 3,168) and is shown separately."""),
    ("code", """comp = pd.read_csv(EXP / "final_model_comparison.csv", encoding="utf-8-sig")
display(comp.round(4))
print("Telugu-only run (different test split, not comparable with the table above):")
display(pd.read_csv(EXP / "recomputed_metrics_comparison.csv", encoding="utf-8-sig")
        .query("model.str.startswith('Telugu')", engine="python")
        [["model", "n", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1", "Negative_f1", "Neutral_f1", "Positive_f1"]].round(4))"""),
    ("code", """lb = pd.read_csv(EXP / "experiment_leaderboard.csv", encoding="utf-8-sig")
cls = lb[lb["experiment_id"].str.match(r"s[0-6]_")]
print(f"Classical validation-only search: {len(cls)} configurations (word / char / union TF-IDF, LR / LinearSVC / SGD, "
      "preprocessing variants, signal features). Top 10 by validation macro F1:")
display(cls.sort_values("validation_macro_f1", ascending=False).head(10)
        [["experiment_id", "model", "validation_accuracy", "validation_macro_precision", "validation_macro_recall", "validation_macro_f1"]].round(4))"""),
    ("md", "## 4. Transformer experiments (Colab GPU)"),
    ("code", """trans = EXP / "transformers"
runs = sorted(trans.glob("*/validation_metrics.json")) if trans.exists() else []
if not runs:
    display(Markdown("**No Transformer results imported yet — Not trained / pending GPU run.**  \\n"
                     "Run `sentiment_app/artifacts/colab_package_phase3/colab_phase3_transformers.ipynb` on a Colab GPU "
                     "(upload `transformer_package.zip`), copy `phase3_results_*.zip` to `sentiment_app/artifacts/colab_results/`, then run "
                     "`python -m pipeline.finalize_phase3 --import` in `sentiment_app/backend`. See `sentiment_app/docs/COLAB_PHASE3.md`."))
else:
    rows = []
    for f in runs:
        v, c = json.loads(f.read_text()), json.loads((f.parent / "config.json").read_text())
        rows.append({"experiment": f.parent.name, "model": c["model"], "val_acc": v["accuracy"], "val_macro_P": v["macro_precision"],
                     "val_macro_R": v["macro_recall"], "val_macro_F1": v["macro_f1"], "val_weighted_F1": v["weighted_f1"],
                     **{f"val_{k}_F1": v["per_class"][k]["f1"] for k in ("Negative", "Neutral", "Positive")},
                     "best_epoch": c.get("best_epoch"), "tested": (f.parent / "test_metrics.json").exists()})
    display(pd.DataFrame(rows).sort_values(["val_macro_F1", "val_macro_R", "val_acc"], ascending=False).round(4))"""),
    ("md", "## 5. Error analysis, statistics and calibration"),
    ("code", """ea = EXP / "error_analysis_model_a_plus"
print("Model A+ test errors by class pair:"); display(pd.read_csv(ea / "class_pair_errors.csv"))
print("By language:"); display(pd.read_csv(ea / "errors_by_language.csv").iloc[:, :6])
print("Accuracy by confidence bucket:"); display(pd.read_csv(ea / "errors_by_confidence.csv"))
display(Image(filename=str(ea / "confusion_matrix_normalized.png"), width=420))
stats = json.loads((EXP / "statistical_comparison.json").read_text())
for k, v in stats.items():
    mc, bs = v["mcnemar"], v["bootstrap"]
    print(f"{k}: McNemar p = {mc['p_value']:.3g}; macro F1 difference {bs['macro_f1_diff_mean']:+.4f} "
          f"(95% CI {bs['macro_f1_diff_ci95'][0]:+.4f} to {bs['macro_f1_diff_ci95'][1]:+.4f})")
for d in sorted((EXP / "final").iterdir()):
    tm = d / "test_metrics.json"
    if tm.exists():
        cal = json.loads(tm.read_text())["calibration_test"]
        print(f"{d.name}: ECE {cal['ece_10_bins']:.4f}, mean confidence {cal['mean_confidence']:.4f}")
if (EXP / "final_model_comparison.png").exists():
    display(Image(filename=str(EXP / "final_model_comparison.png"), width=900))"""),
    ("md", """## 6. Demonstration with the served model
Uses the web app's own `ModelService`: the final selected model if its files are on this machine, otherwise **Multilingual A+**.
The language shown is a heuristic guess; predictions use the current message only."""),
    ("code", """from app.config import settings
from app.model_service import ModelService
svc = ModelService(settings)
print("Serving:", svc.info(svc.default_id)["name"]); print(svc.selection_note())
examples = ["ఈ సినిమా చాలా బాగుంది 👍", "The delivery was late again. Not happy.", "The meeting is scheduled for 3 pm tomorrow.",
            "movie asalu bagaledu, time waste", "Service super ga undi bro, thank you!", "This is not bad at all", "😡😡"]
probs, cleaned, mid, langs = svc.predict_full(examples)
names = svc.class_names()
display(pd.DataFrame({"text": examples, "language (heuristic)": langs, "prediction": [names[i] for i in probs.argmax(1)],
                      "confidence": probs.max(1).round(3), **{f"P({c})": probs[:, i].round(3) for i, c in enumerate(names)},
                      "low_confidence (<0.5)": probs.max(1) < 0.5}))"""),
    ("md", """## 7. Methodology notes
- **TF-IDF + Logistic Regression (A):** word 1–2-grams weighted by TF-IDF, linear classifier with `class_weight="balanced"`.
- **A+:** adds character n-grams (`char_wb` 3–5), which handle Telugu morphology, Romanized spelling variation and code-switching. It is the best of 177 validation-searched classical configurations.
- **XLM-RoBERTa / MuRIL (Phase 3, Colab):** pretrained multilingual Transformers fine-tuned on `cleaned_text` (current message only), selected by validation macro F1 (ties: macro recall, then accuracy).
- **Selection rule:** train → fit; validation → choose model, hyperparameters, ensemble weights, temperature; test → evaluated once per selected model.
- **Metrics:** Macro F1 weights Negative/Neutral/Positive equally (primary); weighted F1 accounts for class frequency; accuracy = share of correct predictions.
- **Model C:** the context builder (`sentiment_core/context.py`, format `[PREVIOUS_2] … [PREVIOUS_1] … [CURRENT] …`, earlier messages only) and `train_transformer.py --input-col context_text` are ready, but training is refused until a real `conversations.csv` with `conversation_id` and `message_order` exists."""),
]


def build(execute: bool = True) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nb.metadata = {"kernelspec": {"display_name": "Python (NLP Sentiment)", "language": "python", "name": KERNEL},
                   "language_info": {"name": "python"}}
    if execute:
        from nbclient import NotebookClient
        NotebookClient(nb, kernel_name=KERNEL, timeout=600, resources={"metadata": {"path": str(NB.parent)}}).execute()
    nbf.write(nb, NB)
    print(f"wrote {NB} ({'executed' if execute else 'not executed'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-execute", action="store_true")
    build(not ap.parse_args().no_execute)
