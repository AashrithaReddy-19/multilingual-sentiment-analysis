"""Write docs/final_report.md from saved, executed results only (nothing is estimated or invented)."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402
from sentiment_core.labels import CLASS_NAMES, MODEL_B_PENDING, MODEL_C_SKIP  # noqa: E402

REPORT = paths.APP_ROOT / "docs" / "final_report.md"


def _load(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _f(v) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"


def main() -> Path:
    tel_a = _load(paths.TELUGU_OUTPUTS / "metrics_baseline.json")
    tel_b = _load(paths.TELUGU_OUTPUTS / "metrics_no_context.json")
    mul_a = _load(paths.MODEL_A / "metrics_model_a.json")
    mul_b = _load(paths.MODEL_B / "metrics_model_b.json")
    mul_ap = _load(paths.MULTI / "model_a_plus" / "metrics_model_a_plus.json")
    if mul_ap and mul_a and mul_ap["macro_f1"] <= mul_a["macro_f1"]:
        mul_ap = None                       # only reported as an improvement when it beats the baseline on test
    clean = _load(paths.DATA / "cleaning_report.json")

    L = [f"# Final report: real results\n", f"_Generated {datetime.now():%Y-%m-%d %H:%M} from saved metric files._\n",
         "Only models that were actually trained and evaluated appear with numbers.\n"]

    L.append("## 1. Multilingual run (TeSent + TweetEval + CMTET)\n")
    if clean:
        sz = clean["split"]["sizes"]
        L.append(f"Split: train {sz['train']}, validation {sz['validation']}, test {sz['test']} "
                 f"(cleaned-text leakage across splits: {'none' if clean['leakage_checks']['passed'] else 'FOUND'}).\n")
    rows = [("A: TF-IDF + Logistic Regression (original baseline)", "cleaned_text", mul_a)]
    if mul_ap:
        rows.append(("A+: word + char TF-IDF + Logistic Regression", "cleaned_text", mul_ap))
    rows.append(("B: XLM-RoBERTa (no context)", "cleaned_text", mul_b))
    L.append("| Model | Input | Accuracy | Macro F1 | Weighted F1 | Telugu F1 | English F1 | Code-mixed F1 | Best val. F1 | Status |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, inp, m in rows:
        if m:
            pl = m["per_language"]
            L.append(f"| {name} | {inp} | {_f(m['accuracy'])} | {_f(m['macro_f1'])} | {_f(m['weighted_f1'])} | "
                     f"{_f(pl['telugu']['macro_f1'])} | {_f(pl['english']['macro_f1'])} | {_f(pl['code_mixed']['macro_f1'])} | "
                     f"{_f(m['best_validation_macro_f1'])} | trained |")
        else:
            L.append(f"| {name} | {inp} | — | — | — | — | — | — | — | {MODEL_B_PENDING} |")
    L.append(f"| C: XLM-RoBERTa (context-aware) | context_text | — | — | — | — | — | — | — | {MODEL_C_SKIP} |\n")

    if mul_ap:
        L.append("Model A+ was chosen from 79 CPU experiment variants by validation macro F1 (then validation accuracy); "
                 "the test split was used once, after selection. The original Model A baseline is unchanged. "
                 "Details: artifacts/multilingual/experiments/experiment_comparison.csv.\n")
    for name, m in (("Model A (multilingual, original baseline)", mul_a), ("Model A+ (multilingual)", mul_ap),
                    ("Model B (multilingual)", mul_b)):
        if not m:
            continue
        L.append(f"### {name}: per-class (test)\n")
        L.append("| Class | Precision | Recall | F1 | Support |\n|---|---|---|---|---|")
        for c in CLASS_NAMES:
            pc = m["per_class"][c]
            L.append(f"| {c} | {_f(pc['precision'])} | {_f(pc['recall'])} | {_f(pc['f1'])} | {pc['support']} |")
        cm = m["confusion_matrix"]
        L.append("\nConfusion matrix (rows = actual, columns = predicted Negative / Neutral / Positive):\n")
        L.append("| Actual | Negative | Neutral | Positive |\n|---|---|---|---|")
        for c, r in zip(CLASS_NAMES, cm):
            L.append(f"| {c} | {r[0]} | {r[1]} | {r[2]} |")
        L.append(f"\nMean confidence {_f(m['confidence']['mean'])} (correct {_f(m['confidence']['mean_when_correct'])}, "
                 f"wrong {_f(m['confidence']['mean_when_wrong'])}). Misclassified: {m['n_misclassified_total']} "
                 f"({m['n_misclassified_saved']} saved with heuristic reasons, which are not ground truth).\n")

    L.append("## 2. Telugu-only run (TeSent only, completed earlier on CPU)\n")
    L.append("| Model | Accuracy | Macro F1 | Weighted F1 | Best val. F1 |\n|---|---|---|---|---|")
    for name, m in (("A: TF-IDF + Logistic Regression", tel_a), ("B: XLM-RoBERTa (no context)", tel_b)):
        if m:
            L.append(f"| {name} | {_f(m['accuracy'])} | {_f(m['macro_f1'])} | {_f(m['weighted_f1'])} | "
                     f"{_f(m['best_validation_macro_f1'])} |")
    L.append(f"| C: XLM-RoBERTa (context-aware) | — | — | — | {MODEL_C_SKIP} |\n")
    L.append("The two runs use different splits and data, so their numbers are not directly comparable.\n")

    L.append("## 3. Model C\n")
    L.append(f"{MODEL_C_SKIP} None of the datasets contain conversation_id/message_order, so context_text equals "
             "cleaned_text for every row; training Model C would duplicate Model B.\n")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(L), encoding="utf-8")
    print("written", REPORT)
    return REPORT


if __name__ == "__main__":
    main()
