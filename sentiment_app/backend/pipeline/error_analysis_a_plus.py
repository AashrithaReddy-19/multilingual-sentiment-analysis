"""Step 3: error analysis of Multilingual A+ from its saved TEST predictions (read-only; nothing is retrained and
nothing here is used to choose a model or a hyperparameter).

Outputs in experiments/error_analysis_model_a_plus/:
  error_analysis_model_a_plus.csv   every misclassified test row with HEURISTIC categories
  misclassified_sample_150.csv      150 mistakes (50 highest-confidence, 50 lowest, 50 random)
  class_pair_errors.csv, errors_by_language.csv, errors_by_length.csv, errors_by_confidence.csv,
  heuristic_category_counts.csv, confusion_matrix(.csv/.png), confusion_matrix_normalized(.csv/.png), summary.json
Heuristic categories are rule-based guesses, NOT ground truth.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402
from pipeline.exp_common import EXP  # noqa: E402
from sentiment_core.evaluation import EMOJI_RE, LATIN_RE, NEG_RE, TELUGU_RE  # noqa: E402
from sentiment_core.labels import CLASS_NAMES  # noqa: E402

OUT = EXP / "error_analysis_model_a_plus"
HEURISTIC_NOTE = "HEURISTIC (rule-based guess, not ground truth)"
# Frequent Romanized Telugu function/sentiment words (spelling varies; this is only a cue list).
ROMAN_TE = re.compile(r"\b(ledu|ledhu|kadu|kaadu|kaadhu|vaddu|avvale|chala|chaala|bagundi|baagundi|bagunnadi|undi|"
                      r"unnadi|emi|enti|endi|em|nenu|nuvvu|meeru|manchi|chesadu|chesaru|cheyyi|anna|annaya|"
                      r"ra|raa|kuda|kooda|inka|asalu|super|sir|ga|gaa|lo|ki|ni|naaku|maaku)\b", re.IGNORECASE)
SARCASM_CUES = re.compile(r"\b(yeah right|sure|great job|wow|thanks a lot|oh really|obviously|lol|lmao)\b|"
                          r"😂|🤣|🙄|😏|\.\.\.|!{2,}|\?{2,}", re.IGNORECASE)
LEN_BINS = [0, 3, 10, 20, 40, 10_000]
LEN_LABELS = ["1-3 words", "4-10 words", "11-20 words", "21-40 words", "41+ words"]
CONF_BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
CONF_LABELS = ["0.00-0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", "0.80-0.90", "0.90-1.00"]


def categories(r) -> list[str]:
    t = str(r.cleaned_text)
    n_words = len(t.split())
    cats = []
    if NEG_RE.search(t):
        cats.append("negation handling")
    if {r.actual_label, r.predicted_label} == {"Negative", "Positive"} and not NEG_RE.search(t):
        cats.append("sarcasm or implicit sentiment" + (" (cue present)" if SARCASM_CUES.search(t) else ""))
    if r.language_type == "code_mixed" or (TELUGU_RE.search(t) and LATIN_RE.search(t)):
        cats.append("code-mixed ambiguity")
    if LATIN_RE.search(t) and not TELUGU_RE.search(t) and ROMAN_TE.search(t) and r.language_type != "english":
        cats.append("Romanized Telugu ambiguity")
    if EMOJI_RE.search(t):
        cats.append("emoji sentiment")
    if "Neutral" in (r.actual_label, r.predicted_label):
        cats.append("neutral-vs-polar ambiguity")
    if n_words <= 3:
        cats.append("short text")
    if n_words > 40:
        cats.append("very long text")
    ann = r.annotations if isinstance(r.annotations, str) else ""
    if ann and len({a.split("-")[0].strip() for a in ann.split("|") if a.strip()}) > 1:
        cats.append("label ambiguity (annotators disagreed)")
    if r.confidence < 0.5:
        cats.append("low-confidence prediction")
    return cats or ["unexplained by heuristics"]


def save_cm(cm: np.ndarray, fmt: str, title: str, stem: str) -> None:
    pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_NAMES], columns=[f"pred_{c}" for c in CLASS_NAMES]) \
        .to_csv(OUT / f"{stem}.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(OUT / f"{stem}.png", dpi=150); plt.close(fig)


def rate_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    g = df.groupby(by, observed=False)
    t = pd.DataFrame({"n": g.size(), "errors": g["wrong"].sum()})
    t["error_rate"] = (t["errors"] / t["n"]).round(4)
    t["accuracy"] = (1 - t["error_rate"]).round(4)
    t["share_of_all_errors"] = (t["errors"] / df["wrong"].sum()).round(4)
    return t.reset_index()


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(paths.MULTI / "model_a_plus" / "predictions_model_a_plus.csv", encoding="utf-8-sig",
                    low_memory=False)
    p["wrong"] = p["actual_label"] != p["predicted_label"]
    p["n_words"] = p["cleaned_text"].astype(str).str.split().str.len()
    p["length_bucket"] = pd.cut(p["n_words"], LEN_BINS, labels=LEN_LABELS)
    p["confidence_bucket"] = pd.cut(p["confidence"], CONF_BINS, labels=CONF_LABELS, right=False)

    cm = confusion_matrix(p["actual_label"], p["predicted_label"], labels=CLASS_NAMES)
    save_cm(cm, "d", "Model A+ (test) confusion matrix", "confusion_matrix")
    save_cm(cm / cm.sum(1, keepdims=True), ".3f", "Model A+ (test) row-normalized\n(recall per true class)",
            "confusion_matrix_normalized")

    wrong = p[p["wrong"]].copy()
    wrong["error_pair"] = wrong["actual_label"] + " -> " + wrong["predicted_label"]
    wrong["heuristic_categories"] = ["; ".join(categories(r)) for r in wrong.itertuples()]
    wrong["category_note"] = HEURISTIC_NOTE
    cols = ["error_pair", "actual_label", "predicted_label", "confidence", "prob_Negative", "prob_Neutral",
            "prob_Positive", "language_type", "source_dataset", "n_words", "raw_text", "cleaned_text", "annotations",
            "heuristic_categories", "category_note"]
    wrong = wrong.sort_values("confidence", ascending=False)
    wrong[cols].to_csv(OUT / "error_analysis_model_a_plus.csv", index=False, encoding="utf-8-sig")
    sample = pd.concat([wrong.head(50), wrong.tail(50),
                        wrong.iloc[50:-50].sample(50, random_state=42)]).drop_duplicates(subset=["cleaned_text"])
    sample[cols].to_csv(OUT / "misclassified_sample_150.csv", index=False, encoding="utf-8-sig")

    pairs = [f"{a} -> {b}" for a in CLASS_NAMES for b in CLASS_NAMES if a != b]
    pair_counts = wrong["error_pair"].value_counts().reindex(pairs, fill_value=0)
    pair_df = pd.DataFrame({"error_pair": pairs, "count": pair_counts.to_numpy(),
                            "share_of_errors": (pair_counts / len(wrong)).round(4).to_numpy()})
    pair_df.to_csv(OUT / "class_pair_errors.csv", index=False, encoding="utf-8-sig")

    by_lang = rate_table(p, "language_type")
    lang_pairs = wrong.groupby(["language_type", "error_pair"]).size().unstack(fill_value=0).reindex(columns=pairs,
                                                                                                    fill_value=0)
    by_lang = by_lang.merge(lang_pairs.reset_index(), on="language_type", how="left")
    by_lang.to_csv(OUT / "errors_by_language.csv", index=False, encoding="utf-8-sig")
    by_len = rate_table(p, "length_bucket")
    by_len.to_csv(OUT / "errors_by_length.csv", index=False, encoding="utf-8-sig")
    by_conf = rate_table(p, "confidence_bucket")
    by_conf.to_csv(OUT / "errors_by_confidence.csv", index=False, encoding="utf-8-sig")

    cat_counts = wrong["heuristic_categories"].str.split("; ").explode().value_counts()
    cat_df = pd.DataFrame({"heuristic_category": cat_counts.index, "errors_flagged": cat_counts.to_numpy(),
                           "share_of_errors": (cat_counts / len(wrong)).round(4).to_numpy()})
    cat_df["note"] = HEURISTIC_NOTE
    cat_df.to_csv(OUT / "heuristic_category_counts.csv", index=False, encoding="utf-8-sig")

    summary = {"model": "Multilingual A+", "split": "test (read-only analysis, not used for selection)",
               "n": int(len(p)), "errors": int(len(wrong)), "error_rate": round(float(p["wrong"].mean()), 4),
               "class_pair_errors": dict(zip(pairs, pair_counts.astype(int).tolist())),
               "heuristic_note": HEURISTIC_NOTE}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"A+ test errors: {len(wrong)} / {len(p)} ({summary['error_rate']:.2%})")
    print("\nConfusion matrix (rows=true, cols=pred):\n", pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES))
    print("\nRow-normalized:\n", pd.DataFrame(cm / cm.sum(1, keepdims=True), index=CLASS_NAMES,
                                               columns=CLASS_NAMES).round(3))
    print("\nClass-pair errors:\n", pair_df.to_string(index=False))
    print("\nBy language:\n", by_lang[["language_type", "n", "errors", "accuracy", "share_of_all_errors"]].to_string(index=False))
    print("\nBy length:\n", by_len.to_string(index=False))
    print("\nBy confidence:\n", by_conf.to_string(index=False))
    print(f"\nHeuristic categories ({HEURISTIC_NOTE}):\n", cat_df[["heuristic_category", "errors_flagged",
                                                                   "share_of_errors"]].to_string(index=False))
    print(f"\nsaved -> {OUT}")
    return summary


if __name__ == "__main__":
    main()
