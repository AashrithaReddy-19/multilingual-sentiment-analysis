"""Phase 1: final multilingual data pipeline (CPU, no training).

Verifies input checksums, applies only documented label mappings, cleans conservatively, removes
missing values and exact duplicate text-label pairs, keeps conflicting-label texts but groups them,
builds context_text (= cleaned_text: no conversations exist), splits 70/15/15 and saves reports.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402
from sentiment_core.cleaning import NEGATIONS, clean_text  # noqa: E402
from sentiment_core.context import build_context, has_real_context  # noqa: E402
from sentiment_core.labels import (CLASS_NAMES, CMTET_MAPPING, ID2LABEL, MODEL_C_SKIP,  # noqa: E402
                                   TESENT_MAPPING)
from sentiment_core.loaders import load_cmtet, load_tesent, load_tweeteval, verify_checksums  # noqa: E402
from sentiment_core.splitting import grouped_stratified_split, leakage_checks  # noqa: E402

STANDARD_COLUMNS = ["raw_text", "cleaned_text", "label", "language_type", "conversation_id", "message_order",
                    "context_text", "source_dataset"]
AUX_COLUMNS = ["source_label", "annotations", "language_tags", "original_split"]


def main() -> None:
    out = paths.DATA
    if (out / "train.csv").exists():
        raise SystemExit(f"{out} already contains a prepared split; refusing to overwrite.")
    out.mkdir(parents=True, exist_ok=True)

    folders = {"TeSent": paths.INPUTS / "telugu_tesent", "TweetEval": paths.INPUTS / "english_tweeteval",
               "CMTET": paths.INPUTS / "code_mixed_cmtet"}
    provenance = {k: verify_checksums(v) for k, v in folders.items()}
    for k, v in provenance.items():
        print(f"{k}: checksums match = {v['all_match']}")
        if not v["all_match"]:
            raise SystemExit(f"{k}: input files differ from the audited versions: {v['files']}")

    tes = load_tesent(folders["TeSent"])
    tw, tw_map = load_tweeteval(folders["TweetEval"])
    cm, cm_audit = load_cmtet(folders["CMTET"])
    mappings = {"TeSent": TESENT_MAPPING, "TweetEval": tw_map, "CMTET": CMTET_MAPPING}

    report = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "per_dataset": {},
              "cmtet_parser_audit": cm_audit}
    frames = []
    for name, df in (("TeSent", tes), ("TweetEval", tw), ("CMTET", cm)):
        st = {"initial_rows": len(df)}
        n = len(df); df = df[df["raw_text"].notna() & (df["raw_text"].astype(str).str.strip() != "")]
        st["removed_missing_text"] = n - len(df)
        n = len(df); df = df[df["label"].notna()]
        st["removed_missing_label"] = n - len(df)
        df = df.assign(cleaned_text=df["raw_text"].map(clean_text))
        n = len(df); df = df[df["cleaned_text"].notna()]
        st["removed_empty_after_cleaning"] = n - len(df)
        report["per_dataset"][name] = st
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    n = len(data)
    dup_mask = data.duplicated(["cleaned_text", "label"], keep="first")
    removed_dups = data[dup_mask]["source_dataset"].value_counts().to_dict()
    data = data[~dup_mask].reset_index(drop=True)
    for name in report["per_dataset"]:
        report["per_dataset"][name]["removed_exact_duplicate_text_label_pairs"] = int(removed_dups.get(name, 0))
        report["per_dataset"][name]["final_rows"] = int((data["source_dataset"] == name).sum())
    report["total_removed_exact_duplicates"] = n - len(data)

    n_labels = data.groupby("cleaned_text")["label"].nunique()
    conflict_texts = set(n_labels[n_labels > 1].index)
    data["conflicting_label_group"] = data["cleaned_text"].isin(conflict_texts)
    report["conflicting_labels"] = {
        "texts_with_more_than_one_label": len(conflict_texts),
        "rows_in_those_groups": int(data["conflicting_label_group"].sum()),
        "by_dataset": data[data["conflicting_label_group"]]["source_dataset"].value_counts().to_dict(),
        "decision": ("Kept: the disagreement is genuine annotation ambiguity and dropping it would hide it. "
                     "Every identical cleaned_text is one split group, so the same text can never appear in "
                     "more than one of train/validation/test."),
    }

    data["conversation_id"] = pd.NA
    data["message_order"] = pd.NA
    data = build_context(data)
    report["context"] = {"rows_with_earlier_message_context": int((data["context_text"] != data["cleaned_text"]).sum()),
                         "true_context_available": has_real_context(data), "model_c": MODEL_C_SKIP,
                         "note": "No conversation_id/message_order in any dataset, so context_text = cleaned_text."}

    data["split"] = grouped_stratified_split(data, seed=42)
    leaks = leakage_checks(data)
    if not leaks["passed"]:
        raise SystemExit(f"Leakage check failed: {leaks}")
    data.insert(0, "row_id", range(len(data)))

    cols = ["row_id"] + STANDARD_COLUMNS + [c for c in AUX_COLUMNS if c in data] + ["conflicting_label_group", "split"]
    data = data[cols]
    data.to_csv(out / "cleaned_all_data.csv", index=False, encoding="utf-8-sig")
    for s in ("train", "validation", "test"):
        data[data["split"] == s].drop(columns="split").to_csv(out / f"{s}.csv", index=False, encoding="utf-8-sig")

    dist = (data.assign(label_name=data["label"].map(ID2LABEL))
            .groupby(["split", "language_type", "label_name"]).size().rename("rows").reset_index())
    dist.to_csv(out / "class_distributions.csv", index=False, encoding="utf-8-sig")
    sizes = data["split"].value_counts().to_dict()
    report["split"] = {"method": "group-level stratified split (seed 42): every identical cleaned_text is one group; "
                                 "groups stratified by label x language_type; 70% train, then 15/15 validation/test",
                       "sizes": sizes, "fractions": {k: round(v / len(data), 4) for k, v in sizes.items()}}
    report["leakage_checks"] = leaks
    report["negation_words_preserved"] = {w: int(data["cleaned_text"].str.contains(rf"\b{w}\b", case=False).sum())
                                          for w in NEGATIONS}

    label_mapping = {"target": {str(k): v for k, v in ID2LABEL.items()}, "documented_mappings": mappings,
                     "sources": {"TeSent": "README.md: classes Positive/Negative/Neutral",
                                 "TweetEval": "mapping.txt (cardiffnlp/tweeteval)",
                                 "CMTET": "README.md tags NEG/NTL/POS + Kusampudi et al. RANLP 2021 Sec. 3.3"}}
    (out / "label_mapping.json").write_text(json.dumps(label_mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "leakage_checks.json").write_text(json.dumps(leaks, indent=2), encoding="utf-8")
    (out / "cleaning_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str),
                                              encoding="utf-8")
    audit_src = paths.NLP_ROOT / "audit_only" / "data_audit_report.json"
    final_audit = {"input_audit": json.loads(audit_src.read_text(encoding="utf-8")) if audit_src.exists() else None,
                   "checksums": provenance, "cleaning": report, "label_mapping": label_mapping,
                   "class_names": CLASS_NAMES}
    (out / "final_audit.json").write_text(json.dumps(final_audit, indent=2, ensure_ascii=False, default=str),
                                          encoding="utf-8")

    print(json.dumps({k: report[k] for k in ("per_dataset", "total_removed_exact_duplicates", "conflicting_labels",
                                             "split", "leakage_checks")}, indent=2, ensure_ascii=False, default=str))
    print(pd.crosstab([data["split"]], [data["language_type"], data["label"].map(ID2LABEL)]))
    print("Saved to", out)


if __name__ == "__main__":
    main()
