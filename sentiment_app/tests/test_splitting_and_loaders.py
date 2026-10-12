import numpy as np
import pandas as pd
import pytest

from sentiment_core.loaders import load_tweeteval, parse_cmtet
from sentiment_core.splitting import grouped_stratified_split, leakage_checks


def _frame(n=600, dup_every=10):
    rng = np.random.default_rng(0)
    texts = [f"text {i // dup_every if i % dup_every == 0 else i}" for i in range(n)]
    return pd.DataFrame({"cleaned_text": texts, "label": rng.integers(0, 3, n),
                         "language_type": rng.choice(["telugu", "english", "code_mixed"], n),
                         "conversation_id": pd.NA})


def test_split_is_70_15_15_and_reproducible():
    df = _frame()
    s1, s2 = grouped_stratified_split(df, seed=42), grouped_stratified_split(df, seed=42)
    assert s1.equals(s2)
    frac = s1.value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.03 and abs(frac["validation"] - 0.15) < 0.03 and abs(frac["test"] - 0.15) < 0.03


def test_identical_text_with_conflicting_labels_stays_in_one_split():
    df = pd.concat([_frame(), pd.DataFrame({"cleaned_text": ["same text"] * 3, "label": [0, 1, 2],
                                            "language_type": "telugu", "conversation_id": pd.NA})],
                   ignore_index=True)
    df["split"] = grouped_stratified_split(df)
    assert df.loc[df["cleaned_text"] == "same text", "split"].nunique() == 1
    checks = leakage_checks(df)
    assert checks["passed"] and sum(checks["cleaned_text"].values()) == 0


def test_conversations_are_grouped():
    df = _frame(300)
    df["conversation_id"] = [f"c{i // 4}" for i in range(len(df))]
    df["split"] = grouped_stratified_split(df)
    assert df.groupby("conversation_id")["split"].nunique().max() == 1


def test_leakage_check_detects_overlap():
    df = pd.DataFrame({"cleaned_text": ["a", "a", "b"], "conversation_id": pd.NA,
                       "split": ["train", "test", "validation"]})
    assert leakage_checks(df)["passed"] is False


def test_cmtet_parser(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("NTL: We need review\nen en en\n\n\nNEG: worst ga undi\nen te te\n\n\nPOS: super bro 👍\nen en univ\n",
                 encoding="utf-8")
    df, audit = parse_cmtet(p)
    assert df["source_label"].tolist() == ["NTL", "NEG", "POS"]
    assert df["raw_text"].tolist() == ["We need review", "worst ga undi", "super bro 👍"]
    assert audit["issues"] == {} and audit["unknown_tags"] == {}


def test_cmtet_parser_reports_unknown_tags(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("XYZ: odd line\nen en\n", encoding="utf-8")
    _, audit = parse_cmtet(p)
    assert audit["unknown_tags"] == {"XYZ": 1}


def test_tweeteval_loader_uses_mapping_and_checks_alignment(tmp_path):
    (tmp_path / "mapping.txt").write_text("0\tnegative\n1\tneutral\n2\tpositive\n", encoding="utf-8")
    for s in ("train", "val", "test"):
        (tmp_path / f"{s}_text.txt").write_text("good day\nbad day\n", encoding="utf-8")
        (tmp_path / f"{s}_labels.txt").write_text("2\n0\n", encoding="utf-8")
    df, mapping = load_tweeteval(tmp_path)
    assert mapping["2"] == "Positive" and df["label"].tolist()[:2] == [2, 0]
    (tmp_path / "test_labels.txt").write_text("2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_tweeteval(tmp_path)
