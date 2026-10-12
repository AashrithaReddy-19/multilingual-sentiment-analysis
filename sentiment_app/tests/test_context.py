import pandas as pd

from sentiment_core.context import build_context, has_real_context


def _conv():
    # shuffled on purpose: build_context must sort by message_order
    return pd.DataFrame({"conversation_id": ["c1", "c1", "c1", "c1", "c2", "c2"],
                         "message_order": [3, 1, 4, 2, 1, 2],
                         "cleaned_text": ["m3", "m1", "m4", "m2", "x1", "x2"]})


def test_context_format_first_second_and_later_messages():
    out = build_context(_conv()).set_index("cleaned_text")["context_text"]
    assert out["m1"] == "[CURRENT] m1"
    assert out["m2"] == "[PREVIOUS_1] m1 [CURRENT] m2"
    assert out["m3"] == "[PREVIOUS_2] m1 [PREVIOUS_1] m2 [CURRENT] m3"
    assert out["m4"] == "[PREVIOUS_2] m2 [PREVIOUS_1] m3 [CURRENT] m4"


def test_context_never_uses_future_or_other_conversations():
    out = build_context(_conv()).set_index("cleaned_text")["context_text"]
    assert "m2" not in out["m1"] and "m3" not in out["m2"]
    assert out["x1"] == "[CURRENT] x1"
    assert "m" not in out["x2"].replace("[CURRENT]", "").replace("[PREVIOUS_1]", "")


def test_without_conversation_metadata_context_equals_cleaned_text():
    df = pd.DataFrame({"conversation_id": [pd.NA, pd.NA], "message_order": [pd.NA, pd.NA],
                       "cleaned_text": ["hello", "నమస్తే"]})
    out = build_context(df)
    assert (out["context_text"] == out["cleaned_text"]).all()
    assert has_real_context(out) is False


def test_missing_columns_fall_back_to_cleaned_text():
    out = build_context(pd.DataFrame({"cleaned_text": ["a", "b"]}))
    assert out["context_text"].tolist() == ["a", "b"]


def test_has_real_context_true_for_conversations():
    assert has_real_context(build_context(_conv())) is True
