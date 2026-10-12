import pytest

from sentiment_core.labels import (CMTET_MAPPING, ID2LABEL, LABEL2ID, MODEL_C_SKIP, TESENT_MAPPING,
                                   LabelMappingError, map_label, parse_tweeteval_mapping, validate_target_label)


def test_target_scheme_is_exact():
    assert LABEL2ID == {"Negative": 0, "Neutral": 1, "Positive": 2}
    assert ID2LABEL == {0: "Negative", 1: "Neutral", 2: "Positive"}


def test_model_c_skip_text_is_exact():
    assert MODEL_C_SKIP == "Skipped — no real conversation_id/message_order available."


def test_tesent_mapping():
    assert [map_label(x, TESENT_MAPPING) for x in ["Negative", "Neutral", "Positive"]] == [0, 1, 2]


def test_cmtet_mapping_only_documented_tags():
    assert [map_label(x, CMTET_MAPPING) for x in ["NEG", "NTL", "POS"]] == [0, 1, 2]
    with pytest.raises(LabelMappingError):
        map_label("MIX", CMTET_MAPPING)


def test_parse_official_tweeteval_mapping():
    m = parse_tweeteval_mapping("0\tnegative\n1\tneutral\n2\tpositive\n")
    assert m == {"0": "Negative", "1": "Neutral", "2": "Positive"}
    assert [map_label(x, m) for x in ["0", "1", "2"]] == [0, 1, 2]


@pytest.mark.parametrize("bad", ["0\tnegative\n1\tpositive\n", "0 negative\n1 neutral\n2 happy\n", "a\tb\n", ""])
def test_rejects_incomplete_or_unexpected_mapping(bad):
    with pytest.raises(LabelMappingError):
        parse_tweeteval_mapping(bad)


def test_unknown_label_is_rejected_not_guessed():
    with pytest.raises(LabelMappingError):
        map_label("3", {"0": "Negative", "1": "Neutral", "2": "Positive"})
    with pytest.raises(LabelMappingError):
        map_label("hate", TESENT_MAPPING)


@pytest.mark.parametrize("value,expected", [(0, 0), (2, 2), ("Negative", 0), ("neutral", 1), (" POSITIVE ", 2)])
def test_validate_target_label(value, expected):
    assert validate_target_label(value) == expected


@pytest.mark.parametrize("value", [3, -1, True, "happy", "", None])
def test_validate_target_label_rejects(value):
    with pytest.raises(LabelMappingError):
        validate_target_label(value)
