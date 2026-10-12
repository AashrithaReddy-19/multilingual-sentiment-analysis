import math

import pytest

from sentiment_core.cleaning import NEGATIONS, clean_text


def test_removes_urls_html_and_extra_whitespace():
    assert clean_text("Great!!  <br> see https://x.com/a?b=1 and www.example.org  ") == "Great!! see and"


def test_unescapes_html_entities():
    assert clean_text("fish &amp; chips &quot;ok&quot;") == 'fish & chips "ok"'


def test_removes_control_characters_but_keeps_zero_width_joiners():
    zwnj = "‌"
    out = clean_text(f"a\x00b\x07c\tసత్{zwnj}యం")
    assert out == f"a b c సత్{zwnj}యం"


def test_preserves_telugu_emoji_and_punctuation():
    s = "నాకు ఇష్టం లేదు 😢 ఎందుకు ?"
    assert clean_text(s) == s


def test_preserves_romanized_telugu_and_code_mixed():
    s = "movie asalu bagaledu bro, waste!"
    assert clean_text(s) == s


@pytest.mark.parametrize("word", NEGATIONS)
def test_preserves_negation_words(word):
    assert word in clean_text(f"idi {word} baagundi").split()


def test_does_not_lowercase():
    assert clean_text("NOT Good") == "NOT Good"


@pytest.mark.parametrize("value", [None, float("nan"), "", "   ", "<br>", "https://only.url"])
def test_empty_inputs_become_none(value):
    assert clean_text(value) is None


def test_nan_is_not_stringified():
    assert clean_text(math.nan) is None
