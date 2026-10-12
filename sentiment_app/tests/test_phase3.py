"""Phase 3 helpers: conservative text variants, label-free signals, exact metric helpers."""
import numpy as np
from sklearn.metrics import f1_score

from pipeline.finalize_phase3 import fast_macro_f1, mcnemar
from sentiment_core.experiment_io import full_metrics
from sentiment_core.text_variants import emoticon_tokens, normalize_repeats, signal_features


def test_repeat_normalisation_keeps_telugu_and_double_letters():
    assert normalize_repeats("goooood baaaad!!!!!") == "good baad!!!"
    assert normalize_repeats("good too baagundi") == "good too baagundi"
    te = "చాలా బాగుంది ledu kaadu ❤❤❤"
    assert normalize_repeats(te) == te


def test_emoticons_become_tokens_only_when_standalone():
    assert emoticon_tokens("nice :) bad :(") == "nice EMO_POS bad EMO_NEG"
    assert emoticon_tokens("http://x") == "http://x"


def test_signal_features_ratios_bounded():
    f = signal_features(["Not good 😡!! ledu", "చాలా బాగుంది ❤", ""])
    assert f.shape == (3, 10)
    assert np.all(f[:, 8:] <= 1.0) and np.all(f[:, 8:] >= 0.0)
    assert f[0, 5] == 2  # "Not", "ledu"


def test_fast_macro_f1_matches_sklearn_and_full_metrics():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 3, 500), rng.integers(0, 3, 500)
    ref = f1_score(y, p, average="macro")
    assert abs(fast_macro_f1(y, p) - ref) < 1e-12
    assert abs(full_metrics(y, p)["macro_f1"] - ref) < 1e-12


def test_mcnemar_counts():
    y = np.array([0, 0, 1, 1, 2])
    r = mcnemar(y, np.array([0, 0, 1, 0, 2]), np.array([0, 1, 1, 1, 2]))
    assert (r["a_right_b_wrong"], r["a_wrong_b_right"]) == (1, 1)
