"""Phase-3 search helpers: class weights, language sampler, ensemble grids/search, language-aware routing, served input."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend" / "pipeline"))
import colab_runner as cr  # noqa: E402
import train_transformer as tt  # noqa: E402
from pipeline import finalize_phase3 as fp  # noqa: E402


def test_class_weight_schemes_keep_loss_scale_and_order():
    y = np.array([0] * 100 + [1] * 300 + [2] * 200)
    n = np.bincount(y)
    for scheme in ("inverse", "sqrt_inverse", "effective"):
        w = tt.class_weights(y, scheme)
        assert abs((w * n).sum() / n.sum() - 1) < 1e-5            # count-weighted mean weight is 1
        assert w[0] > w[2] > w[1]                                   # rarer class -> larger weight
    assert np.allclose(tt.class_weights(y, "inverse"), n.sum() / (3 * n))


def test_language_sampler_shares():
    langs = ["english"] * 600 + ["telugu"] * 300 + ["code_mixed"] * 100
    s = pd.Series(langs)
    for scheme, expect in (("lang_balanced", [1 / 3] * 3), ("lang_sqrt", np.sqrt([600, 300, 100]) / np.sqrt([600, 300, 100]).sum())):
        w = tt.language_sample_weights(langs, scheme)
        share = [w[(s == lg).to_numpy()].sum() / w.sum() for lg in ("english", "telugu", "code_mixed")]
        assert np.allclose(share, expect)


def test_lang_token_and_variant_input_order():
    df = pd.DataFrame({"cleaned_text": ["sooooo good :)"], "language_type": ["english"]})
    assert tt.model_inputs(df, True, variant="v4_repeat_norm_emoticons") == ["[LANG_ENGLISH] soo good EMO_POS"]


def test_exp_id_defaults_unchanged_and_new_factors_named():
    base = cr.exp_id(dict(cr.DEFAULT, family="xlmr"))
    assert base == "xlmr_lr2e-05_ce_ep3p2_bs16x1_wd0.01_wu0.1_len128"
    eid = cr.exp_id(dict(cr.DEFAULT, family="e5", loss="wce", class_weight_scheme="effective", sampler="lang_sqrt",
                         text_variant="v3_repeat_norm", scheduler="cosine", freeze_lower_layers=3,
                         freeze_mode="permanent"))
    for part in ("wceeff", "cosine", "langsqrt", "v3", "freeze3perm"):
        assert part in eid


def test_ensemble_grids_sum_to_one():
    for k, zero in ((2, False), (2, True), (3, False), (3, True)):
        g = fp.ensemble_grid(k, with_zero=zero)
        assert g and all(abs(sum(w) - 1) < 1e-9 and len(w) == k for w in g)
    assert (0.35, 0.65) in fp.ensemble_grid(2)
    assert all(min(w) >= 0.1 for w in fp.ensemble_grid(3))


def _member(name, probs):
    return {"id": name, "name": name, "val": probs}


def test_search_ensembles_and_language_routing():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 600)
    langs = np.array(["english", "telugu", "code_mixed"] * 200)
    onehot = np.eye(3)[y]
    good_en = np.where((langs == "english")[:, None], 0.7 * onehot + 0.1, rng.dirichlet([1, 1, 1], 600))
    good_te = np.where((langs != "english")[:, None], 0.7 * onehot + 0.1, rng.dirichlet([1, 1, 1], 600))
    noise = rng.dirichlet([1, 1, 1], 600)
    pool = [_member("en", good_en), _member("te", good_te), _member("noise", noise)]
    ens = fp.search_ensembles(pool, y)
    assert {m["id"] for m in ens["members"][:2]} == {"en", "te"}
    lw, lm = fp.search_language_weights(ens["members"], ens["weights"], y, langs)
    assert lm["macro_f1"] >= ens["val_metrics"]["macro_f1"]
    idx = [m["id"] for m in ens["members"]]
    assert lw["english"][idx.index("en")] > lw["english"][idx.index("te")]
    assert lw["telugu"][idx.index("te")] > lw["telugu"][idx.index("en")]
    routed = fp.routed(ens["members"], lw, langs, "val")
    assert np.allclose(routed.sum(1), 1)


def test_target_check_flags():
    m = {"accuracy": 0.76, "macro_f1": 0.74,
         "per_class": {k: {"f1": f} for k, f in zip(("Negative", "Neutral", "Positive"), (0.8, 0.7, 0.81))},
         "per_language": {lg: {"macro_f1": f} for lg, f in zip(fp.LANGS, (0.7, 0.72, 0.8))}}
    t = fp.target_check(m, "x")
    met = {c["target"]: c["met"] for c in t["checks"]}
    assert met == {"Accuracy >= 0.75": True, "Accuracy >= 0.80": False, "Macro F1 >= 0.75": False,
                   "Macro F1 >= 0.80": False, "All class F1 >= 0.80": False}
    assert t["limiting_class"] == "Neutral" and t["limiting_language"] == "English"


def test_service_language_aware_ensemble_routes_by_detected_language(tmp_path):
    from app.config import Settings
    from app.model_service import ModelService
    multi = tmp_path / "sentiment_app" / "artifacts" / "multilingual"
    d = multi / "experiments" / "lang_ensemble_validation_tuned"
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"members": ["a", "b"], "member_ids": ["model_a_plus", "x"],
                                               "weights_by_language": {"english": [1.0, 0.0], "telugu": [0.0, 1.0],
                                                                       "code_mixed": [0.5, 0.5]}}))
    svc = ModelService(Settings(nlp_root=tmp_path))
    spec = svc.specs["lang_ensemble_phase3"]
    svc._loaded["m1"], svc._loaded["m2"] = "a", "b"
    svc._loaded["lang_ensemble_phase3"] = ["m1", "m2"]
    fixed = {"m1": np.array([1.0, 0, 0]), "m2": np.array([0, 0, 1.0])}
    svc._proba_orig = svc._proba
    svc._proba = lambda mid, cleaned, langs: (np.tile(fixed[mid], (len(cleaned), 1)) if mid in fixed
                                              else svc._proba_orig(mid, cleaned, langs))
    p = svc._proba_orig(spec.id, ["x", "y", "z"], ["english", "telugu", "code_mixed"])
    assert np.allclose(p, [[1, 0, 0], [0, 0, 1], [0.5, 0, 0.5]])


def test_pair_grid_contains_the_requested_a_plus_weights():
    pairs = fp.ensemble_grid(2)
    for a in (0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        assert (a, round(1 - a, 2)) in pairs


def test_complexity_counts_transformers_and_routing():
    cands = [{"id": "model_a_plus", "kind": None}, {"id": "e5", "kind": "transformer"}]
    assert fp.complexity(cands[0], cands) == 0 and fp.complexity(cands[1], cands) == 1
    pair = {"kind": "ensemble", "member_ids": ["model_a_plus", "e5"]}
    trio = {"kind": "ensemble", "member_ids": ["e5", "xlmr", "muril"]}
    lang = {"kind": "lang_ensemble", "member_ids": ["model_a_plus", "e5"]}
    inc = {"kind": "incumbent", "member_ids": ["model_a_plus", "e5"]}
    assert fp.complexity(pair, cands) == fp.complexity(inc, cands) == 1.5
    assert fp.complexity(lang, cands) == 2.0 and fp.complexity(trio, cands) == 3.5


def _cand(cid, kind, val, test, member_ids=None):
    return {"id": cid, "name": cid, "kind": kind, "val": val, "test": test, "best_val": 0.0,
            "member_ids": member_ids or [], "family": "xlmr"}


def _select(cands, y_val, y_test):
    results = {c["id"]: {"m": fp.full_metrics(y_test, c["test"].argmax(1), np.resize(np.array(list(fp.LANGS)), len(y_test))),
                         "out": {}} for c in cands}
    inc = next(c for c in cands if c["kind"] == "incumbent")
    pi = inc["test"].argmax(1)
    stats = {f"{c['name']} vs {inc['name']}": {"mcnemar": fp.mcnemar(y_test, pi, c["test"].argmax(1)),
                                               "bootstrap": fp.bootstrap(y_test, pi, c["test"].argmax(1), n=200)}
             for c in cands if c is not inc}
    return fp.select_final(cands, results, stats, y_val)


def test_incumbent_kept_unless_challenger_is_significantly_better_on_test():
    rng = np.random.default_rng(1)
    yv, yt = rng.integers(0, 3, 900), rng.integers(0, 3, 900)
    oh = lambda y, acc: np.where((rng.random(len(y)) < acc)[:, None], np.eye(3)[y], np.eye(3)[(y + 1) % 3])
    inc = _cand("production_incumbent", "incumbent", oh(yv, 0.75), oh(yt, 0.75), ["model_a_plus", "e5"])
    a_plus = _cand("model_a_plus", None, oh(yv, 0.70), oh(yt, 0.70))
    # better on validation by > 0.005 but NOT better on test -> incumbent stays
    weak = _cand("xlmr_run", "transformer", oh(yv, 0.80), inc["test"].copy())
    sel = _select([a_plus, inc, weak], yv, yt)
    assert sel["selected_id"] == "production_incumbent" and sel["served_model_id"] == "production_ensemble"
    # better on validation AND significantly better on test -> replaces the incumbent
    strong = _cand("xlmr_run", "transformer", oh(yv, 0.85), oh(yt, 0.90))
    sel = _select([a_plus, inc, strong], yv, yt)
    assert sel["selected_id"] == "xlmr_run" and sel["served_model_id"] == "xlmr_phase3"
