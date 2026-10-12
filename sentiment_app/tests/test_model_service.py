"""Default-model rules: final selection if its files exist, otherwise Multilingual A+ (never the Telugu-only model
while a multilingual model exists). Uses empty placeholder files in a temp dir; nothing is loaded."""
import json

from app.config import Settings
from app.model_service import ModelService
from sentiment_core.language import detect_language


def _touch(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def _tree(tmp_path, a_plus=True):
    multi = tmp_path / "sentiment_app" / "artifacts" / "multilingual"
    _touch(multi / "model_a" / "tfidf_logreg_multilingual.joblib")
    if a_plus:
        _touch(multi / "model_a_plus" / "model_a_plus.joblib")
    _touch(tmp_path / "outputs" / "xlm_roberta_no_context" / "model.safetensors")   # Telugu-only XLM-R
    return multi


def test_a_plus_is_default_without_final_selection(tmp_path):
    _tree(tmp_path)
    svc = ModelService(Settings(nlp_root=tmp_path))
    assert svc.default_id == "tfidf_plus_multilingual"
    assert "A+" in svc.selection_note()
    assert all(m["context_aware"] is False for m in svc.list())


def test_multilingual_a_before_telugu_only(tmp_path):
    _tree(tmp_path, a_plus=False)
    assert ModelService(Settings(nlp_root=tmp_path)).default_id == "tfidf_multilingual"


def test_final_selection_used_only_when_files_exist(tmp_path):
    multi = _tree(tmp_path)
    (multi / "final_selection.json").write_text(json.dumps(
        {"served_model_id": "xlmr_phase3", "selected_model_name": "XLM-R"}), encoding="utf-8")
    svc = ModelService(Settings(nlp_root=tmp_path))
    assert svc.default_id == "tfidf_plus_multilingual"            # weights missing -> clean fallback
    assert "not available on this machine" in svc.selection_note()
    _touch(multi / "phase3_models" / "xlmr" / "model.safetensors")
    assert ModelService(Settings(nlp_root=tmp_path)).default_id == "xlmr_phase3"


def test_ensemble_needs_all_members(tmp_path):
    multi = _tree(tmp_path)
    ens = multi / "experiments" / "ensemble_validation_tuned"
    ens.mkdir(parents=True)
    (ens / "config.json").write_text(json.dumps({"members": ["Multilingual A+", "XLM-R"],
                                                 "member_ids": ["model_a_plus", "xlmr_x"], "weights": [0.3, 0.7]}))
    svc = ModelService(Settings(nlp_root=tmp_path))
    assert svc.available("ensemble_phase3") is False
    (multi / "phase3_models" / "xlmr").mkdir(parents=True)
    (multi / "phase3_models" / "xlmr" / "phase3_config.json").write_text(json.dumps({"experiment_id": "xlmr_x"}))
    _touch(multi / "phase3_models" / "xlmr" / "model.safetensors")
    assert ModelService(Settings(nlp_root=tmp_path)).available("ensemble_phase3") is True


def test_language_heuristic():
    assert detect_language("ఈ సినిమా చాలా బాగుంది 👍") == "telugu"
    assert detect_language("The delivery was late again. Not happy.") == "english"
    assert detect_language("movie asalu bagaledu, time waste") == "code_mixed"
    assert detect_language("నాకు ఈ movie నచ్చింది") == "code_mixed"
    assert detect_language("👍👍") == "unknown"
