"""Model A+ is shown only when its saved test macro F1 beats the original Model A (fixtures in a temp dir)."""
import json

import pytest

from app.artifacts import Artifacts
from app.config import Settings


def _write(root, name, macro_f1):
    d = root / "sentiment_app" / "artifacts" / "multilingual" / name
    d.mkdir(parents=True, exist_ok=True)
    m = {"accuracy": macro_f1, "macro_f1": macro_f1, "weighted_f1": macro_f1, "best_validation_macro_f1": None,
         "per_class": {}, "per_language": {}, "n": 1}
    (d / f"metrics_{name}.json").write_text(json.dumps(m), encoding="utf-8")


@pytest.mark.parametrize("plus,shown", [(0.70, True), (0.66, False), (0.60, False)])
def test_a_plus_listed_only_when_it_beats_baseline(tmp_path, plus, shown):
    _write(tmp_path, "model_a", 0.66)
    _write(tmp_path, "model_a_plus", plus)
    runs = Artifacts(Settings(nlp_root=tmp_path)).runs()
    ids = [m["id"] for m in runs["multilingual"]["models"]]
    assert ("A+" in ids) is shown
    assert ids[0] == "A"                                   # original baseline always kept


def test_a_plus_absent_without_results(tmp_path):
    _write(tmp_path, "model_a", 0.66)
    ids = [m["id"] for m in Artifacts(Settings(nlp_root=tmp_path)).runs()["multilingual"]["models"]]
    assert ids == ["A", "B", "B2", "E"]                    # Phase-3 rows are listed as pending
