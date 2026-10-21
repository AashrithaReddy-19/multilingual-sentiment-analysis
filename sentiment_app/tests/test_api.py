"""API tests with a small fake model (no real weights are loaded)."""
import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app, get_models
from sentiment_core.cleaning import clean_text


class FakeModels:
    default_id = "fake_telugu"
    _info = {"fake_telugu": {"id": "fake_telugu", "name": "Fake Telugu-only", "kind": "tfidf", "scope": "telugu_only",
                             "scope_label": "Telugu-only model currently loaded", "available": True, "loaded": True,
                             "is_default": True, "path": "-"},
             "fake_missing": {"id": "fake_missing", "name": "Not trained", "kind": "transformer",
                              "scope": "multilingual", "scope_label": "Multilingual model", "available": False,
                              "loaded": False, "is_default": False, "path": "-"}}

    def ensure_loaded(self, mid=None):
        mid = mid or self.default_id
        if mid not in self._info:
            raise KeyError(f"Unknown model id {mid!r}")
        if not self._info[mid]["available"]:
            raise LookupError("not trained")
        return mid

    def predict_proba(self, texts, mid=None):
        cleaned = [clean_text(t) or "" for t in texts]
        probs = np.array([[0.1, 0.2, 0.7] if "good" in c else [0.4, 0.35, 0.25] if "meh" in c else [0.6, 0.3, 0.1]
                          for c in cleaned]).reshape(-1, 3)
        return probs, cleaned, self.ensure_loaded(mid)

    def predict_full(self, texts, mid=None):
        from sentiment_core.language import detect_language
        probs, cleaned, mid = self.predict_proba(texts, mid)
        return probs, cleaned, mid, [detect_language(c) for c in cleaned]

    def selection_note(self):
        return "fake"

    def info(self, mid):
        return self._info[mid]

    def list(self):
        return list(self._info.values())


@pytest.fixture()
def client():
    with TestClient(app) as c:
        app.dependency_overrides[get_models] = lambda: FakeModels()
        yield c
    app.dependency_overrides.clear()


def test_predict_returns_label_probabilities_and_model(client):
    r = client.post("/api/predict", json={"text": "  very good <b>movie</b> 👍 "})
    assert r.status_code == 200
    j = r.json()
    assert j["label"] == "Positive" and j["label_id"] == 2
    assert abs(sum(p["probability"] for p in j["probabilities"]) - 1) < 1e-6
    assert [p["label"] for p in j["probabilities"]] == ["Negative", "Neutral", "Positive"]
    assert j["cleaned_text"] == "very good movie 👍"
    assert j["model"]["scope_label"] == "Telugu-only model currently loaded"
    assert j["language"]["detected"] == "english" and "Heuristic" in j["language"]["method"]
    assert j["low_confidence"] is False and j["context_aware"] is False


def test_predict_flags_low_confidence_and_detects_telugu(client):
    j = client.post("/api/predict", json={"text": "meh సినిమా"}).json()
    assert j["low_confidence"] is True and j["confidence"] < j["low_confidence_threshold"]
    assert j["language"]["detected"] == "code_mixed"


@pytest.mark.parametrize("payload", [{"text": ""}, {"text": "   "}, {}, {"text": None}, {"text": 123}])
def test_predict_rejects_invalid_text(client, payload):
    assert client.post("/api/predict", json=payload).status_code == 422


def test_predict_rejects_too_long_text(client):
    assert client.post("/api/predict", json={"text": "a" * 2001}).status_code == 422


def test_predict_unknown_and_untrained_models(client):
    assert client.post("/api/predict", json={"text": "hi", "model_id": "nope"}).status_code == 400
    assert client.post("/api/predict", json={"text": "hi", "model_id": "fake_missing"}).status_code == 503


def _csv(text: str) -> dict:
    return {"file": ("in.csv", text.encode("utf-8"), "text/csv")}


def test_batch_inspect_suggests_text_column(client):
    r = client.post("/api/batch/inspect", files=_csv("id,message\n1,good\n2,సినిమా\n"))
    assert r.status_code == 200
    assert r.json()["columns"] == ["id", "message"] and r.json()["suggested_column"] == "message"
    assert r.json()["n_rows"] == 2


def test_batch_predict_returns_csv_with_predictions(client):
    r = client.post("/api/batch/predict", files=_csv("id,message\n1,good day\n2,చెడ్డ రోజు\n3,\n"),
                    data={"text_column": "message"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert r.headers["x-model-id"] == "fake_telugu"
    df = pd.read_csv(io.BytesIO(r.content), encoding="utf-8-sig", keep_default_na=False)
    assert {"predicted_label", "confidence", "prob_Negative", "prob_Neutral", "prob_Positive", "model_scope",
            "detected_language", "low_confidence"} <= set(df.columns)
    assert df["detected_language"].tolist()[:2] == ["english", "telugu"]
    assert df["predicted_label"].tolist() == ["Positive", "Negative", ""]
    assert df["empty_text"].astype(str).tolist() == ["False", "False", "True"]
    assert df["message"].tolist()[1] == "చెడ్డ రోజు"            # Telugu survives the round trip


def test_batch_predict_missing_column(client):
    r = client.post("/api/batch/predict", files=_csv("id,text\n1,good\n"), data={"text_column": "message"})
    assert r.status_code == 400 and "not found" in r.json()["detail"]


def test_batch_predict_row_limit(client):
    rows = "\n".join(f"{i},good" for i in range(6))
    assert client.post("/api/batch/predict", files=_csv("id,text\n" + rows), data={"text_column": "text"}).status_code == 413


def test_batch_rejects_empty_and_non_utf8(client):
    assert client.post("/api/batch/inspect", files=_csv("")).status_code == 400
    bad = {"file": ("in.csv", "text\nతెలుగు\n".encode("utf-16"), "text/csv")}
    assert client.post("/api/batch/inspect", files=bad).status_code == 400


def test_context_status_reports_exact_skip(client):
    j = client.get("/api/context-status").json()
    assert j["model_c_status"] == "Skipped — no real conversation_id/message_order available."
    assert j["required_columns"] == ["conversation_id", "message_order", "text", "label", "language_type"]
    assert j["conversations_available"] is False


def test_file_endpoint_blocks_unknown_paths(client):
    assert client.get("/api/files/multilingual/..%2F..%2Fsecret.png").status_code == 404
    assert client.get("/api/files/nope/confusion_matrix_model_a.png").status_code == 404


def test_docs_available(client):
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").json()["info"]["title"].startswith("Context-Aware")


def test_comparison_uses_only_saved_results_and_exact_status_texts(client):
    runs = {r["run"]: r for r in client.get("/api/comparison").json()}
    multi = {m["id"]: m for m in runs["multilingual"]["models"]}
    from app.config import settings
    sel = settings.multi / "experiments" / "transformers" / "_selected_for_test.json"
    if sel.exists() or (settings.multi / "model_b" / "metrics_model_b.json").exists():   # real imported Colab run
        assert multi["B"]["status"] == "trained" and isinstance(multi["B"]["macro_f1"], float)
    else:
        for mid in ("B", "B-E5", "B2", "B-IndicBERT", "B-mDeBERTa", "E", "E-lang"):
            assert multi[mid]["status"] == "pending" and "macro_f1" not in multi[mid]
        assert multi["B"]["note"] == "Not trained / pending GPU run."
    assert multi["A+"]["macro_precision"] is not None and multi["A+"]["macro_recall"] is not None
    for run in runs.values():
        c = next(m for m in run["models"] if m["id"] == "C")
        assert c["status"] == "skipped" and c["note"] == "Skipped — no real conversation_id/message_order available."
        assert "macro_f1" not in c
