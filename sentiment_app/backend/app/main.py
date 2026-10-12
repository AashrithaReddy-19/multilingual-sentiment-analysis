"""FastAPI application. Interactive API documentation: http://127.0.0.1:8000/docs"""
from __future__ import annotations

import io
import threading
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from sentiment_core.context import REQUIRED_CONVERSATION_COLUMNS
from sentiment_core.labels import CLASS_NAMES, MODEL_C_SKIP
from sentiment_core.language import LANGUAGE_NOTE

from .artifacts import MODEL_C_EXPLANATION, Artifacts
from .config import settings
from .model_service import ModelService
from .schemas import BatchInspectResponse, ModelInfo, PredictRequest, PredictResponse

# Below this confidence the A+ test accuracy was 46% (experiments/error_analysis_model_a_plus/errors_by_confidence.csv).
LOW_CONFIDENCE_THRESHOLD = 0.5
CURRENT_MESSAGE_NOTE = "Prediction uses the current message only; it is not context-aware (Model C was not trained)."
TEXT_COLUMN_HINTS = ("text", "cleaned_text", "content", "comment", "comments", "message", "sentence", "tweet", "review")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = ModelService(settings)
    app.state.artifacts = Artifacts(settings)
    if settings.preload_model and app.state.models.default_id:
        # load in the background so /docs and data pages are usable immediately
        threading.Thread(target=lambda: _safe_preload(app.state.models), daemon=True).start()
    yield


def _safe_preload(svc: ModelService) -> None:
    try:
        svc.ensure_loaded()
    except Exception:
        pass  # reason is kept in svc.load_error and shown by /api/models


app = FastAPI(title="Context-Aware Sentiment Analysis API",
              description="Telugu, English and Telugu-English code-mixed sentiment (Negative / Neutral / Positive). "
                          "All metrics come from saved, executed results. Served models use the current message "
                          "only; context-aware Model C was not trained (no real conversation data).",
              version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=list(settings.cors_origins), allow_methods=["GET", "POST"],
                   allow_headers=["*"])


def get_models() -> ModelService:
    return app.state.models


def get_artifacts() -> Artifacts:
    return app.state.artifacts


@app.get("/api/health", tags=["system"])
def health():
    return {"status": "ok"}


@app.get("/api/models", response_model=list[ModelInfo], tags=["models"])
def list_models(svc: ModelService = Depends(get_models)):
    return svc.list()


@app.get("/api/dashboard", tags=["results"])
def dashboard(art: Artifacts = Depends(get_artifacts), svc: ModelService = Depends(get_models)):
    audit = art.audit()
    datasets = []
    for name, prov in (audit.get("provenance") or {}).items():
        stats = ((audit.get("cleaning") or {}).get("per_dataset") or {}).get(name, {})
        datasets.append({"name": name, "source": prov.get("source"), "licence_note": prov.get("licence_note"),
                         "checksums_match": prov.get("all_match"), "final_rows": stats.get("final_rows")})
    default = svc.default_id
    return {"title": "Context-Aware Sentiment Analysis for Telugu, English, and Telugu-English Code-Mixed Conversational Chat",
            "datasets": datasets, "runs": art.comparison(),
            "serving_model": svc.info(default) if default else None, "model_c": MODEL_C_SKIP,
            "selection_note": svc.selection_note(), "current_message_note": CURRENT_MESSAGE_NOTE}


@app.get("/api/comparison", tags=["results"])
def comparison(art: Artifacts = Depends(get_artifacts)):
    return art.comparison()


@app.get("/api/audit", tags=["results"])
def audit(art: Artifacts = Depends(get_artifacts)):
    return art.audit()


@app.get("/api/errors/sources", tags=["results"])
def error_sources(art: Artifacts = Depends(get_artifacts)):
    return art.error_sources()


@app.get("/api/errors", tags=["results"])
def errors(run: str, tag: str, q: str | None = None, language: str | None = None, actual: str | None = None,
           predicted: str | None = None, reason: str | None = None, offset: int = Query(0, ge=0),
           limit: int = Query(50, ge=1, le=200), art: Artifacts = Depends(get_artifacts)):
    return art.errors(run, tag, q, language, actual, predicted, reason, offset, limit)


@app.get("/api/context-status", tags=["results"])
def context_status():
    return {"model_c_status": MODEL_C_SKIP, "explanation": MODEL_C_EXPLANATION,
            "required_file": "conversations.csv", "required_columns": list(REQUIRED_CONVERSATION_COLUMNS),
            "column_rules": {
                "conversation_id": "same id for every message of one conversation",
                "message_order": "1, 2, 3 ... within each conversation (no gaps, no repeats)",
                "text": "the message",
                "label": "Negative, Neutral or Positive",
                "language_type": "telugu, english or code_mixed"},
            "context_format": "[PREVIOUS_2] <message t-2> [PREVIOUS_1] <message t-1> [CURRENT] <message t>",
            "conversations_available": False}


@app.get("/api/files/{run}/{name}", tags=["results"])
def files(run: str, name: str, art: Artifacts = Depends(get_artifacts)):
    p = art.file_path(run, name)
    if p is None:
        raise HTTPException(404, "file not found")
    return FileResponse(p, media_type="image/png")


def _resolve_model(svc: ModelService, model_id: str | None) -> str:
    try:
        return svc.ensure_loaded(model_id)
    except KeyError as e:
        raise HTTPException(400, str(e))
    except LookupError as e:
        raise HTTPException(503, str(e))


@app.post("/api/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest, svc: ModelService = Depends(get_models)):
    if len(req.text) > settings.max_text_chars:
        raise HTTPException(422, f"text longer than {settings.max_text_chars} characters")
    mid = _resolve_model(svc, req.model_id)
    probs, cleaned, mid, langs = svc.predict_full([req.text], mid)
    p = probs[0]
    k = int(p.argmax())
    return {"label": CLASS_NAMES[k], "label_id": k, "confidence": float(p[k]),
            "probabilities": [{"label": c, "probability": float(v)} for c, v in zip(CLASS_NAMES, p)],
            "cleaned_text": cleaned[0], "model": svc.info(mid),
            "language": {"detected": langs[0], "method": LANGUAGE_NOTE},
            "low_confidence": bool(p[k] < LOW_CONFIDENCE_THRESHOLD),
            "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
            "context_aware": False, "note": CURRENT_MESSAGE_NOTE}


async def _read_csv(file: UploadFile) -> pd.DataFrame:
    raw = await file.read()
    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(413, f"file larger than {settings.max_upload_mb} MB")
    if not raw.strip():
        raise HTTPException(400, "the uploaded file is empty")
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except UnicodeDecodeError:
        raise HTTPException(400, "the CSV must be UTF-8 encoded (needed for Telugu text)")
    except Exception as e:
        raise HTTPException(400, f"could not read CSV: {e}")
    if df.empty or not len(df.columns):
        raise HTTPException(400, "the CSV has no rows")
    return df


@app.post("/api/batch/inspect", response_model=BatchInspectResponse, tags=["prediction"])
async def batch_inspect(file: UploadFile = File(...)):
    df = await _read_csv(file)
    suggested = next((c for c in df.columns if c.strip().lower() in TEXT_COLUMN_HINTS), None)
    return {"columns": list(df.columns), "n_rows": len(df), "preview": df.head(5).to_dict(orient="records"),
            "suggested_column": suggested, "max_rows": settings.max_batch_rows}


@app.post("/api/batch/predict", tags=["prediction"])
async def batch_predict(file: UploadFile = File(...), text_column: str = Form(...), model_id: str | None = Form(None),
                        svc: ModelService = Depends(get_models)):
    df = await _read_csv(file)
    if text_column not in df.columns:
        raise HTTPException(400, f"column {text_column!r} not found; available: {list(df.columns)}")
    if len(df) > settings.max_batch_rows:
        raise HTTPException(413, f"{len(df)} rows; the limit is {settings.max_batch_rows}")
    mid = _resolve_model(svc, model_id)
    texts = df[text_column].astype(str).tolist()
    probs, cleaned, mid, langs = svc.predict_full(texts, mid)
    info = svc.info(mid)
    out = df.copy()
    out["cleaned_text"] = cleaned
    out["predicted_label"] = [CLASS_NAMES[int(i)] for i in probs.argmax(1)] if len(probs) else []
    out["confidence"] = probs.max(1).round(6) if len(probs) else []
    for i, c in enumerate(CLASS_NAMES):
        out[f"prob_{c}"] = probs[:, i].round(6) if len(probs) else []
    out["detected_language"] = langs
    out["low_confidence"] = (probs.max(1) < LOW_CONFIDENCE_THRESHOLD) if len(probs) else []
    empty = [not c for c in cleaned]
    out["empty_text"] = empty
    out.loc[empty, ["predicted_label"]] = ""          # no prediction for rows with no usable text
    out.loc[empty, ["confidence"] + [f"prob_{c}" for c in CLASS_NAMES]] = float("nan")
    out.loc[empty, ["low_confidence"]] = False
    out["model_id"], out["model_scope"] = info["id"], info["scope_label"]
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    data = ("﻿" + buf.getvalue()).encode("utf-8")
    return StreamingResponse(io.BytesIO(data), media_type="text/csv; charset=utf-8",
                             headers={"Content-Disposition": 'attachment; filename="predictions.csv"',
                                      "X-Model-Id": info["id"], "X-Model-Scope": info["scope"],
                                      "Access-Control-Expose-Headers": "X-Model-Id, X-Model-Scope"})
