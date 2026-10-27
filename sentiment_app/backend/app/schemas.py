"""Pydantic request/response models."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Scope = Literal["multilingual", "telugu_only"]


class PredictRequest(BaseModel):
    text: str = Field(..., description="Telugu, English, Romanized Telugu, code-mixed text or emojis", examples=["సినిమా చాలా బాగుంది 👍"])
    model_id: str | None = Field(None, description="Optional model id from /api/models; default = final selected model, else Multilingual A+")

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("text must not be empty")
        return v


class ClassProbability(BaseModel):
    label: str
    probability: float


class LanguageGuess(BaseModel):
    detected: Literal["telugu", "english", "code_mixed", "unknown"]
    method: str


class ModelInfo(BaseModel):
    id: str
    name: str
    kind: Literal["transformer", "tfidf", "ensemble"]
    scope: Scope
    scope_label: str
    available: bool
    loaded: bool
    display_name: str | None = None
    test_accuracy: float | None = None
    test_macro_f1: float | None = None
    inference_available: bool = False
    is_default: bool
    path: str
    context_aware: bool = False


class PredictResponse(BaseModel):
    label: str
    label_id: int
    confidence: float
    probabilities: list[ClassProbability]
    cleaned_text: str
    model: ModelInfo
    language: LanguageGuess | None = None
    low_confidence: bool = False
    low_confidence_threshold: float = 0.5
    context_aware: bool = False
    note: str | None = None


class BatchInspectResponse(BaseModel):
    columns: list[str]
    n_rows: int
    preview: list[dict]
    suggested_column: str | None
    max_rows: int
