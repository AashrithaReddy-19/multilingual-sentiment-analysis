"""Model registry and loading. Serves only models that were actually trained and saved.

Default model:
1. the model named in artifacts/multilingual/final_selection.json (written by pipeline.select_final_model after the
   validation-selected models were tested), if its files are present on this machine;
2. otherwise Multilingual A+ (word + char TF-IDF + LR), the best completed multilingual model;
3. otherwise Multilingual A, then the Telugu-only models (always labelled "Telugu-only model currently loaded").

Every prediction uses the CURRENT message only. No served model is context-aware (Model C was not trained).
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sentiment_core.cleaning import clean_text
from sentiment_core.labels import CLASS_NAMES
from sentiment_core.language import detect_language
from sentiment_core.text_variants import apply_variant

from .config import Settings

SCOPE_LABELS = {"multilingual": "Multilingual model (Telugu + English + code-mixed)",
                "telugu_only": "Telugu-only model currently loaded"}
LANG_TOKENS = {"english": "[LANG_ENGLISH]", "telugu": "[LANG_TELUGU]", "code_mixed": "[LANG_CODE_MIXED]"}
FALLBACK_ORDER = ("tfidf_plus_multilingual", "tfidf_multilingual", "xlmr_telugu", "tfidf_telugu")


def _json(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    except (OSError, ValueError):
        return None


@dataclass
class ModelSpec:
    id: str
    name: str
    kind: str          # "transformer" | "tfidf" | "ensemble" (weights global or per detected language)
    scope: str         # "multilingual" | "telugu_only"
    path: Path
    extra: dict = field(default_factory=dict)   # transformer: max_length/lang_tokens; ensemble: members/weights

    def files_present(self) -> bool:
        if self.kind == "transformer":
            return (self.path / "model.safetensors").exists() or (self.path / "pytorch_model.bin").exists()
        if self.kind == "ensemble":
            return self.path.exists()
        return self.path.exists()


class ModelService:
    def __init__(self, settings: Settings):
        s = settings
        self.s = s
        p3 = s.multi / "phase3_models"
        specs = [
            ModelSpec("tfidf_plus_multilingual", "Word + char TF-IDF + Logistic Regression (multilingual Model A+)",
                      "tfidf", "multilingual", s.multi / "model_a_plus" / "model_a_plus.joblib"),
            ModelSpec("tfidf_multilingual", "TF-IDF + Logistic Regression (multilingual Model A)", "tfidf",
                      "multilingual", s.multi / "model_a" / "tfidf_logreg_multilingual.joblib"),
            ModelSpec("xlmr_multilingual", "XLM-RoBERTa, no context (multilingual Model B, Phase 2 package)",
                      "transformer", "multilingual", s.multi / "model_b" / "xlm_roberta_multilingual"),
            ModelSpec("xlmr_telugu", "XLM-RoBERTa, no context (Telugu-only Model B)", "transformer",
                      "telugu_only", s.telugu_outputs / "xlm_roberta_no_context"),
            ModelSpec("tfidf_telugu", "TF-IDF + Logistic Regression (Telugu-only Model A)", "tfidf",
                      "telugu_only", s.telugu_outputs / "baseline_tfidf_logreg.joblib"),
        ]
        for fam, label in (("xlmr", "XLM-RoBERTa (best validation run, multilingual)"),
                           ("muril", "MuRIL (best validation run, multilingual)"),
                           ("indicbert", "IndicBERTv2 (best validation run, multilingual)"),
                           ("mdeberta", "mDeBERTa-v3 (best validation run, multilingual)"),
                           ("e5", "multilingual-e5-small (best validation run, multilingual)")):
            cfg = _json(p3 / fam / "phase3_config.json") or {}
            specs.append(ModelSpec(f"{fam}_phase3", label + (f" — {cfg['experiment_id']}" if cfg.get("experiment_id") else ""),
                                   "transformer", "multilingual", p3 / fam,
                                   {"max_length": cfg.get("max_length", 128), "lang_tokens": cfg.get("lang_tokens", False),
                                    "input_prefix": cfg.get("input_prefix", ""),
                                    "text_variant": cfg.get("text_variant", "v1_current"),
                                    "experiment_id": cfg.get("experiment_id")}))
        ens = _json(s.multi / "experiments" / "ensemble_validation_tuned" / "config.json")
        if ens:
            specs.append(ModelSpec("ensemble_phase3", "Validation-tuned ensemble (" + ", ".join(ens["members"]) + ")",
                                   "ensemble", "multilingual", s.multi / "experiments" / "ensemble_validation_tuned",
                                   {"member_ids": ens["member_ids"], "weights": ens["weights"]}))
        lens = _json(s.multi / "experiments" / "lang_ensemble_validation_tuned" / "config.json")
        if lens:
            specs.append(ModelSpec("lang_ensemble_phase3", "Language-aware ensemble (" + ", ".join(lens["members"]) + ")",
                                   "ensemble", "multilingual", s.multi / "experiments" / "lang_ensemble_validation_tuned",
                                   {"member_ids": lens["member_ids"], "weights_by_language": lens["weights_by_language"]}))
        self.specs: dict[str, ModelSpec] = {m.id: m for m in specs}
        self._loaded: dict[str, object] = {}
        self._lock = threading.RLock()
        self.load_error: dict[str, str] = {}
        self.final_selection = _json(s.multi / "final_selection.json")

    # ---- availability ------------------------------------------------------------------
    def _member_spec(self, member_id: str) -> str | None:
        """Map an ensemble member (experiment id) to a served model id."""
        if member_id == "model_a_plus":
            return "tfidf_plus_multilingual"
        for sid in ("xlmr_phase3", "muril_phase3", "indicbert_phase3", "mdeberta_phase3", "e5_phase3"):
            if self.specs[sid].extra.get("experiment_id") == member_id:
                return sid
        return None

    def available(self, mid: str) -> bool:
        spec = self.specs[mid]
        if spec.kind == "ensemble":
            members = [self._member_spec(m) for m in spec.extra["member_ids"]]
            return spec.files_present() and all(m and self.specs[m].files_present() for m in members)
        return spec.files_present()

    @property
    def default_id(self) -> str | None:
        sel = (self.final_selection or {}).get("served_model_id")
        if sel in self.specs and self.available(sel):
            return sel
        for mid in FALLBACK_ORDER:
            if self.available(mid):
                return mid
        return None

    def selection_note(self) -> str:
        sel = self.final_selection or {}
        sid = sel.get("served_model_id")
        if not sel:
            return "No Phase-3 final selection yet; serving Multilingual A+ (best completed multilingual model)."
        if sid in self.specs and self.available(sid):
            return f"Final selected model: {sel.get('selected_model_name', sid)}."
        return (f"Final selected model {sel.get('selected_model_name', sid)} is not available on this machine "
                f"(weights missing); falling back to {self.specs[self.default_id].name if self.default_id else 'none'}.")

    def info(self, mid: str) -> dict:
        m = self.specs[mid]
        return {"id": m.id, "name": m.name, "kind": m.kind, "scope": m.scope, "scope_label": SCOPE_LABELS[m.scope],
                "available": self.available(mid), "loaded": mid in self._loaded, "is_default": mid == self.default_id,
                "path": str(m.path), "context_aware": False}

    def list(self) -> list[dict]:
        return [self.info(mid) for mid in self.specs]

    # ---- loading -------------------------------------------------------------------------
    def _load(self, mid: str):
        with self._lock:
            if mid in self._loaded:
                return self._loaded[mid]
            spec = self.specs[mid]
            if spec.kind == "tfidf":
                import joblib
                obj = joblib.load(spec.path)
            elif spec.kind == "transformer":
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                tok = AutoTokenizer.from_pretrained(spec.path)
                model = AutoModelForSequenceClassification.from_pretrained(spec.path)
                model.eval()
                obj = (tok, model, torch)
            else:   # ensemble: load every member
                obj = [self._member_spec(m) for m in spec.extra["member_ids"]]
                for m in obj:
                    self._load(m)
            self._loaded[mid] = obj
            return obj

    def ensure_loaded(self, mid: str | None = None) -> str:
        mid = mid or self.default_id
        if mid is None:
            raise LookupError("No trained model is available.")
        if mid not in self.specs:
            raise KeyError(f"Unknown model id {mid!r}")
        if not self.available(mid):
            raise LookupError(f"Model {mid!r} has not been trained or its files are missing.")
        try:
            self._load(mid)
        except Exception as e:  # surface the reason instead of a bare 500
            self.load_error[mid] = str(e)
            raise
        return mid

    # ---- prediction ------------------------------------------------------------------------
    def _proba(self, mid: str, cleaned: list[str], langs: list[str]) -> np.ndarray:
        spec, obj = self.specs[mid], self._loaded[mid]
        if spec.kind == "tfidf":
            return obj.predict_proba(cleaned)[:, np.argsort(obj.classes_)]
        if spec.kind == "ensemble":
            member_probs = [self._proba(m, cleaned, langs) for m in obj]
            if "weights_by_language" not in spec.extra:
                return sum(w * p for w, p in zip(spec.extra["weights"], member_probs))
            wbl = spec.extra["weights_by_language"]      # routed by the heuristic language of each text
            w = np.array([wbl.get(lg, wbl["english"]) for lg in langs])          # (n, members)
            return sum(w[:, [j]] * p for j, p in enumerate(member_probs))
        tok, model, torch = obj
        # same input as training: [prefix][language token ]text-variant(cleaned)
        texts = apply_variant(cleaned, spec.extra.get("text_variant", "v1_current"))
        if spec.extra.get("lang_tokens"):
            texts = [f"{LANG_TOKENS.get(lg, LANG_TOKENS['english'])} {t}" for t, lg in zip(texts, langs)]
        if spec.extra.get("input_prefix"):
            texts = [spec.extra["input_prefix"] + t for t in texts]
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                enc = tok(texts[i:i + 32], truncation=True, max_length=spec.extra.get("max_length", 128),
                          padding=True, return_tensors="pt")
                out.append(torch.softmax(model(**enc).logits.float(), dim=-1).numpy())
        return np.vstack(out) if out else np.zeros((0, 3))

    def predict_proba(self, texts: list[str], mid: str | None = None) -> tuple[np.ndarray, list[str], str]:
        probs, cleaned, mid, _ = self.predict_full(texts, mid)
        return probs, cleaned, mid

    def predict_full(self, texts: list[str], mid: str | None = None):
        """Probabilities, cleaned texts, model id and heuristic language per text."""
        mid = self.ensure_loaded(mid)
        cleaned = [clean_text(t) or "" for t in texts]
        langs = [detect_language(t) for t in cleaned]
        return self._adjust(mid, self._proba(mid, cleaned, langs)), cleaned, mid, langs

    def _adjust(self, mid: str, probs: np.ndarray) -> np.ndarray:
        """Validation-learned post-processing recorded in final_selection.json, applied to the selected model only:
        per-class log-probability offsets if they were promoted, otherwise temperature scaling (labels unchanged)."""
        sel = self.final_selection or {}
        if not len(probs) or sel.get("served_model_id") != mid:
            return probs
        logp = np.log(np.clip(probs, 1e-12, 1.0))
        if sel.get("class_log_bias"):
            logp = logp + np.asarray(sel["class_log_bias"], dtype=float)
        elif sel.get("temperature"):
            logp = logp / float(sel["temperature"])
        else:
            return probs
        z = np.exp(logp - logp.max(1, keepdims=True))
        return z / z.sum(1, keepdims=True)

    @staticmethod
    def class_names() -> list[str]:
        return CLASS_NAMES
