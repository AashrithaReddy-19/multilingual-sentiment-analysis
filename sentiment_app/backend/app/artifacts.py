"""Read-only access to saved, executed results and audit files. Nothing here computes new metrics."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sentiment_core.evaluation import HEURISTIC_NOTE, heuristic_error_reason
from sentiment_core.labels import CLASS_NAMES, LABEL2ID, MODEL_B_PENDING, MODEL_C_SKIP

from .config import Settings

# Presentation layer for the web app (the saved records are unchanged). Every TRAINED model with real test metrics is
# dashboard_visible, whatever its score; pending / skipped / metric-less entries are never shown. "category" only
# describes a model (production, evaluated = >= 0.70 accuracy and macro F1, baseline = below that).
CATEGORY_MIN = 0.70
DISPLAY = {   # (run, row id) -> (display name, short description)
    ("multilingual", "PROD"): ("Production Ensemble", "Language-aware: Enhanced TF-IDF + two Multilingual E5 models"),
    ("multilingual", "P"): ("Previous Production Ensemble", "Enhanced TF-IDF + Multilingual E5 (equal weights)"),
    ("multilingual", "E"): ("Optimized Ensemble", "Enhanced TF-IDF + two Multilingual E5 models (global weights)"),
    ("multilingual", "E-lang"): ("Language-Aware Ensemble (v1)", "Per-language weights, heuristic language routing"),
    ("multilingual", "B-E5"): ("Multilingual E5", "Fine-tuned multilingual-e5-small"),
    ("multilingual", "A+"): ("Enhanced TF-IDF", "Word + character TF-IDF + logistic regression"),
    ("multilingual", "A"): ("TF-IDF Baseline", "Word TF-IDF + logistic regression"),
    ("telugu_only", "A"): ("Telugu Model A", "TF-IDF + logistic regression (Telugu-only split)"),
    ("telugu_only", "B"): ("Telugu Model B", "XLM-RoBERTa fine-tuned (Telugu-only split)"),
    ("telugu_only", "A+"): ("Telugu Model A+", "Character TF-IDF + logistic regression (Telugu-only split)"),
}


MODEL_C_EXPLANATION = ("None of the datasets contain conversation_id/message_order, so context_text equals "
                       "cleaned_text for every row. Training Model C would duplicate Model B (same input, same "
                       "settings), so it was intentionally skipped. It is not a failure.")


def _json(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


class Artifacts:
    def __init__(self, settings: Settings):
        self.s = settings
        self._cache: dict = {}

    # ---- runs and their metric files -------------------------------------------------
    def runs(self) -> dict:
        s = self.s
        return {
            "multilingual": {
                "title": "Multilingual (TeSent + TweetEval + CMTET)",
                "models": [
                    {"id": "A", "name": "TF-IDF + Logistic Regression", "input": "cleaned_text",
                     "metrics": s.multi / "model_a" / "metrics_model_a.json", "tag": "model_a",
                     "dir": s.multi / "model_a", "pending_note": None},
                    *self._model_a_plus(),
                    *self._production_row(),
                    *self._phase3(),
                ]},
            "telugu_only": {
                "title": "Telugu-only (TeSent)",
                "models": [
                    {"id": "A", "name": "TF-IDF + Logistic Regression", "input": "cleaned_text",
                     "metrics": s.telugu_outputs / "metrics_baseline.json", "tag": "baseline",
                     "dir": s.telugu_outputs, "pending_note": None},
                    {"id": "B", "name": "XLM-RoBERTa (no context)", "input": "cleaned_text",
                     "metrics": s.telugu_outputs / "metrics_no_context.json", "tag": "no_context",
                     "dir": s.telugu_outputs, "pending_note": None},
                    *self._telugu_refined(),
                ]},
        }

    def _model_a_plus(self) -> list[dict]:
        """Model A+ is listed only when it beat the original Model A on test macro F1 (real saved results)."""
        plus, base = _json(self.s.multi / "model_a_plus" / "metrics_model_a_plus.json"),             _json(self.s.multi / "model_a" / "metrics_model_a.json")
        if not plus or not base or plus["macro_f1"] <= base["macro_f1"]:
            return []
        return [{"id": "A+", "name": "Word + char TF-IDF + Logistic Regression", "input": "cleaned_text",
                 "metrics": self.s.multi / "model_a_plus" / "metrics_model_a_plus.json", "tag": "model_a_plus",
                 "dir": self.s.multi / "model_a_plus", "pending_note": None,
                 "baseline_note": "Improved variant selected on validation; the original Model A baseline is kept unchanged."}]

    def _telugu_refined(self) -> list[dict]:
        """Compact validation-selected refinement of the Telugu-only TF-IDF baseline (pipeline.telugu_tfidf_refine)."""
        d = self.s.nlp_root / "sentiment_app" / "artifacts" / "telugu_only_refine"
        if not (d / "metrics_telugu_char_tfidf.json").exists():
            return []
        return [{"id": "A+", "name": "Character TF-IDF + logistic regression", "input": "cleaned_text",
                 "metrics": d / "metrics_telugu_char_tfidf.json", "tag": "telugu_char_tfidf", "dir": d, "pending_note": None}]

    def _production_row(self) -> list[dict]:
        """The language-aware ensemble promoted on 2026-10-27 (experiments/final/lang_aware_v2)."""
        d = self.s.multi / "experiments" / "final" / "lang_aware_v2"
        if not (d / "metrics_lang_aware_v2.json").exists():
            return []
        return [{"id": "PROD", "name": "Language-aware ensemble (A+ / E5 / E5-b, learned language detector)",
                 "input": "cleaned_text", "metrics": d / "metrics_lang_aware_v2.json", "tag": "lang_aware_v2", "dir": d,
                 "pending_note": None}]

    def _phase3(self) -> list[dict]:
        """Phase-3 Transformer rows (validation-selected per family) and the validation-tuned ensemble.

        Metrics come only from experiments/final/<id>/metrics_<id>.json written by pipeline.finalize_phase3 after a
        real (non-smoke, split-verified) Colab import. Otherwise the row is shown as pending.
        """
        exp = self.s.multi / "experiments"
        sel = _json(exp / "transformers" / "_selected_for_test.json") or {}
        rows = []
        for fam, mid, name in (("xlmr", "B", "XLM-RoBERTa (no context, best validation run)"),
                               ("e5", "B-E5", "multilingual-e5-small (best validation run)"),
                               ("muril", "B2", "MuRIL (best validation run)"),
                               ("indicbert", "B-IndicBERT", "IndicBERTv2 (best validation run)"),
                               ("mdeberta", "B-mDeBERTa", "mDeBERTa-v3 (best validation run)")):
            eid = sel.get(fam)
            if eid and (exp / "final" / eid / f"metrics_{eid}.json").exists():
                rows.append({"id": mid, "name": f"{name}: {eid}", "input": "cleaned_text",
                             "metrics": exp / "final" / eid / f"metrics_{eid}.json", "tag": eid,
                             "dir": exp / "final" / eid, "pending_note": MODEL_B_PENDING,
                             "baseline_note": self._served_note(exp / "final" / eid)})
            elif fam == "xlmr" and (self.s.multi / "model_b" / "metrics_model_b.json").exists():   # Phase-2 import
                rows.append({"id": "B", "name": "XLM-RoBERTa (no context)", "input": "cleaned_text",
                             "metrics": self.s.multi / "model_b" / "metrics_model_b.json", "tag": "model_b",
                             "dir": self.s.multi / "model_b", "pending_note": MODEL_B_PENDING})
            else:
                rows.append({"id": mid, "name": name, "input": "cleaned_text", "metrics": exp / "_none.json",
                             "tag": f"pending_{fam}", "dir": exp / "final", "pending_note": MODEL_B_PENDING})
        for mid, tag, name in (("P", "production_incumbent", "Previous production ensemble (frozen A+ / E5 weights)"),
                               ("E", "ensemble_validation_tuned", "Ensemble (weights tuned on validation)"),
                               ("E-lang", "lang_ensemble_validation_tuned",
                                "Language-aware ensemble (per-language weights tuned on validation)")):
            d = exp / "final" / tag
            rows.append({"id": mid, "name": name, "input": "cleaned_text", "metrics": d / f"metrics_{tag}.json",
                         "tag": tag, "dir": d, "baseline_note": self._served_note(d),
                         "pending_note": "Not trained / not promoted (needs Transformer results and a validation gain)."})
        return rows

    @staticmethod
    def _served_note(d: Path) -> str | None:
        """Language-dependent models: the test score uses the dataset language labels; the app uses its heuristic."""
        tm = _json(d / "test_metrics.json") or {}
        srv = tm.get("as_served_detected_language")
        if not srv:
            return None
        return (f"Test score uses the dataset's language labels. With the app's heuristic language detector (as served): "
                f"accuracy {srv['accuracy']:.4f}, macro F1 {srv['macro_f1']:.4f}.")

    def comparison(self) -> list[dict]:
        out = []
        for run_id, run in self.runs().items():
            models = []
            for spec in run["models"]:
                m = _json(spec["metrics"])
                entry = {"id": spec["id"], "name": spec["name"], "input": spec["input"],
                         "status": "trained" if m else "pending",
                         "note": spec.get("baseline_note") if m else spec["pending_note"]}
                if m:
                    entry.update({
                        "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
                        **self._macro_pr(m),
                        "best_validation_macro_f1": m.get("best_validation_macro_f1"),
                        "per_class": m["per_class"], "confidence": m.get("confidence"),
                        "n_test": m.get("n") or m.get("n_test"),
                        "per_language": {k: {kk: v.get(kk) for kk in ("status", "n", "accuracy", "macro_f1", "weighted_f1")}
                                         for k, v in m["per_language"].items()},
                        "confusion_matrix": m.get("confusion_matrix") or self._cm_from_csv(spec),
                        "confusion_png": f"/api/files/{run_id}/confusion_matrix_{spec['tag']}.png",
                    })
                models.append(entry)
            models.append({"id": "C", "name": "XLM-RoBERTa (context-aware)", "input": "context_text",
                           "status": "skipped", "note": MODEL_C_SKIP, "explanation": MODEL_C_EXPLANATION})
            prod_tag = (_json(self.s.multi / "final_selection.json") or {}).get("selected_id")
            tags = {spec["id"]: spec["tag"] for spec in run["models"]}
            for m in models:
                m.update(self.presentation(run_id, m, is_production=run_id == "multilingual" and prod_tag is not None
                                           and tags.get(m["id"]) == prod_tag))
            out.append({"run": run_id, "title": run["title"], "models": models})
        return out

    @staticmethod
    def presentation(run_id: str, m: dict, is_production: bool) -> dict:
        name, desc = DISPLAY.get((run_id, m["id"]), (m["name"], ""))
        if m["status"] != "trained":
            cat = m["status"]                                  # pending / skipped: never shown in the polished UI
        elif is_production:
            cat = "production"
        else:
            strong = (m.get("accuracy") or 0) >= CATEGORY_MIN and (m.get("macro_f1") or 0) >= CATEGORY_MIN
            cat = "evaluated" if strong else "baseline"
        visible = m["status"] == "trained" and m.get("accuracy") is not None and m.get("macro_f1") is not None
        return {"display_name": name, "description": desc, "category": cat, "is_production": is_production,
                "dashboard_visible": visible}

    @staticmethod
    def _macro_pr(m: dict) -> dict:
        rep = m.get("classification_report") or {}
        mac, wei = rep.get("macro avg") or {}, rep.get("weighted avg") or {}
        return {"macro_precision": mac.get("precision"), "macro_recall": mac.get("recall"),
                "weighted_precision": wei.get("precision"), "weighted_recall": wei.get("recall")}

    @staticmethod
    def _cm_from_csv(spec) -> list[list[int]] | None:
        p = spec["dir"] / f"confusion_matrix_{spec['tag']}.csv"
        return pd.read_csv(p, index_col=0).to_numpy().tolist() if p.exists() else None

    def file_path(self, run: str, name: str) -> Path | None:
        allowed = {m["tag"]: m["dir"] for m in self.runs().get(run, {}).get("models", [])}
        for tag, d in allowed.items():
            if name == f"confusion_matrix_{tag}.png":
                p = d / name
                return p if p.exists() else None
        return None

    # ---- audit -----------------------------------------------------------------------
    def audit(self) -> dict:
        d = self.s.multi / "data"
        cleaning = _json(d / "cleaning_report.json")
        dist_path = d / "class_distributions.csv"
        return {
            "input_audit": _json(self.s.nlp_root / "audit_only" / "data_audit_report.json"),
            "provenance": _json(d / "provenance.json"),
            "label_mapping": _json(d / "label_mapping.json"),
            "cleaning": cleaning,
            "class_distributions": pd.read_csv(dist_path).to_dict(orient="records") if dist_path.exists() else [],
            "leakage_checks": _json(d / "leakage_checks.json"),
            "available": cleaning is not None,
        }

    # ---- errors ----------------------------------------------------------------------
    def _mistakes(self, run: str, tag: str) -> pd.DataFrame | None:
        spec = next((m for m in self.runs().get(run, {}).get("models", []) if m["tag"] == tag), None)
        if spec is None:
            return None
        p = spec["dir"] / f"predictions_{tag}.csv"
        if not p.exists():
            return None          # not cached: results imported later appear without a restart
        key = (run, tag, p.stat().st_mtime)
        if key not in self._cache:
            self._cache[key] = self._load_mistakes(p)
        return self._cache[key]

    @staticmethod
    def _load_mistakes(p: Path) -> pd.DataFrame:
        df = pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
        if "actual_label" not in df:  # Telugu-only prediction files store ids
            df["actual_label"] = df["actual_id"].map(dict(enumerate(CLASS_NAMES)))
            df["predicted_label"] = df["pred_id"].map(dict(enumerate(CLASS_NAMES)))
        wrong = df[df["actual_label"] != df["predicted_label"]].copy()
        wrong["likely_error_reason"] = [
            heuristic_error_reason(str(r.cleaned_text), str(r.language_type), LABEL2ID[r.actual_label],
                                   LABEL2ID[r.predicted_label], float(r.confidence),
                                   getattr(r, "annotations", None))
            for r in wrong.itertuples()]
        cols = ["actual_label", "predicted_label", "confidence", "raw_text", "cleaned_text", "language_type",
                "likely_error_reason"] + (["source_dataset"] if "source_dataset" in wrong else [])
        return wrong[cols].sort_values("confidence", ascending=False).reset_index(drop=True)

    def errors(self, run: str, tag: str, q: str | None, language: str | None, actual: str | None,
               predicted: str | None, reason: str | None, offset: int, limit: int) -> dict:
        df = self._mistakes(run, tag)
        if df is None:
            return {"available": False, "total": 0, "rows": [], "heuristic_note": HEURISTIC_NOTE}
        f = df
        if q:
            f = f[f["cleaned_text"].str.contains(q, case=False, regex=False, na=False)]
        if language:
            f = f[f["language_type"] == language]
        if actual:
            f = f[f["actual_label"] == actual]
        if predicted:
            f = f[f["predicted_label"] == predicted]
        if reason:
            f = f[f["likely_error_reason"].str.contains(reason, regex=False)]
        page = f.iloc[offset: offset + limit]
        return {"available": True, "total": int(len(f)), "total_mistakes": int(len(df)),
                "rows": json.loads(page.to_json(orient="records", force_ascii=False)),
                "heuristic_note": HEURISTIC_NOTE}

    def error_sources(self) -> list[dict]:
        cmp = {(r["run"], m["id"]): m for r in self.comparison() for m in r["models"]}
        out = []
        for run_id, run in self.runs().items():
            for m in run["models"]:
                if (m["dir"] / f"predictions_{m['tag']}.csv").exists():
                    c = cmp.get((run_id, m["id"]), {})
                    out.append({"run": run_id, "tag": m["tag"], "id": m["id"],
                                "label": f"{run['title']}: Model {m['id']} ({m['name']})",
                                "display_name": c.get("display_name", m["name"]), "category": c.get("category"),
                                "dashboard_visible": bool(c.get("dashboard_visible")),
                                "is_production": bool(c.get("is_production"))})
        return sorted(out, key=lambda r: (not r["is_production"], not r["dashboard_visible"]))
