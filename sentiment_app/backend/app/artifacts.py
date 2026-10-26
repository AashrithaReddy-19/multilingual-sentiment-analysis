"""Read-only access to saved, executed results and audit files. Nothing here computes new metrics."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sentiment_core.evaluation import HEURISTIC_NOTE, heuristic_error_reason
from sentiment_core.labels import CLASS_NAMES, LABEL2ID, MODEL_B_PENDING, MODEL_C_SKIP

from .config import Settings

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
        for mid, tag, name in (("P", "production_incumbent", "Current production ensemble (frozen A+ / E5 weights)"),
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
            out.append({"run": run_id, "title": run["title"], "models": models})
        return out

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
        out = []
        for run_id, run in self.runs().items():
            for m in run["models"]:
                if (m["dir"] / f"predictions_{m['tag']}.csv").exists():
                    out.append({"run": run_id, "tag": m["tag"], "label": f"{run['title']}: Model {m['id']} ({m['name']})"})
        return out
