"""Colab orchestration for the Transformer study. Resumable: every finished run is stored on Google Drive and skipped
on the next "Run all", so the plan can be spread over several Colab sessions.

Staged (coordinate) search: every stage changes ONE factor of the current best XLM-R configuration, where "best" means
highest VALIDATION macro F1 (ties: validation macro recall, then validation accuracy). test.csv is never opened here
except by final_test(), which scores only the best validation run of each model family.

  Stages run in ORDER: lr -> (lr refine) -> epochs -> loss -> (smoothing) -> (batch) -> (wd/warmup) -> language
  tokens -> MuRIL -> (IndicBERTv2) -> (mDeBERTa) -> (freezing) -> (max_len 192).
  Tier 1 (core)      1 lr {1e-5, 2e-5, 3e-5}      5 epochs: 3, and 8 with patience 3 (10 only if 8 was still improving)
                     3 loss: weighted CE, focal 1.5/2.0 (+1.0 or 2.5 toward the better side)
                     8 language tokens             9 MuRIL (best recipe, lr 2e-5 / 3e-5)
  Tier 2 (refine)    2 lr refinement (1.5e-5 / 2.5e-5 next to the best lr)
                     4 label smoothing 0.05 / 0.1 (CE/WCE recipes only)     10 IndicBERTv2 (lr 2e-5 / 3e-5)
  Tier 3 (optional)  6 effective batch 32 and 8    7 weight decay 0.0 / 0.02, warmup 0.05
                     11 mDeBERTa-v3 (fp32)   12 freeze lower 6 layers in epoch 1   13 max_length 192
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import train_transformer as tt

MODELS = {"xlmr": "xlm-roberta-base", "muril": "google/muril-base-cased",
          "indicbert": "ai4bharat/IndicBERTv2-MLM-only", "mdeberta": "microsoft/mdeberta-v3-base"}
FAMILY_OF = {v: k for k, v in MODELS.items()}
# Execution order (stage numbers are ids): lr -> lr refine -> epochs -> loss -> label smoothing -> batch ->
# weight decay / warmup -> language tokens -> MuRIL -> IndicBERTv2 -> mDeBERTa -> layer freezing -> max_len 192
ORDER = (1, 2, 5, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13)
TIER1, TIER2, TIER3 = (1, 5, 3, 8, 9), (2, 4, 10), (6, 7, 11, 12, 13)
LR_LADDER = [1e-5, 1.5e-5, 2e-5, 2.5e-5, 3e-5]
DEFAULT = {"family": "xlmr", "lr": 2e-5, "epochs": 5, "max_length": 128, "loss": "ce", "gamma": 2.0,
           "label_smoothing": 0.0, "weight_decay": 0.01, "warmup_ratio": 0.1, "batch_size": 16, "grad_accum": 1,
           "patience": 2, "lang_tokens": False, "freeze_lower_layers": 0, "no_fp16": False}


def exp_id(c: dict) -> str:
    loss = c["loss"] + (format(c["gamma"], "g") if c["loss"] == "focal" else "")
    parts = [c["family"], f"lr{c['lr']:g}", loss, f"ep{c['epochs']}p{c['patience']}",
             f"bs{c['batch_size']}x{c['grad_accum']}", f"wd{c['weight_decay']:g}", f"wu{c['warmup_ratio']:g}",
             f"len{c['max_length']}"]
    if c["label_smoothing"]:
        parts.append(f"ls{c['label_smoothing']:g}")
    if c["lang_tokens"]:
        parts.append("langtok")
    if c["freeze_lower_layers"]:
        parts.append(f"freeze{c['freeze_lower_layers']}")
    if c["no_fp16"]:
        parts.append("fp32")
    return "_".join(parts)


def cfg_from_saved(saved: dict) -> dict:
    """Rebuild the runner config from a run's config.json."""
    return {"family": FAMILY_OF[saved["model"]], "lr": saved["learning_rate"], "epochs": saved["epochs_max"],
            "max_length": saved["max_length"], "loss": saved["loss"], "gamma": saved.get("focal_gamma") or 2.0,
            "label_smoothing": saved.get("label_smoothing", 0.0), "weight_decay": saved.get("weight_decay", 0.01),
            "warmup_ratio": saved.get("warmup_ratio", 0.1), "batch_size": saved["batch_size"],
            "grad_accum": saved["grad_accum"], "patience": saved.get("early_stopping_patience", 2),
            "lang_tokens": saved["lang_tokens"], "freeze_lower_layers": saved.get("freeze_lower_layers_epoch1", 0),
            "no_fp16": bool(saved.get("no_fp16_flag", False))}


def rank_key(m: dict):
    return (m["macro_f1"], m["macro_recall"], m["accuracy"])


class Runner:
    def __init__(self, data_dir: Path, drive_runs: Path, local_runs: Path = Path("/content/runs"),
                 smoke_rows: int = 0):
        self.data, self.drive, self.local = Path(data_dir), Path(drive_runs), Path(local_runs)
        self.smoke = smoke_rows                   # >0 only for code checks; such runs are never results
        self.drive.mkdir(parents=True, exist_ok=True)
        self.local.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ bookkeeping
    def val(self, eid: str) -> dict | None:
        f = self.drive / eid / "validation_metrics.json"
        return json.loads(f.read_text()) if f.exists() else None

    def saved_cfg(self, eid: str) -> dict:
        return json.loads((self.drive / eid / "config.json").read_text())

    def finished(self, family: str | None = None) -> list[tuple[str, dict]]:
        out = []
        for d in sorted(self.drive.iterdir()):
            if d.is_dir() and (d / "validation_metrics.json").exists() and (d / "config.json").exists():
                c = self.saved_cfg(d.name)
                if bool(c.get("smoke_test")) != bool(self.smoke):
                    continue
                if family is None or FAMILY_OF.get(c["model"]) == family:
                    out.append((d.name, self.val(d.name), (d / "validation_metrics.json").stat().st_mtime))
        # exact ties keep the incumbent (the run that finished first)
        out.sort(key=lambda x: x[2])
        out.sort(key=lambda x: rank_key(x[1]), reverse=True)
        return [(eid, m) for eid, m, _ in out]

    def best_id(self, family: str = "xlmr") -> str | None:
        runs = self.finished(family)
        return runs[0][0] if runs else None

    def best_cfg(self, family: str = "xlmr") -> dict:
        eid = self.best_id(family)
        return cfg_from_saved(self.saved_cfg(eid)) if eid else dict(DEFAULT, family=family)

    def family_best_dir(self, family: str) -> Path:
        return self.drive / f"_best_model_{family}"

    # ------------------------------------------------------------ one run
    def run(self, cfg: dict) -> dict | None:
        c = dict(DEFAULT, **cfg)
        eid = exp_id(c)
        if self.val(eid):
            m = self.val(eid)
            print(f"[{eid}] already on Drive: val macro F1 {m['macro_f1']:.4f} (skipped)")
            return m
        argv = ["--data-dir", str(self.data), "--out-dir", str(self.local), "--exp-id", eid,
                "--model", MODELS[c["family"]], "--lr", str(c["lr"]), "--epochs", str(c["epochs"]),
                "--max-length", str(c["max_length"]), "--loss", c["loss"], "--gamma", str(c["gamma"]),
                "--label-smoothing", str(c["label_smoothing"]), "--weight-decay", str(c["weight_decay"]),
                "--warmup-ratio", str(c["warmup_ratio"]), "--patience", str(c["patience"]),
                "--batch-size", str(c["batch_size"]), "--grad-accum", str(c["grad_accum"]),
                "--freeze-lower-layers", str(c["freeze_lower_layers"]),
                "--checkpoint-dir", str(self.local / "_ckpt") if self.smoke else "/content/checkpoints", "--keep-model"]
        if c["lang_tokens"]:
            argv.append("--lang-tokens")
        if c["no_fp16"]:
            argv.append("--no-fp16")
        if self.smoke:
            argv += ["--smoke-rows", str(self.smoke)]
        shutil.rmtree(self.local / eid, ignore_errors=True)
        try:
            m = tt.main(argv)
        except RuntimeError as e:                  # CUDA OOM -> same effective batch with accumulation
            if "out of memory" not in str(e).lower() or c["batch_size"] <= 8:
                raise
            import torch
            torch.cuda.empty_cache()
            print(f"[{eid}] CUDA out of memory; retrying with half the batch and 2x accumulation")
            return self.run({**cfg, "batch_size": c["batch_size"] // 2, "grad_accum": c["grad_accum"] * 2})
        src, dst = self.local / eid, self.drive / eid
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("model", "_checkpoints"))
        best_dir = self.family_best_dir(c["family"])
        meta = best_dir / "meta.json"
        prev = json.loads(meta.read_text()) if meta.exists() else None
        if prev is None or rank_key(m) > tuple(prev["rank_key"]):
            t0 = time.time()
            shutil.rmtree(best_dir, ignore_errors=True)
            shutil.copytree(src / "model", best_dir / "model")
            shutil.copy(src / "config.json", best_dir / "model" / "phase3_config.json")
            meta.write_text(json.dumps({"experiment_id": eid, "val_macro_f1": m["macro_f1"], "rank_key": rank_key(m)},
                                       indent=2))
            print(f"[{eid}] new best {c['family']} model saved to Drive ({time.time() - t0:.0f}s)")
        shutil.rmtree(src / "model", ignore_errors=True)
        return m

    def promoted(self, before: str | None, stage: str) -> None:
        after = self.best_id("xlmr")
        msg = "kept previous best" if after == before else f"NEW best -> {after}"
        print(f"[stage {stage}] {msg}")

    # ------------------------------------------------------------ stages
    def plan(self, stages=TIER1) -> None:
        stages = set(stages)
        for st in [x for x in ORDER if x in stages]:
            before = self.best_id("xlmr")
            b = self.best_cfg("xlmr")
            if st == 1:
                for lr in (1e-5, 2e-5, 3e-5):
                    self.run(dict(DEFAULT, lr=lr))
            elif st == 2:
                i = LR_LADDER.index(b["lr"]) if b["lr"] in LR_LADDER else None
                for j in ((i - 1, i + 1) if i is not None else ()):
                    if 0 <= j < len(LR_LADDER) and LR_LADDER[j] not in (1e-5, 2e-5, 3e-5):
                        self.run(dict(b, lr=LR_LADDER[j]))
            elif st == 3:
                base = dict(b, loss="ce", label_smoothing=0.0)
                self.run(dict(base, loss="wce"))
                f15 = self.run(dict(base, loss="focal", gamma=1.5))
                f20 = self.run(dict(base, loss="focal", gamma=2.0))
                self.run(dict(base, loss="focal", gamma=1.0 if rank_key(f15) >= rank_key(f20) else 2.5))
            elif st == 4:
                if b["loss"] in ("ce", "wce"):
                    for ls in (0.05, 0.1):
                        self.run(dict(b, label_smoothing=ls))
                else:
                    print("[stage 4] best recipe uses focal loss; label smoothing is defined for CE/WCE only - skipped")
            elif st == 5:
                self.run(dict(b, epochs=3, patience=2))
                m8 = self.run(dict(b, epochs=8, patience=3))
                c8 = self.saved_cfg(exp_id(dict(DEFAULT, **dict(b, epochs=8, patience=3))))
                if m8 and c8.get("still_improving_at_last_epoch"):
                    self.run(dict(b, epochs=10, patience=3))
                else:
                    print("[stage 5] epoch 8 was not still improving at the end -> 10 epochs not run")
            elif st == 6:
                self.run(dict(b, batch_size=32, grad_accum=1))
                self.run(dict(b, batch_size=8, grad_accum=1))
            elif st == 7:
                for wd in (0.0, 0.02):
                    self.run(dict(b, weight_decay=wd))
                self.run(dict(self.best_cfg("xlmr"), warmup_ratio=0.05))
            elif st == 8:
                self.run(dict(b, lang_tokens=True))
            elif st in (9, 10):
                fam = "muril" if st == 9 else "indicbert"
                for lr in (2e-5, 3e-5):
                    self.run(dict(b, family=fam, lr=lr))
            elif st == 11:
                self.run(dict(b, family="mdeberta", lr=2e-5, no_fp16=True))
            elif st == 12:
                self.run(dict(b, freeze_lower_layers=6))
            elif st == 13:
                self.run(dict(b, max_length=192))
            if st not in (9, 10, 11):
                self.promoted(before, str(st))
            self.report(f"after stage {st}")

    def report(self, title: str) -> None:
        print(f"\n=== VALIDATION leaderboard {title} (rank: macro F1, then macro recall, then accuracy) ===")
        for eid, m in self.finished():
            pc = m["per_class"]
            print(f"  {eid:<62} F1 {m['macro_f1']:.4f} R {m['macro_recall']:.4f} acc {m['accuracy']:.4f} "
                  f"P {m['macro_precision']:.4f} | F1 neg/neu/pos {pc['Negative']['f1']:.3f}/{pc['Neutral']['f1']:.3f}/"
                  f"{pc['Positive']['f1']:.3f} | Neutral P/R {pc['Neutral']['precision']:.3f}/{pc['Neutral']['recall']:.3f}"
                  + (f" | English F1 {m['per_language']['english']['macro_f1']:.3f}" if m.get("per_language", {}).get("english") else ""))
        top = self.finished()
        if top:
            print(f"DONE: {title}\nCURRENT BEST VALIDATION MACRO F1: {top[0][1]['macro_f1']:.4f}\n"
                  f"CURRENT BEST MODEL: {top[0][0]}")

    # ------------------------------------------------------------ final test (selected configs only)
    def final_test(self) -> dict:
        out = {}
        for family in MODELS:
            meta = self.family_best_dir(family) / "meta.json"
            if not meta.exists():
                continue
            sel = json.loads(meta.read_text())["experiment_id"]
            top = self.best_id(family)
            assert rank_key(self.val(sel)) >= rank_key(self.val(top)), f"saved model {sel} is not the best run {top}"
            if not (self.drive / sel / "test_metrics.json").exists():
                argv = ["--data-dir", str(self.data), "--out-dir", str(self.drive), "--exp-id", sel,
                        "--predict-test", str(self.family_best_dir(family) / "model")]
                tt.main(argv + (["--smoke-rows", str(self.smoke)] if self.smoke else []))
            out[family] = sel
        (self.drive / "_selected_for_test.json").write_text(json.dumps(out, indent=2))
        return out
