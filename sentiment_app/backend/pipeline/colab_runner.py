"""Colab orchestration for the Transformer study. Resumable: every finished run is stored on Google Drive and skipped
on the next "Run all", so the plan can be spread over several Colab sessions.

Staged (coordinate) search per model family: every stage varies ONE factor of the family's current best configuration,
where "best" means highest VALIDATION macro F1 (ties: validation macro recall, then validation accuracy). The next stage
starts from the new best. test.csv is never opened here except by final_test(), which scores only the best validation
run of each model family.

Stages (each family runs the stages listed for it in FAMILY_STAGES, tier by tier):
  lr            E5 {2e-5, 3e-5, 5e-5} (+1e-5 if 2e-5 wins); XLM-R {1e-5, 2e-5, 3e-5}; MuRIL/IndicBERT/mDeBERTa {2e-5, 3e-5}
                - all at 3 epochs
  lr_refine     ladder neighbours of the best lr that were not run yet (XLM-R 1.5e-5 / 2.5e-5)
  epochs        4 -> 5 -> 6 -> 8, continuing only while the previous run was still improving at its last epoch or set a
                new family best (patience 2 up to 5 epochs, 3 from 6 epochs)
  max_length    E5: 64 / 96 / 192; XLM-R: 192 (0.18% of training texts exceed 128 XLM-R tokens, 0.03% exceed 192)
  loss          weighted CE (inverse frequency), focal 1.5 / 2.0, then focal 1.0 or 2.5 toward the better side
  class_weights weighted CE with sqrt-inverse and effective-number (beta 0.9999) weights
  smoothing     label smoothing 0.03 / 0.05 / 0.10 (CE/WCE recipes only)
  lang_tokens   [LANG_ENGLISH]/[LANG_TELUGU]/[LANG_CODE_MIXED] special tokens (embeddings resized)
  sampler       language-reweighted TRAIN sampling: sqrt (temperature) and fully balanced
  text_variant  repeat normalisation; repeat normalisation + emoticon tokens
  optimizer     weight decay 0 / 0.02, warmup 0.03 / 0.05, cosine schedule
  freezing      bottom 3 / 6 layers frozen permanently; bottom 6 frozen in epoch 1 only (gradual unfreezing)
  batch         batch 32 (OOM -> 16 x 2 accumulation)
MuRIL / IndicBERTv2 / mDeBERTa start from the best XLM-R recipe (loss, smoothing, tokens, sampler, variant, optimizer),
with their own lr and epoch search.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import train_transformer as tt

MODELS = {"e5": "intfloat/multilingual-e5-small", "xlmr": "xlm-roberta-base", "muril": "google/muril-base-cased",
          "indicbert": "ai4bharat/IndicBERTv2-MLM-only", "mdeberta": "microsoft/mdeberta-v3-base"}
FAMILY_OF = {v: k for k, v in MODELS.items()}
E5 = "e5"
DEFAULT = {"family": "xlmr", "lr": 2e-5, "epochs": 3, "max_length": 128, "loss": "ce", "gamma": 2.0,
           "class_weight_scheme": "inverse", "label_smoothing": 0.0, "weight_decay": 0.01, "warmup_ratio": 0.1,
           "scheduler": "linear", "batch_size": 16, "grad_accum": 1, "patience": 2, "lang_tokens": False,
           "sampler": "random", "text_variant": "v1_current", "freeze_lower_layers": 0, "freeze_mode": "epoch1",
           "no_fp16": False}
FIRST_LRS = {"e5": (2e-5, 3e-5, 5e-5), "xlmr": (1e-5, 2e-5, 3e-5), "muril": (2e-5, 3e-5),
             "indicbert": (2e-5, 3e-5), "mdeberta": (2e-5, 3e-5)}
LR_LADDER = {"e5": [1e-5, 2e-5, 3e-5, 5e-5], "xlmr": [1e-5, 1.5e-5, 2e-5, 2.5e-5, 3e-5]}
EPOCH_LADDER = (4, 5, 6, 8)
# factors a borrowing family takes over from the best XLM-R recipe (never lr / epochs / max_length / batch)
RECIPE_KEYS = ("loss", "gamma", "class_weight_scheme", "label_smoothing", "lang_tokens", "sampler", "text_variant",
               "weight_decay", "warmup_ratio", "scheduler")
FAMILY_STAGES = {
    "e5": {"core": ("lr", "epochs", "max_length", "loss", "class_weights", "smoothing", "lang_tokens", "sampler"),
           "extra": ("text_variant", "optimizer", "freezing", "batch")},
    "xlmr": {"core": ("lr", "epochs", "loss", "lang_tokens", "sampler"),
             "refine": ("lr_refine", "class_weights", "smoothing", "text_variant"),
             "extra": ("optimizer", "batch", "freezing", "max_length")},
    "muril": {"core": ("lr", "epochs"), "refine": ("loss", "lang_tokens", "sampler")},
    "indicbert": {"core": ("lr", "epochs"), "refine": ("loss", "lang_tokens", "sampler")},
    "mdeberta": {"core": ("lr", "epochs")},
}
# default session plan, in order; finished runs are reused, so it can be cut short and resumed
PLAN = [("e5", "core"), ("xlmr", "core"), ("muril", "core"), ("indicbert", "core"),
        ("e5", "extra"), ("xlmr", "refine"), ("muril", "refine"), ("indicbert", "refine"),
        ("mdeberta", "core"), ("xlmr", "extra")]


def exp_id(c: dict) -> str:
    loss = c["loss"] + (format(c["gamma"], "g") if c["loss"] == "focal" else "")
    if c["loss"] == "wce" and c["class_weight_scheme"] != "inverse":
        loss += {"sqrt_inverse": "sqrt", "effective": "eff"}[c["class_weight_scheme"]]
    parts = [c["family"], f"lr{c['lr']:g}", loss, f"ep{c['epochs']}p{c['patience']}",
             f"bs{c['batch_size']}x{c['grad_accum']}", f"wd{c['weight_decay']:g}", f"wu{c['warmup_ratio']:g}",
             f"len{c['max_length']}"]
    if c["scheduler"] != "linear":
        parts.append(c["scheduler"])
    if c["label_smoothing"]:
        parts.append(f"ls{c['label_smoothing']:g}")
    if c["lang_tokens"]:
        parts.append("langtok")
    if c["sampler"] != "random":
        parts.append(c["sampler"].replace("_", ""))
    if c["text_variant"] != "v1_current":
        parts.append(c["text_variant"].split("_")[0])
    if c["freeze_lower_layers"]:
        parts.append(f"freeze{c['freeze_lower_layers']}" + ("perm" if c["freeze_mode"] == "permanent" else ""))
    if c["no_fp16"]:
        parts.append("fp32")
    return "_".join(parts)


def cfg_from_saved(saved: dict) -> dict:
    """Rebuild the runner config from a run's config.json."""
    return {"family": FAMILY_OF[saved["model"]], "lr": saved["learning_rate"], "epochs": saved["epochs_max"],
            "max_length": saved["max_length"], "loss": saved["loss"], "gamma": saved.get("focal_gamma") or 2.0,
            "class_weight_scheme": saved.get("class_weight_scheme") or "inverse",
            "label_smoothing": saved.get("label_smoothing", 0.0), "weight_decay": saved.get("weight_decay", 0.01),
            "warmup_ratio": saved.get("warmup_ratio", 0.1), "scheduler": saved.get("scheduler", "linear"),
            "batch_size": saved["batch_size"], "grad_accum": saved["grad_accum"],
            "patience": saved.get("early_stopping_patience", 2), "lang_tokens": saved["lang_tokens"],
            "sampler": saved.get("sampler", "random"), "text_variant": saved.get("text_variant", "v1_current"),
            "freeze_lower_layers": saved.get("freeze_lower_layers_epoch1", 0),
            "freeze_mode": saved.get("freeze_mode") or "epoch1", "no_fp16": bool(saved.get("no_fp16_flag", False))}


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

    def best_id(self, family: str) -> str | None:
        runs = self.finished(family)
        return runs[0][0] if runs else None

    def start_cfg(self, family: str) -> dict:
        """Starting point of a family that has no runs yet: defaults, plus the best XLM-R recipe for the BERT-style
        alternatives (MuRIL / IndicBERTv2 / mDeBERTa)."""
        c = dict(DEFAULT, family=family, no_fp16=family == "mdeberta")
        src = self.best_id("xlmr") if family not in ("xlmr", E5) else None
        if src:
            c.update({k: cfg_from_saved(self.saved_cfg(src))[k] for k in RECIPE_KEYS})
        return c

    def best_cfg(self, family: str) -> dict:
        eid = self.best_id(family)
        return cfg_from_saved(self.saved_cfg(eid)) if eid else self.start_cfg(family)

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
                "--class-weight-scheme", c["class_weight_scheme"],
                "--label-smoothing", str(c["label_smoothing"]), "--weight-decay", str(c["weight_decay"]),
                "--warmup-ratio", str(c["warmup_ratio"]), "--scheduler", c["scheduler"],
                "--patience", str(c["patience"]), "--batch-size", str(c["batch_size"]),
                "--grad-accum", str(c["grad_accum"]), "--sampler", c["sampler"], "--text-variant", c["text_variant"],
                "--freeze-lower-layers", str(c["freeze_lower_layers"]), "--freeze-mode", c["freeze_mode"],
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

    # ------------------------------------------------------------ stages
    def stage(self, family: str, name: str) -> None:
        b = self.best_cfg(family)
        if name == "lr":
            lrs = FIRST_LRS[family]
            for lr in lrs:
                self.run(dict(b, lr=lr, epochs=3, patience=2))
            if family == E5 and self.best_cfg(family)["lr"] == 2e-5:
                self.run(dict(self.best_cfg(family), lr=1e-5))
        elif name == "lr_refine":
            ladder = LR_LADDER.get(family, [])
            i = ladder.index(b["lr"]) if b["lr"] in ladder else None
            for j in ((i - 1, i + 1) if i is not None else ()):
                if 0 <= j < len(ladder):
                    self.run(dict(b, lr=ladder[j]))
        elif name == "epochs":
            for ep in EPOCH_LADDER:
                if ep <= b["epochs"]:
                    continue
                before = self.best_id(family)
                cfg = dict(b, epochs=ep, patience=2 if ep <= 5 else 3)
                self.run(cfg)
                saved = self.drive / exp_id(dict(DEFAULT, **cfg)) / "config.json"   # absent after an OOM retry
                still = saved.exists() and json.loads(saved.read_text()).get("still_improving_at_last_epoch")
                if not still and self.best_id(family) == before:
                    print(f"[{family} epochs] {ep} epochs: peaked before the last epoch and no new best -> stop")
                    break
        elif name == "max_length":
            for L in ((64, 96, 192) if family == E5 else (192,)):
                if L != b["max_length"]:
                    self.run(dict(b, max_length=L))
        elif name == "loss":
            base = dict(b, loss="ce", label_smoothing=0.0, class_weight_scheme="inverse")
            for c in (base, dict(base, loss="wce")):
                self.run(c)
            f15 = self.run(dict(base, loss="focal", gamma=1.5))
            f20 = self.run(dict(base, loss="focal", gamma=2.0))
            self.run(dict(base, loss="focal", gamma=1.0 if rank_key(f15) >= rank_key(f20) else 2.5))
        elif name == "class_weights":
            for sch in ("sqrt_inverse", "effective"):
                self.run(dict(b, loss="wce", gamma=2.0, class_weight_scheme=sch))
        elif name == "smoothing":
            if b["loss"] in ("ce", "wce"):
                for ls in (0.03, 0.05, 0.1):
                    self.run(dict(b, label_smoothing=ls))
            else:
                print(f"[{family} smoothing] best recipe uses focal loss; label smoothing is defined for CE/WCE only")
        elif name == "lang_tokens":
            self.run(dict(b, lang_tokens=not b["lang_tokens"]))
        elif name == "sampler":
            for s in ("lang_sqrt", "lang_balanced"):
                if s != b["sampler"]:
                    self.run(dict(b, sampler=s))
        elif name == "text_variant":
            for v in ("v3_repeat_norm", "v4_repeat_norm_emoticons"):
                if v != b["text_variant"]:
                    self.run(dict(b, text_variant=v))
        elif name == "optimizer":
            for wd in (0.0, 0.01, 0.02):
                if wd != b["weight_decay"]:
                    self.run(dict(b, weight_decay=wd))
            b2 = self.best_cfg(family)
            for wu in (0.03, 0.05, 0.1):
                if wu != b2["warmup_ratio"]:
                    self.run(dict(b2, warmup_ratio=wu))
            self.run(dict(self.best_cfg(family), scheduler="cosine"))
        elif name == "freezing":
            self.run(dict(b, freeze_lower_layers=3, freeze_mode="permanent"))
            self.run(dict(b, freeze_lower_layers=6, freeze_mode="permanent"))
            self.run(dict(b, freeze_lower_layers=6, freeze_mode="epoch1"))
        elif name == "batch":
            if b["batch_size"] * b["grad_accum"] != 32:
                self.run(dict(b, batch_size=32, grad_accum=1))
        else:
            raise ValueError(name)
        after = self.best_id(family)
        print(f"[{family} stage {name}] best -> {after}")

    def plan(self, plan=PLAN) -> None:
        for family, tier in plan:
            for name in FAMILY_STAGES[family].get(tier, ()):
                self.stage(family, name)
            self.report(f"after {family} {tier}")

    def report(self, title: str) -> None:
        print(f"\n=== VALIDATION leaderboard {title} (rank: macro F1, then macro recall, then accuracy) ===")
        for eid, m in self.finished():
            pc, pl = m["per_class"], m.get("per_language", {})
            langs = " ".join(f"{k[:2]} {pl[k]['macro_f1']:.3f}" for k in ("english", "telugu", "code_mixed") if k in pl)
            print(f"  {eid:<70} F1 {m['macro_f1']:.4f} acc {m['accuracy']:.4f} R {m['macro_recall']:.4f} | "
                  f"neg/neu/pos {pc['Negative']['f1']:.3f}/{pc['Neutral']['f1']:.3f}/{pc['Positive']['f1']:.3f} | "
                  f"Neutral P/R {pc['Neutral']['precision']:.3f}/{pc['Neutral']['recall']:.3f} | {langs}")
        for fam in MODELS:
            eid = self.best_id(fam)
            if eid:
                print(f"BEST {fam}: {eid}  val macro F1 {self.val(eid)['macro_f1']:.4f}")

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
