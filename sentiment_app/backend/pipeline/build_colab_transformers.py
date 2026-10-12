"""Build the Phase-3 Colab package (Transformer study, Steps 8-12) in artifacts/colab_package_phase3/.

The package contains the EXACT local train/validation/test files (checksummed), sentiment_core, train_transformer.py
and colab_runner.py. The older artifacts/colab_package (single Model B run) is left untouched.
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402

BACKEND = Path(__file__).resolve().parents[1]
OUT = paths.APP_ROOT / "artifacts" / "colab_package_phase3"
PKG_NAME = "transformer_package.zip"
PACKAGE_VERSION = "phase3-v3 (ordered stages lr->epochs->loss->...; per-language P/R; IndicBERTv2, mDeBERTa)"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_zip(dest: Path) -> dict:
    members = {f"sentiment_core/{p.name}": p for p in sorted((BACKEND / "sentiment_core").glob("*.py"))}
    members["train_transformer.py"] = BACKEND / "pipeline" / "train_transformer.py"
    members["colab_runner.py"] = BACKEND / "pipeline" / "colab_runner.py"
    for s in ("train", "validation", "test"):
        members[f"data/{s}.csv"] = paths.DATA / f"{s}.csv"
    members["data/label_mapping.json"] = paths.DATA / "label_mapping.json"
    manifest = {name: sha(p) for name, p in members.items()}
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for name, p in members.items():
            z.write(p, name)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
        z.writestr("VERSION.txt", PACKAGE_VERSION)
    return manifest


CELLS = [
    ("md", """# Phase 3 — multilingual Transformer study (XLM-R Model B, losses, tuning, MuRIL / IndicBERTv2 / mDeBERTa, language tokens)

**Before starting:** Runtime → Change runtime type → **T4 GPU** (or better). Then **Runtime → Run all**.

- Data: the exact local split (train 70,540 / validation 15,121 / test 15,124), verified against `MANIFEST.json`. No new split is created.
- Selection: **validation macro F1** (ties: macro recall, then accuracy). Each stage changes one factor of the current best XLM-R recipe.
- `test.csv` is opened only in the *Final test* cell, and only for the best validation run of each model family.
- Resumable: runs are stored in `MyDrive/NLP_sentiment/phase3_runs/` and skipped on the next *Run all*. Reconnect and *Run all* after a disconnect.
- Per epoch: validation loss, accuracy, macro P/R/F1, weighted F1, per-class P/R/F1 and training loss (`validation_history.csv`).
- Model C: Skipped — no real conversation_id/message_order available.

| Tier | Stages (run in this order) | Runs | T4 time (approx.) |
|---|---|---|---|
| 1 (core) | lr {1e-5,2e-5,3e-5} → epochs 3 / 8 (10 only if still improving) → loss (WCE, focal 1.5/2.0 + 1.0 or 2.5) → language tokens → MuRIL | ~12 | ~6 h |
| 2 (refine) | lr 1.5e-5/2.5e-5 · label smoothing 0.05/0.1 · IndicBERTv2 | ~6 | ~3 h |
| 3 (optional) | batch 32/8 · weight decay 0/0.02, warmup 0.05 · mDeBERTa (fp32) · layer freezing · max_len 192 | ~9 | ~5 h |

Start with Tier 1; add tiers in later sessions by editing `STAGES` (finished runs are reused)."""),
    ("code", """import sys, json, hashlib, zipfile, shutil
from pathlib import Path
from datetime import datetime
import torch
if not torch.cuda.is_available():
    raise RuntimeError("GPU runtime required: Runtime -> Change runtime type -> T4 GPU, then Run all again.")
!nvidia-smi -L
from google.colab import drive, files
drive.mount("/content/drive")
DRIVE = Path("/content/drive/MyDrive/NLP_sentiment"); DRIVE.mkdir(parents=True, exist_ok=True)
PKG_ZIP = DRIVE / "transformer_package.zip"
print("Choose transformer_package.zip (D:\\NLP\\sentiment_app\\artifacts\\colab_package_phase3)" if not PKG_ZIP.exists()
      else f"Using {PKG_ZIP} (delete it from Drive to upload a newer package)")
if not PKG_ZIP.exists():
    up = files.upload()
    PKG_ZIP.write_bytes(up[next(iter(up))])
PKG = Path("/content/pkg"); shutil.rmtree(PKG, ignore_errors=True)
with zipfile.ZipFile(PKG_ZIP) as z: z.extractall(PKG)
manifest = json.loads((PKG / "MANIFEST.json").read_text())
bad = [n for n, h in manifest.items() if hashlib.sha256((PKG / n).read_bytes()).hexdigest() != h]
if bad: raise SystemExit(f"Package files do not match MANIFEST.json: {bad}")
print(f"All {len(manifest)} package files verified. Package: {(PKG / 'VERSION.txt').read_text() if (PKG / 'VERSION.txt').exists() else 'v1'}")"""),
    ("code", """!pip -q install datasets accelerate sentencepiece protobuf   # only installs what is missing"""),
    ("code", """# ---- settings ----
sys.path.insert(0, str(PKG))
import colab_runner
STAGES = colab_runner.TIER1                  # later: colab_runner.TIER1 + colab_runner.TIER2 (+ colab_runner.TIER3)
RUNS = DRIVE / "phase3_runs"
runner = colab_runner.Runner(PKG / "data", RUNS)
runner.plan(stages=STAGES)
runner.report("all planned stages")"""),
    ("md", """## Final test (validation-selected configurations only)
For each model family the run with the best validation macro F1 is scored once on `test.csv`."""),
    ("code", """selected = runner.final_test()
print("Selected on validation and scored on test:", selected)
for fam, eid in selected.items():
    t = json.loads((RUNS / eid / "test_metrics.json").read_text())
    print(f"  {fam}: {eid}  TEST macro F1 {t['macro_f1']:.4f}  acc {t['accuracy']:.4f}")"""),
    ("code", """RUN_ID = datetime.now().strftime("%Y%m%d_%H%M")
res_zip = DRIVE / f"phase3_results_{RUN_ID}.zip"
with zipfile.ZipFile(res_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(RUNS.rglob("*")):
        if p.is_file() and not any(part.startswith("_best_model_") for part in p.relative_to(RUNS).parts):
            z.write(p, p.relative_to(RUNS).as_posix())
print("Saved on Drive:", res_zip, f"({res_zip.stat().st_size / 1e6:.1f} MB)")
files.download(str(res_zip))     # metrics/predictions/plots only (small) -> your Downloads folder"""),
    ("code", """# Weights for the web app (~1.1 GB per model). Needed only if a Transformer or the ensemble is selected as the final model.
DOWNLOAD_MODELS = []             # e.g. ["xlmr"] or ["xlmr", "muril"]
for fam in DOWNLOAD_MODELS:
    best = RUNS / f"_best_model_{fam}" / "model"
    mz = DRIVE / f"phase3_model_{fam}_{RUN_ID}.zip"
    with zipfile.ZipFile(mz, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(best.rglob("*")):
            if p.is_file(): z.write(p, p.relative_to(best).as_posix())
    print("Saved on Drive:", mz)
    files.download(str(mz))"""),
]


def build_notebook(dest: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in CELLS]
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"}}
    nbf.write(nb, dest)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = build_zip(OUT / PKG_NAME)
    build_notebook(OUT / "colab_phase3_transformers.ipynb")
    (OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"package: {OUT / PKG_NAME} ({(OUT / PKG_NAME).stat().st_size / 1e6:.1f} MB, {len(manifest)} files)\n"
          f"notebook: {OUT / 'colab_phase3_transformers.ipynb'}")


if __name__ == "__main__":
    main()
