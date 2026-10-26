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
PACKAGE_VERSION = ("phase3-v6 (GPU search for XLM-R, MuRIL, IndicBERTv2, mDeBERTa; E5 winner already trained locally; "
                   "focal 1.0-2.5, class-weight schemes, language sampler, 4 text variants, freezing, cosine; "
                   "as-served language tokens)")


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
    ("md", """# Phase 3 — full multilingual Transformer study (E5, XLM-R, MuRIL, IndicBERTv2, mDeBERTa)

**Before starting:** Runtime → Change runtime type → **GPU** (L4 or A100 preferred, T4 works). Then **Runtime → Run all**.

- Data: the exact local split (train 70,540 / validation 15,121 / test 15,124), verified against `MANIFEST.json` (SHA-256). No new split, no removed samples, no test rebalancing.
- Selection: **validation macro F1** (ties: macro recall, then accuracy). Every stage changes ONE factor of the family's current best run; the best epoch of each run is kept (early stopping on validation macro F1).
- `test.csv` is opened only in the *Final test* cell, and only for the best validation run of each model family.
- Resumable: runs are stored on Drive when available (otherwise `/content/NLP_sentiment/phase3_runs/`) and skipped on the next *Run all*.
- Per epoch: validation loss, accuracy, macro P/R/F1, weighted F1, per-class P/R/F1, training loss (per-language metrics: best epoch only, in `validation_metrics.json`) (`validation_history.csv`).
- Model C: Skipped — no real conversation_id/message_order available.

| Step of `PLAN` | Stages (one factor at a time) | Runs (approx.) | T4 / L4 / A100 (rough estimate, not measured) |
|---|---|---|---|
| XLM-R core | lr 1e-5/2e-5/3e-5 @3 ep → epochs 4→5→6→8 while improving → loss (CE, WCE, focal 1.0/1.5/2.0/2.5) → language tokens → sampler (sqrt, balanced) | ~14 | 7 h / 4 h / 2 h |
| MuRIL core, IndicBERTv2 core | best XLM-R recipe; lr 2e-5/3e-5 → epochs | ~5 each | 3 h each / 1.7 h / 0.9 h |
| XLM-R refine | lr 1.5e-5/2.5e-5 → class weights (sqrt-inverse, effective-number) → label smoothing → text variants (repeat norm, letters-only repeat norm, + emoticon tokens, emoticons only) | ~9 | 5 h / 3 h / 1.5 h |
| MuRIL / IndicBERTv2 refine | loss → language tokens → sampler | ~6 each | 3 h each |
| mDeBERTa core (fp32) | lr 2e-5/3e-5 → epochs | ~5 | 8 h / 4 h / 2 h |
| XLM-R extra | wd/warmup/cosine → batch 32 → freezing → max_len 192 | ~9 | 5 h / 3 h / 1.5 h |

E5 is not in `PLAN`: its family winner `e5_lr5e-05_ce_ep3p2_bs32_len128_cpu` was trained on the local CPU and is already imported (use `colab_runner.PLAN_WITH_E5` for a wider E5 search).

Cut `PLAN` short for a session and resume later: finished runs are reused, so nothing is retrained."""),
    ("code", """import sys, json, hashlib, zipfile, shutil
from pathlib import Path
from datetime import datetime
import torch
print("torch.cuda.is_available():", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("GPU runtime required: select an L4 / A100 / T4 runtime, then run all again.")
props = torch.cuda.get_device_properties(0)
print(f"GPU: {props.name} | VRAM {props.total_memory / 2**30:.1f} GiB | CUDA (torch build) {torch.version.cuda} | torch {torch.__version__}")
!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
from google.colab import files
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    DRIVE = Path("/content/drive/MyDrive/NLP_sentiment")
    DRIVE.mkdir(parents=True, exist_ok=True)
    print("Google Drive enabled:", DRIVE)
except Exception as exc:
    DRIVE = Path("/content/NLP_sentiment")
    DRIVE.mkdir(parents=True, exist_ok=True)
    print("Google Drive unavailable; using local Colab storage:", exc)
PKG_ZIP = DRIVE / "transformer_package.zip"
if not PKG_ZIP.exists():
    print("Upload transformer_package.zip from D:/NLP/sentiment_app/artifacts/colab_package_phase3")
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
PLAN = colab_runner.PLAN          # full order (see the table above); e.g. PLAN[:2] = XLM-R core + MuRIL core only
RUNS = DRIVE / "phase3_runs"
runner = colab_runner.Runner(PKG / "data", RUNS)
runner.plan(PLAN)
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
    ("code", """# Weights for the web app and ensembles (E5 ~0.5 GB, the base models ~1.1 GB each). Chosen on VALIDATION only:
# every family whose best validation macro F1 beats A+ (0.6962). Set a list to override, [] to skip.
A_PLUS_VAL_MACRO_F1 = 0.6962040371748555
DOWNLOAD_MODELS = None           # None = automatic; e.g. ["xlmr"] or [] to skip
if DOWNLOAD_MODELS is None:
    val = {f: json.loads((RUNS / e / "validation_metrics.json").read_text())["macro_f1"] for f, e in selected.items()}
    DOWNLOAD_MODELS = [f for f in sorted(val, key=val.get, reverse=True) if val[f] > A_PLUS_VAL_MACRO_F1]
    print("Validation macro F1 per family:", {f: round(v, 4) for f, v in val.items()}, "-> downloading", DOWNLOAD_MODELS)
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
