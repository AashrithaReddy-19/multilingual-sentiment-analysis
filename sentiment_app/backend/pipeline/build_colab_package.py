"""Build the self-contained Colab package for Model B (GPU): code + prepared split + checksums + notebook."""
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


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_zip(dest: Path) -> dict:
    members = {f"sentiment_core/{p.name}": p for p in sorted((BACKEND / "sentiment_core").glob("*.py"))}
    members["train_model_b.py"] = BACKEND / "pipeline" / "train_model_b.py"
    for s in ("train", "validation", "test"):
        members[f"data/{s}.csv"] = paths.DATA / f"{s}.csv"
    members["data/label_mapping.json"] = paths.DATA / "label_mapping.json"
    manifest = {name: sha(p) for name, p in members.items()}
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
        for name, p in members.items():
            z.write(p, name)
        z.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
    return manifest


NOTEBOOK_CELLS = [
    ("md", """# Model B (multilingual XLM-RoBERTa, no context): Colab GPU run

Runs automatically after two prompts: **Google Drive permission** and **choosing `model_b_package.zip`** (only on the first run; it is then kept on Drive).

- **Data:** the prepared and audited TeSent + TweetEval + CMTET split (70/15/15, no text leakage). Every file is checked against `MANIFEST.json`.
- **Settings:** xlm-roberta-base, max_length 128, 5 epochs, lr 2e-5, weight decay 0.01, warmup 0.1, evaluation every epoch, best model by validation macro F1, early stopping patience 2, fp16 on GPU, seed 42.
- **Model C:** Skipped — no real conversation_id/message_order available.
- **Outputs:** `MyDrive/NLP_sentiment/model_b_runs/run_<time>/`. At the end, `model_b_results_<time>.zip` (metrics, plots, predictions, errors) is downloaded to your PC automatically."""),
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
PKG_ZIP = DRIVE / "model_b_package.zip"
if not PKG_ZIP.exists():
    print("Choose model_b_package.zip (D:\\\\NLP\\\\sentiment_app\\\\artifacts\\\\colab_package)")
    up = files.upload()
    PKG_ZIP.write_bytes(up[next(iter(up))])
PKG = Path("/content/pkg"); shutil.rmtree(PKG, ignore_errors=True)
with zipfile.ZipFile(PKG_ZIP) as z: z.extractall(PKG)
manifest = json.loads((PKG / "MANIFEST.json").read_text())
bad = [n for n, h in manifest.items() if hashlib.sha256((PKG / n).read_bytes()).hexdigest() != h]
if bad: raise SystemExit(f"Package files do not match MANIFEST.json: {bad}")
print(f"All {len(manifest)} package files verified.")"""),
    ("code", """!pip -q install datasets accelerate   # only installs what is missing; no upgrades"""),
    ("code", """RUN_ID = datetime.now().strftime("%Y%m%d_%H%M")
RUN = DRIVE / "model_b_runs" / f"run_{RUN_ID}"
sys.path.insert(0, str(PKG))
import train_model_b
metrics = train_model_b.main(["--data-dir", str(PKG / "data"), "--out-dir", str(RUN),
                              "--checkpoint-dir", "/content/checkpoints"])"""),
    ("code", """def zip_dir(src, dest, include_weights):
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED if not include_weights else zipfile.ZIP_STORED) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file() and (include_weights or not any(x in p.parts for x in ("xlm_roberta_multilingual", "tokenizer_multilingual"))):
                z.write(p, p.relative_to(src).as_posix())
small = DRIVE / f"model_b_results_{RUN_ID}.zip"
full = DRIVE / f"model_b_full_{RUN_ID}.zip"
zip_dir(RUN, small, include_weights=False)
zip_dir(RUN, full, include_weights=True)
print("Saved on Drive:", small, "and", full)
files.download(str(small))          # metrics/plots/predictions -> your Downloads folder
try:
    files.download(str(full))       # includes the 1.1 GB model; large downloads can fail - it stays on Drive
except Exception as e:
    print("Full model download did not start automatically; it is on Drive:", full, e)"""),
    ("code", """print("MODEL B TEST RESULTS (multilingual test split)")
for k in ("accuracy", "macro_f1", "weighted_f1", "best_validation_macro_f1"):
    print(f"  {k}: {metrics[k]:.4f}")
for lang, v in metrics["per_language"].items():
    print(f"  {lang}: macro F1 {v['macro_f1']:.4f} (n={v['n']})")
print("Model C: Skipped — no real conversation_id/message_order available.")
print("RUN COMPLETE:", RUN)"""),
]


def build_notebook(dest: Path) -> None:
    nb = nbf.v4.new_notebook()
    nb.cells = [nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
                for kind, src in NOTEBOOK_CELLS]
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"}}
    nbf.write(nb, dest)


def main() -> None:
    out = paths.COLAB_PACKAGE
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_zip(out / "model_b_package.zip")
    build_notebook(out / "colab_train_model_b.ipynb")
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"package: {out / 'model_b_package.zip'} ({(out / 'model_b_package.zip').stat().st_size / 1e6:.1f} MB, "
          f"{len(manifest)} files)\nnotebook: {out / 'colab_train_model_b.ipynb'}")


if __name__ == "__main__":
    main()
