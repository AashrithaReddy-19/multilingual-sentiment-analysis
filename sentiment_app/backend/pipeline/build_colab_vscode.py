"""Build a self-contained Phase-3 notebook for the Google Colab VS Code extension (Select Kernel > Colab).

The VS Code Colab runtime cannot read local files and cannot download into this PC, so:
- the verified transformer package (train/validation/test + code, MANIFEST-checked) is embedded in the notebook as
  xz + base64 chunks (the notebook is gitignored: it contains the dataset splits, which are not redistributable);
- every plan step exports its finished runs as notebook outputs (mime type text/x-phase3-chunk): configs, metrics,
  per-epoch history and probabilities (uint16, x10^4) - VS Code saves outputs into the local .ipynb, and
  pipeline.import_vscode_results turns them back into ordinary run folders for finalize_phase3 --import.
- runs and the best weights per family are stored on Google Drive (if mounted) so Run all resumes after a disconnect.

    python -m pipeline.build_colab_vscode
"""
from __future__ import annotations

import base64
import io
import lzma
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import build_colab_transformers as bct  # noqa: E402

OUT = bct.OUT / "colab_vscode_phase3.ipynb"
CHUNK_CHARS = 1_000_000
LINE = 4000

INTRO = """# Phase 3 Transformer study — Google Colab in VS Code

1. **Select Kernel** (top right) → **Colab** → **Auto Connect** (or *New Colab Server* → **GPU**: A100 / L4 / T4) and sign in once.
2. **Run All**. Allow Google Drive when asked (it keeps finished runs across disconnects; without it runs live only in this session).
3. Keep VS Code open. If the runtime disconnects, Run All again: finished runs are reused.

The package (exact frozen split, SHA-256 verified) is embedded below. `test.csv` is used only by the *Final test* cell, for the
best validation run of each family. Model C: Skipped — no real conversation_id/message_order available."""

SETUP = """import torch, sys, json, base64, lzma, tarfile, io, hashlib, shutil, os, time
print("torch.cuda.is_available():", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("No GPU: Select Kernel > Colab > New Colab Server > GPU, then Run All again.")
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name} | VRAM {p.total_memory / 2**30:.1f} GiB | CUDA (torch build) {torch.version.cuda} | torch {torch.__version__}")
_P = []"""

UNPACK = """from pathlib import Path
PKG = Path("/content/pkg"); shutil.rmtree(PKG, ignore_errors=True); PKG.mkdir(parents=True)
with tarfile.open(fileobj=io.BytesIO(lzma.decompress(base64.b64decode("".join(_P))))) as t:
    t.extractall(PKG)
del _P
manifest = json.loads((PKG / "MANIFEST.json").read_text())
bad = [n for n, h in manifest.items() if hashlib.sha256((PKG / n).read_bytes()).hexdigest() != h]
if bad: raise SystemExit(f"package files do not match MANIFEST.json: {bad}")
print(f"All {len(manifest)} package files verified:", (PKG / "VERSION.txt").read_text())"""

INSTALL = """!pip -q install datasets accelerate sentencepiece protobuf"""

DRIVE = """try:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = Path("/content/drive/MyDrive/NLP_sentiment")
except Exception as exc:
    print("Google Drive not mounted (runs are kept only in this session):", exc)
    BASE = Path("/content/NLP_sentiment")
RUNS = BASE / "phase3_runs"; RUNS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(PKG))
import numpy as np, pandas as pd, colab_runner
from IPython.display import display
runner = colab_runner.Runner(PKG / "data", RUNS)
DATA = {s: pd.read_csv(PKG / "data" / f"{s}.csv", encoding="utf-8-sig", low_memory=False) for s in ("validation", "test")}

def _probs(csv, split):
    df = pd.read_csv(csv, encoding="utf-8-sig")
    assert np.array_equal(df["row_id"].to_numpy(), DATA[split]["row_id"].to_numpy()), csv
    return np.round(df[["prob_Negative", "prob_Neutral", "prob_Positive"]].to_numpy() * 10000).astype(np.uint16)

def export(tag, families):
    \"\"\"Emit every finished (non-smoke) run of these families as notebook outputs for import_vscode_results.\"\"\"
    buf = io.BytesIO()
    n = 0
    with tarfile.open(fileobj=buf, mode="w") as t:
        def add(name, data):
            ti = tarfile.TarInfo(name); ti.size = len(data); t.addfile(ti, io.BytesIO(data))
        for d in sorted(RUNS.iterdir()):
            cfgf = d / "config.json"
            if d.name.startswith("_") or not (d / "validation_metrics.json").exists() or not cfgf.exists():
                continue
            cfg = json.loads(cfgf.read_text())
            if cfg.get("smoke_test") or colab_runner.FAMILY_OF.get(cfg["model"]) not in families:
                continue
            for f in ("config.json", "validation_metrics.json", "validation_history.csv", "test_metrics.json"):
                if (d / f).exists():
                    add(f"{d.name}/{f}", (d / f).read_bytes())
            for rel, split in (("predictions.csv", "validation"), ("predictions_test.csv", "test"),
                               ("detected_lang/predictions.csv", "validation"),
                               ("detected_lang/predictions_test.csv", "test")):
                if (d / rel).exists():
                    b = io.BytesIO(); np.save(b, _probs(d / rel, split)); add(f"{d.name}/{rel}.npy", b.getvalue())
            n += 1
        sel = RUNS / "_selected_for_test.json"
        if sel.exists() and tag == "final":
            add("_selected_for_test.json", sel.read_bytes())
    payload = base64.b64encode(lzma.compress(buf.getvalue(), preset=9)).decode()
    parts = [payload[i:i + 1_000_000] for i in range(0, len(payload), 1_000_000)] or [""]
    for i, p in enumerate(parts):
        display({"text/x-phase3-chunk": json.dumps({"export": tag, "part": i, "of": len(parts), "runs": n, "data": p})},
                raw=True)
    print(f"exported {n} runs ({len(payload) / 1e6:.1f} MB) as '{tag}'")"""

FINAL = """selected = runner.final_test()
print("Selected on validation and scored once on test:", selected)
for fam, eid in selected.items():
    t = json.loads((RUNS / eid / "test_metrics.json").read_text())
    print(f"  {fam}: {eid}  TEST macro F1 {t['macro_f1']:.4f}  acc {t['accuracy']:.4f}")
runner.report("final")
export("final", set(selected))
import zipfile
for fam in selected:            # weights are too large for notebook outputs: one zip per family winner on Drive
    best = RUNS / f"_best_model_{fam}" / "model"
    mz = BASE / f"phase3_model_{fam}.zip"     # contains phase3_config.json, as pipeline.install_phase3_model expects
    with zipfile.ZipFile(mz, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(best.iterdir()):
            if f.is_file():
                z.write(f, f.name)
    print(f"{fam}: {mz} ({mz.stat().st_size / 1e9:.2f} GB) - download it only if finalize_phase3 selects a model "
          f"that uses {fam}")"""


def package_chunks() -> list[str]:
    with tempfile.TemporaryDirectory() as tmp:
        zp = Path(tmp) / "pkg.zip"
        bct.build_zip(zp)
        buf = io.BytesIO()
        with zipfile.ZipFile(zp) as z, tarfile.open(fileobj=buf, mode="w") as t:
            for n in z.namelist():
                d = z.read(n)
                ti = tarfile.TarInfo(n)
                ti.size = len(d)
                t.addfile(ti, io.BytesIO(d))
    b64 = base64.b64encode(lzma.compress(buf.getvalue(), preset=9 | lzma.PRESET_EXTREME)).decode()
    return [b64[i:i + CHUNK_CHARS] for i in range(0, len(b64), CHUNK_CHARS)]


def main() -> None:
    from colab_runner import PLAN   # noqa: E402  (pipeline/ is on sys.path via build_colab_transformers' import)
    cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(SETUP)]
    for i, ch in enumerate(package_chunks()):
        lines = "\n".join(ch[j:j + LINE] for j in range(0, len(ch), LINE))
        cells.append(nbf.v4.new_code_cell(f'# embedded package part {i + 1}\n_P.append("""\n{lines}\n""".replace("\\n", ""))'))
    cells += [nbf.v4.new_code_cell(UNPACK), nbf.v4.new_code_cell(INSTALL), nbf.v4.new_code_cell(DRIVE)]
    for fam, tier in PLAN:
        cells.append(nbf.v4.new_code_cell(
            f'# plan step: {fam} {tier}\nrunner.plan([("{fam}", "{tier}")])\nexport("{fam}_{tier}", {{"{fam}"}})'))
    cells.append(nbf.v4.new_code_cell(FINAL))
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU"}
    nbf.write(nb, OUT)
    print(f"notebook: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {len(cells)} cells)")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
