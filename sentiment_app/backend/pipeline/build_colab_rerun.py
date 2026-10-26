"""Build a Colab notebook that re-trains the validation-selected Transformer family winners whose weights were lost
(a Colab runtime ended before Google Drive finished uploading the weight files), scores each ONCE on test right after
training, and forces the Drive upload to finish before the session can end.

    python -m pipeline.build_colab_rerun

The winners are the best validation runs per family in experiments/transformers/ (rank: macro F1, macro recall,
accuracy); their saved config.json files are embedded, and each re-run must reproduce the same experiment id.
Output: artifacts/colab_package_phase3/colab_rerun_winners.ipynb (uses transformer_package.zip from Drive, or asks
for it). Results per family: MyDrive/NLP_sentiment/phase3_rerun/phase3_results_rerun_<family>.zip and
phase3_model_<family>.zip, each verified on Drive before the next family starts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import build_colab_transformers as bct  # noqa: E402
from pipeline.finalize_phase3 import rank_key, transformer_runs  # noqa: E402

OUT = bct.OUT / "colab_rerun_winners.ipynb"
PRIORITY = ("indicbert", "xlmr", "muril", "mdeberta")      # best validation family first


def winners() -> list[dict]:
    best = {}
    for r in transformer_runs():
        if r["family"] == "e5" or r["cfg"].get("device") == "cpu":
            continue
        if r["family"] not in best or rank_key(r["val"]) > rank_key(best[r["family"]]["val"]):
            best[r["family"]] = r
    return [best[f]["cfg"] for f in PRIORITY if f in best]


INTRO = """# Phase 3 — recovery: re-train the validation-selected family winners (weights were lost)
The previous GPU session ended (Colab free-tier usage limit) before Google Drive finished uploading the trained weights.
This notebook re-trains ONLY the family winners chosen on validation, with their exact saved configurations (no new
hyperparameters), one family at a time: train (best epoch chosen on validation by early stopping, as before) -> score
`test.csv` once -> model ZIP + results ZIP to Drive -> `drive.flush_and_unmount()` + remount -> verify the ZIPs on Drive
(size, CRC, required files) -> only then the next family. Re-running resumes: verified families are skipped.
**Run all** on a GPU runtime and allow Google Drive. Model C: Skipped — no real conversation_id/message_order available."""

RERUN = """import shutil
from google.colab import drive
sys.path.insert(0, str(PKG))
import colab_runner
RUNS2 = Path("/content/rerun_runs"); RUNS2.mkdir(parents=True, exist_ok=True)   # work on the VM disk
OUTD = DRIVE / "phase3_rerun"; OUTD.mkdir(parents=True, exist_ok=True)          # only verified ZIPs go to Drive
runner = colab_runner.Runner(PKG / "data", RUNS2)
WINNERS = json.loads(r\'\'\'{winners}\'\'\')
WEIGHTS = ("model.safetensors", "pytorch_model.bin")
TOKENIZER = ("tokenizer.json", "sentencepiece.bpe.model", "spiece.model", "vocab.txt")
REQUIRED = ("config.json", "phase3_config.json", "tokenizer_config.json")

# the exact frozen split (the package itself was already checked file-by-file against MANIFEST.json above)
import hashlib, pandas as pd
ROWS = {{"train": 70540, "validation": 15121, "test": 15124}}
for split, n in ROWS.items():
    df = pd.read_csv(PKG / "data" / f"{{split}}.csv", encoding="utf-8-sig", low_memory=False)
    assert len(df) == n, (split, len(df))
    h = hashlib.sha256((PKG / "data" / f"{{split}}.csv").read_bytes()).hexdigest()
    assert all(w["split_sha256"][split] == h for w in WINNERS), f"{{split}}.csv differs from the winners' split"
print("SPLIT_OK", ROWS)

def check_model_zip(path, eid):
    if not path.exists():
        return "missing"
    size = path.stat().st_size
    if size < 300e6:
        return f"too small ({{size / 1e6:.1f}} MB)"
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            if z.testzip() is not None:
                return "CRC error"
            if json.loads(z.read("phase3_config.json")).get("experiment_id") != eid:
                return "wrong experiment"
    except (zipfile.BadZipFile, KeyError, OSError, ValueError) as e:      # e.g. an upload that never completed
        return f"unreadable ({{type(e).__name__}})"
    if not any(w in names for w in WEIGHTS):
        return "no weight file"
    if not any(t in names for t in TOKENIZER):
        return "no tokenizer file"
    missing = [r for r in REQUIRED if r not in names]
    return f"missing {{missing}}" if missing else "ok"

def check_results_zip(path, eid):
    if not path.exists():
        return "missing"
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            if z.testzip() is not None:
                return "CRC error"
    except (zipfile.BadZipFile, OSError) as e:
        return f"unreadable ({{type(e).__name__}})"
    need = [f"{{eid}}/{{n}}" for n in ("config.json", "validation_metrics.json", "predictions.csv", "test_metrics.json",
                                       "predictions_test.csv")]
    missing = [n for n in need if n not in names]
    return f"missing {{missing}}" if missing else "ok"

def persist_and_verify(mz, rz, eid):
    drive.flush_and_unmount()                # blocks until every cached file is uploaded
    drive.mount("/content/drive")            # files are now read back from Drive itself
    m, r = check_model_zip(mz, eid), check_results_zip(rz, eid)
    print("VERIFY", mz.name, m, f"{{mz.stat().st_size / 1e9:.2f}} GB" if mz.exists() else "", "|", rz.name, r)
    if m != "ok" or r != "ok":
        raise RuntimeError(f"Drive verification failed for {{eid}}: model {{m}}, results {{r}}")

for saved in WINNERS:
    cfg = colab_runner.cfg_from_saved(saved)
    eid = colab_runner.exp_id(dict(colab_runner.DEFAULT, **cfg))
    assert eid == saved["experiment_id"], (eid, saved["experiment_id"])       # exact saved configuration
    fam = cfg["family"]
    mz, rz = OUTD / f"phase3_model_{{fam}}.zip", OUTD / f"phase3_results_rerun_{{fam}}.zip"
    if check_model_zip(mz, eid) == "ok" and check_results_zip(rz, eid) == "ok":
        print("SKIP (already verified on Drive):", fam, eid)
        continue
    print("=== re-training", fam, eid, flush=True)
    shutil.rmtree(RUNS2 / eid, ignore_errors=True)       # a half-finished earlier attempt is never reused
    shutil.rmtree(RUNS2 / f"_best_model_{{fam}}", ignore_errors=True)
    runner.run(cfg)                                      # validation-only training; best epoch by val macro F1
    runner.final_test()                                  # test.csv once, for this family's (only) run
    v = json.loads((RUNS2 / eid / "validation_metrics.json").read_text())
    t = json.loads((RUNS2 / eid / "test_metrics.json").read_text())
    c = json.loads((RUNS2 / eid / "config.json").read_text())
    print(f"{{fam}}: best epoch {{c['best_epoch']:g}} | VAL macro F1 {{v['macro_f1']:.4f}} | TEST macro F1 "
          f"{{t['macro_f1']:.4f}} acc {{t['accuracy']:.4f}}")
    local_m = Path(f"/content/phase3_model_{{fam}}.zip")
    with zipfile.ZipFile(local_m, "w", zipfile.ZIP_STORED) as z:
        for p in sorted((RUNS2 / f"_best_model_{{fam}}" / "model").iterdir()):
            if p.is_file():
                z.write(p, p.name)
    local_r = Path(f"/content/phase3_results_rerun_{{fam}}.zip")
    with zipfile.ZipFile(local_r, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((RUNS2 / eid).rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(RUNS2).as_posix())
        z.writestr("_selected_for_test.json", json.dumps({{fam: eid}}))
    assert check_model_zip(local_m, eid) == "ok", check_model_zip(local_m, eid)
    shutil.copy(local_m, mz)
    shutil.copy(local_r, rz)
    persist_and_verify(mz, rz, eid)
    print("PERSISTED", fam, eid, flush=True)
print("ALL_FAMILIES_PERSISTED")"""


def main() -> None:
    ws = winners()
    if not ws:
        raise SystemExit("No GPU family winners found in experiments/transformers/.")
    setup = next(s for k, s in bct.CELLS if k == "code" and "drive.mount" in s)
    install = next(s for k, s in bct.CELLS if k == "code" and "pip -q install" in s)
    cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(setup), nbf.v4.new_code_cell(install),
             nbf.v4.new_code_cell(RERUN.format(winners=json.dumps(ws, indent=1)))]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"}}
    nbf.write(nb, OUT)
    print(f"notebook: {OUT} | winners: {[w['experiment_id'] for w in ws]}")


if __name__ == "__main__":
    main()
