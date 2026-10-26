"""Build a Colab notebook that re-trains the validation-selected Transformer family winners whose weights were lost
(a Colab runtime ended before Google Drive finished uploading the weight files), scores each ONCE on test right after
training, and forces the Drive upload to finish before the session can end.

    python -m pipeline.build_colab_rerun

The winners are the best validation runs per family in experiments/transformers/ (rank: macro F1, macro recall,
accuracy); their saved config.json files are embedded, and each re-run must reproduce the same experiment id.
Output: artifacts/colab_package_phase3/colab_rerun_winners.ipynb (uses transformer_package.zip from Drive, or asks
for it). Results: MyDrive/NLP_sentiment/phase3_results_rerun.zip and phase3_model_<family>.zip.
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


INTRO = """# Phase 3 — re-train the validation-selected family winners (weights were lost)
The previous GPU session ended (Colab free-tier usage limit) before Google Drive finished uploading the trained weights.
This notebook re-trains ONLY the family winners chosen on validation, with their exact saved configurations, scores each
once on `test.csv` right after training, and calls `drive.flush_and_unmount()` at the end so every file is uploaded.
**Run all** on a GPU runtime and allow Google Drive. Model C: Skipped — no real conversation_id/message_order available."""

RERUN = """sys.path.insert(0, str(PKG))
import colab_runner
RUNS2 = DRIVE / "phase3_rerun_winners"
runner = colab_runner.Runner(PKG / "data", RUNS2)
WINNERS = json.loads(r'''{winners}''')

def export(tag):
    z_path = DRIVE / "phase3_results_rerun.zip"
    with zipfile.ZipFile(z_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(RUNS2.rglob("*")):
            if p.is_file() and not any(part.startswith("_") for part in p.relative_to(RUNS2).parts):
                z.write(p, p.relative_to(RUNS2).as_posix())
        if (RUNS2 / "_selected_for_test.json").exists():
            z.write(RUNS2 / "_selected_for_test.json", "_selected_for_test.json")
    print("EXPORTED", tag, z_path, f"({{z_path.stat().st_size / 1e6:.1f}} MB)")

for saved in WINNERS:
    cfg = colab_runner.cfg_from_saved(saved)
    eid = colab_runner.exp_id(dict(colab_runner.DEFAULT, **cfg))
    assert eid == saved["experiment_id"], (eid, saved["experiment_id"])
    fam = cfg["family"]
    print("=== re-training", fam, eid)
    runner.run(cfg)
    runner.final_test()                      # tests the best (= only) run of each family present, once
    t = json.loads((RUNS2 / eid / "test_metrics.json").read_text())
    print(f"{{fam}}: {{eid}} TEST macro F1 {{t['macro_f1']:.4f}} acc {{t['accuracy']:.4f}}")
    best = RUNS2 / f"_best_model_{{fam}}" / "model"
    mz = DRIVE / f"phase3_model_{{fam}}.zip"
    with zipfile.ZipFile(mz, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(best.iterdir()):
            if p.is_file(): z.write(p, p.name)
    print("WEIGHTS", fam, mz, f"({{mz.stat().st_size / 1e9:.2f}} GB)")
    export(fam)"""

FLUSH = """# Wait until Google Drive has really uploaded everything (the mount writes to a local cache first).
from google.colab import drive
drive.flush_and_unmount()
print("DRIVE_FLUSHED: all files uploaded")"""


def main() -> None:
    ws = winners()
    if not ws:
        raise SystemExit("No GPU family winners found in experiments/transformers/.")
    setup = next(s for k, s in bct.CELLS if k == "code" and "drive.mount" in s)
    install = next(s for k, s in bct.CELLS if k == "code" and "pip -q install" in s)
    cells = [nbf.v4.new_markdown_cell(INTRO), nbf.v4.new_code_cell(setup), nbf.v4.new_code_cell(install),
             nbf.v4.new_code_cell(RERUN.format(winners=json.dumps(ws, indent=1))), nbf.v4.new_code_cell(FLUSH)]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4"}}
    nbf.write(nb, OUT)
    print(f"notebook: {OUT} | winners: {[w['experiment_id'] for w in ws]}")


if __name__ == "__main__":
    main()
