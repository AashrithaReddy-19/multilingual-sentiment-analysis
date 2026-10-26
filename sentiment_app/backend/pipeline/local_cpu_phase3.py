"""Hand a finished LOCAL (CPU) Transformer run to the Phase-3 finalisation, the same way the Colab notebook does.

    python -m pipeline.local_cpu_phase3 EXP_ID [--runs-dir artifacts/multilingual/experiments/transformers_cpu]

The run must have been trained with pipeline.train_transformer --allow-cpu --keep-model (validation only). This script
1. refuses smoke runs and runs on a different split,
2. predicts test.csv ONCE for this run, which is the only (hence validation-selected) run of its family,
3. copies the run (without weights) to experiments/transformers/ and records it in _selected_for_test.json,
4. writes artifacts/colab_results/phase3_model_<family>_local.zip for pipeline.install_phase3_model.
Then run: finalize_phase3, install_phase3_model, write_phase3_report.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402
from pipeline import train_transformer as tt  # noqa: E402
from pipeline.finalize_phase3 import FAMILY_OF_MODEL, TRANS, split_hashes  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_id")
    ap.add_argument("--runs-dir", type=Path, default=paths.MULTI / "experiments" / "transformers_cpu")
    args = ap.parse_args(argv)
    run = args.runs_dir / args.exp_id
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    if cfg.get("smoke_test"):
        raise SystemExit(f"{args.exp_id} is a smoke test; not a result.")
    if cfg.get("split_sha256") != split_hashes():
        raise SystemExit(f"{args.exp_id} was trained on a different split.")
    if not (run / "model" / "model.safetensors").exists() and not (run / "model" / "pytorch_model.bin").exists():
        raise SystemExit(f"{run / 'model'} has no weights (train with --keep-model).")
    fam = FAMILY_OF_MODEL[cfg["model"]]

    selected = json.loads((TRANS / "_selected_for_test.json").read_text(encoding="utf-8")) \
        if (TRANS / "_selected_for_test.json").exists() else {}
    prev = selected.get(fam)
    if prev not in (None, args.exp_id):
        from pipeline.finalize_phase3 import rank_key
        new_v = json.loads((run / "validation_metrics.json").read_text(encoding="utf-8"))
        old_v = json.loads((TRANS / prev / "validation_metrics.json").read_text(encoding="utf-8"))
        if rank_key(new_v) <= rank_key(old_v):
            raise SystemExit(f"{args.exp_id} (val macro F1 {new_v['macro_f1']:.4f}) does not beat the selected {fam} run "
                             f"{prev} ({old_v['macro_f1']:.4f}) on validation; not tested.")
        print(f"{args.exp_id} beats {prev} on validation ({new_v['macro_f1']:.4f} > {old_v['macro_f1']:.4f}); "
              f"it becomes the {fam} family winner.")

    if not (run / "test_metrics.json").exists():
        tt.main(["--data-dir", str(paths.MULTI / "data"), "--out-dir", str(args.runs_dir), "--exp-id", args.exp_id,
                 "--predict-test", str(run / "model")])

    TRANS.mkdir(parents=True, exist_ok=True)
    dest = TRANS / args.exp_id
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(run, dest, ignore=shutil.ignore_patterns("model", "_checkpoints"))
    selected[fam] = args.exp_id
    (TRANS / "_selected_for_test.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    out = paths.APP_ROOT / "artifacts" / "colab_results" / f"phase3_model_{fam}_local.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted((run / "model").iterdir()):
            if f.is_file():
                z.write(f, f.name)
        z.write(run / "config.json", "phase3_config.json")
    print(f"{args.exp_id}: test predicted, copied to {dest}, selected for {fam}; weights -> {out}")


if __name__ == "__main__":
    main()
