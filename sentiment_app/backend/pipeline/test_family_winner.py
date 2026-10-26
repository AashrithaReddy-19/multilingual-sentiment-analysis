"""Score a downloaded Transformer family winner on test.csv ONCE (on this PC), when the GPU session ended before the
notebook's Final test cell ran. The weights come from the run's Drive folder _best_model_<family>/model.

    python -m pipeline.test_family_winner MODEL_DIR [--eval-batch-size 32]

Rules enforced: MODEL_DIR/phase3_config.json must be a real (non-smoke) run on this exact split, already imported into
experiments/transformers/, and it must be the BEST VALIDATION run of its family there (macro F1, then macro recall,
then accuracy) - only family winners are tested. A run is never tested twice. Afterwards the run is recorded in
_selected_for_test.json and its weights are packed as artifacts/colab_results/phase3_model_<family>.zip for
pipeline.install_phase3_model.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402
from pipeline import train_transformer as tt  # noqa: E402
from pipeline.finalize_phase3 import FAMILY_OF_MODEL, TRANS, rank_key, split_hashes, transformer_runs  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", type=Path)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    args = ap.parse_args(argv)
    cfg = json.loads((args.model_dir / "phase3_config.json").read_text(encoding="utf-8"))
    eid, fam = cfg["experiment_id"], FAMILY_OF_MODEL[cfg["model"]]
    if cfg.get("smoke_test") or cfg.get("split_sha256") != split_hashes():
        raise SystemExit(f"{eid}: smoke test or different split; not tested.")
    if not (args.model_dir / "model.safetensors").exists() and not (args.model_dir / "pytorch_model.bin").exists():
        raise SystemExit(f"{args.model_dir} has no weights.")
    run = TRANS / eid
    if not (run / "config.json").exists():
        raise SystemExit(f"{eid} is not imported (finalize_phase3 import first).")
    if json.loads((run / "config.json").read_text(encoding="utf-8")) != cfg:
        raise SystemExit(f"{eid}: the weights' phase3_config.json differs from the imported run config.")
    fam_runs = [r for r in transformer_runs() if r["family"] == fam]
    best = max(fam_runs, key=lambda r: rank_key(r["val"]))
    if best["id"] != eid:
        raise SystemExit(f"{eid} is not the best validation {fam} run ({best['id']}); only family winners are tested.")
    if not (run / "test_metrics.json").exists():
        tt.main(["--data-dir", str(paths.MULTI / "data"), "--out-dir", str(TRANS), "--exp-id", eid,
                 "--predict-test", str(args.model_dir), "--eval-batch-size", str(args.eval_batch_size)])
    sel_file = TRANS / "_selected_for_test.json"
    sel = json.loads(sel_file.read_text(encoding="utf-8")) if sel_file.exists() else {}
    sel[fam] = eid
    sel_file.write_text(json.dumps(sel, indent=2), encoding="utf-8")
    out = paths.APP_ROOT / "artifacts" / "colab_results" / f"phase3_model_{fam}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for f in sorted(args.model_dir.iterdir()):
            if f.is_file():
                z.write(f, f.name)
    t = json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
    print(f"{fam}: {eid} TEST macro F1 {t['macro_f1']:.4f} acc {t['accuracy']:.4f}; selected; weights -> {out}")


if __name__ == "__main__":
    main()
