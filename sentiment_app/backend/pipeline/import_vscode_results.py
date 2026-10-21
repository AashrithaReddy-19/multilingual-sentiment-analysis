"""Turn the outputs saved in the executed VS Code Colab notebook (colab_vscode_phase3.ipynb) back into ordinary run
folders and pack them as artifacts/colab_results/phase3_results_vscode_<time>.zip for finalize_phase3 --import.

Probabilities travel as uint16 (x10^4); predictions.csv files are rebuilt against the LOCAL split files (row_id order
checked). validation_metrics.json / test_metrics.json are kept exactly as computed on the GPU, and the rebuilt
predictions are checked against them.

    python -m pipeline.import_vscode_results [NOTEBOOK]
"""
from __future__ import annotations

import base64
import io
import json
import lzma
import shutil
import sys
import tarfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402
from pipeline.exp_common import read_split  # noqa: E402
from sentiment_core.experiment_io import full_metrics, save_eval_files  # noqa: E402

NB = paths.APP_ROOT / "artifacts" / "colab_package_phase3" / "colab_vscode_phase3.ipynb"
RESULTS = paths.APP_ROOT / "artifacts" / "colab_results"
MIME = "text/x-phase3-chunk"


def exports(nb_path: Path) -> dict[str, bytes]:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    parts: dict[str, dict] = {}
    for cell in nb["cells"]:
        for out in cell.get("outputs", []):
            raw = out.get("data", {}).get(MIME)
            if raw is None:
                continue
            c = json.loads("".join(raw) if isinstance(raw, list) else raw)
            parts.setdefault(c["export"], {"of": c["of"], "p": {}})["p"][c["part"]] = c["data"]
    done = {}
    for tag, v in parts.items():
        if len(v["p"]) != v["of"]:
            print(f"  export '{tag}': {len(v['p'])}/{v['of']} parts saved -> skipped (cell still running or not saved)")
            continue
        done[tag] = lzma.decompress(base64.b64decode("".join(v["p"][i] for i in range(v["of"]))))
    return done


def main(argv=None) -> Path | None:
    args = sys.argv[1:] if argv is None else argv
    nb_path = Path(args[0]) if args else NB
    ex = exports(nb_path)
    if not ex:
        print(f"No complete exports in {nb_path} yet.")
        return None
    splits = {"validation": read_split("validation"), "test": read_split("test")}
    tmp = RESULTS / "_vscode_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    order = [t for t in ex if t != "final"] + (["final"] if "final" in ex else [])   # later exports overwrite
    for tag in order:
        with tarfile.open(fileobj=io.BytesIO(ex[tag])) as t:
            for m in t.getmembers():
                dest = tmp / m.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(t.extractfile(m).read())
    n = 0
    for run in sorted(p for p in tmp.iterdir() if p.is_dir()):
        for npy in sorted(run.rglob("*.csv.npy")):
            split = "test" if npy.name.startswith("predictions_test") else "validation"
            df = splits[split]
            probs = np.load(npy).astype(np.float64) / 10000.0
            probs = probs / probs.sum(1, keepdims=True)
            if len(probs) != len(df):
                raise SystemExit(f"{npy}: {len(probs)} rows, local {split} has {len(df)}")
            folder = npy.parent
            keep = {f: (folder / f).read_bytes() for f in (f"{split}_metrics.json",) if (folder / f).exists()}
            m = save_eval_files(folder, df["label"], probs, df, split, run.name)
            for f, b in keep.items():                 # keep the GPU-computed metrics, check the rebuilt predictions
                ref = json.loads(b)
                if abs(ref["macro_f1"] - m["macro_f1"]) > 1e-3:
                    raise SystemExit(f"{folder.name}/{f}: rebuilt macro F1 {m['macro_f1']:.4f} != {ref['macro_f1']:.4f}")
                (folder / f).write_bytes(b)
            npy.unlink()
        n += 1
    RESULTS.mkdir(parents=True, exist_ok=True)
    zp = RESULTS / f"phase3_results_vscode_{datetime.now():%Y%m%d_%H%M}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(tmp.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(tmp).as_posix())
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"Rebuilt {n} runs from exports {sorted(ex)} -> {zp}")
    return zp


if __name__ == "__main__":
    main()
