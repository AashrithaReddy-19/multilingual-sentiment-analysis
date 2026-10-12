"""Import real Model B results produced by the Colab notebook.

Finds the newest model_b_full_*.zip (preferred, includes weights) or model_b_results_*.zip in the given
folder or in ~/Downloads, checks that it is a real run (not a smoke test) on exactly this project's split,
unpacks it into artifacts/multilingual/model_b and refreshes final_report.md.

    python -m pipeline.import_colab_results [FOLDER_OR_ZIP]
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402
from pipeline.write_final_report import main as write_report  # noqa: E402


def find_zip(arg: str | None) -> Path:
    if arg and Path(arg).is_file():
        return Path(arg)
    folder = Path(arg) if arg else Path.home() / "Downloads"
    for pattern in ("model_b_full_*.zip", "model_b_results_*.zip"):
        found = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        if found:
            return found[0]
    raise SystemExit(f"No model_b_full_*.zip or model_b_results_*.zip in {folder}")


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    src = find_zip(argv[0] if argv else None)
    dest = paths.MODEL_B
    if (dest / "metrics_model_b.json").exists():
        raise SystemExit(f"{dest} already holds Model B results; refusing to overwrite.")
    with zipfile.ZipFile(src) as z:
        if z.testzip() is not None:
            raise SystemExit(f"{src} is corrupted")
        metrics = json.loads(z.read("metrics_model_b.json"))
        if metrics.get("smoke_test"):
            raise SystemExit("This is a smoke-test run, not a real result; not imported.")
        local = {s: hashlib.sha256((paths.DATA / f"{s}.csv").read_bytes()).hexdigest()
                 for s in ("train", "validation", "test")}
        if metrics.get("split_sha256") != local:
            raise SystemExit("Model B was trained on a different split than artifacts/multilingual/data; not imported.")
        dest.mkdir(parents=True, exist_ok=True)
        z.extractall(dest)
    has_weights = (dest / "xlm_roberta_multilingual" / "model.safetensors").exists()
    print(f"Imported {src.name} -> {dest} (model weights included: {has_weights})")
    write_report()


if __name__ == "__main__":
    main()
