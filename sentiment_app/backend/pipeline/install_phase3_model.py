"""Install Transformer weights downloaded from the Phase-3 Colab notebook (phase3_model_<family>_<time>.zip) so the web
app can serve them. Checks that the weights belong to the validation-selected run that was imported
(experiments/transformers/_selected_for_test.json), on this project's exact split, and not a smoke test. Then loads the
model on CPU and times one prediction.

    python -m pipeline.install_phase3_model [ZIP_OR_FOLDER ...]     (default: artifacts/colab_results and Downloads)
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import paths  # noqa: E402
from pipeline.finalize_phase3 import FAMILY_OF_MODEL, TRANS, split_hashes  # noqa: E402

DEST = paths.MULTI / "phase3_models"


def candidates(args: list[str]) -> list[Path]:
    if args:
        out = []
        for a in args:
            p = Path(a)
            out += [p] if p.is_file() else sorted(p.glob("phase3_model_*.zip"))
        return out
    found = {}
    for folder in (paths.APP_ROOT / "artifacts" / "colab_results", Path.home() / "Downloads"):
        for z in sorted(folder.glob("phase3_model_*.zip"), key=lambda p: p.stat().st_mtime) if folder.is_dir() else []:
            found[z.name.split("_")[2]] = z               # newest per family wins
    return list(found.values())


def install(zpath: Path) -> None:
    selected = json.loads((TRANS / "_selected_for_test.json").read_text(encoding="utf-8")) \
        if (TRANS / "_selected_for_test.json").exists() else {}
    with zipfile.ZipFile(zpath) as z:
        if z.testzip() is not None:
            raise SystemExit(f"{zpath} is corrupted")
        cfg = json.loads(z.read("phase3_config.json"))
        fam = FAMILY_OF_MODEL[cfg["model"]]
        if cfg.get("smoke_test"):
            raise SystemExit(f"{zpath.name}: smoke-test weights; not installed.")
        if cfg.get("split_sha256") != split_hashes():
            raise SystemExit(f"{zpath.name}: trained on a different split; not installed.")
        if selected.get(fam) != cfg["experiment_id"]:
            raise SystemExit(f"{zpath.name}: {cfg['experiment_id']} is not the imported validation-selected {fam} run "
                             f"({selected.get(fam)}). Import phase3_results first (finalize_phase3 --import).")
        dest = DEST / fam
        tmp = DEST / f"_{fam}_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        z.extractall(tmp)
    shutil.rmtree(dest, ignore_errors=True)
    tmp.rename(dest)
    print(f"installed {fam}: {cfg['experiment_id']} -> {dest}")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(dest)
    model = AutoModelForSequenceClassification.from_pretrained(dest).eval()
    load_s = time.time() - t0
    text = ("[LANG_ENGLISH] " if cfg.get("lang_tokens") else "") + "The delivery was late again. Not happy."
    t0 = time.time()
    with torch.no_grad():
        p = torch.softmax(model(**tok([text], return_tensors="pt", truncation=True, max_length=cfg["max_length"])).logits, -1)
    print(f"  CPU check: load {load_s:.1f}s, one prediction {1000 * (time.time() - t0):.0f} ms, probs {p.numpy().round(3).tolist()}")


def main(argv=None) -> None:
    zips = candidates(sys.argv[1:] if argv is None else argv)
    if not zips:
        raise SystemExit("No phase3_model_*.zip found (artifacts/colab_results or Downloads).")
    for z in zips:
        install(z)
    print("Restart the backend to serve the installed model (it is used only if final_selection.json selects it).")


if __name__ == "__main__":
    main()
