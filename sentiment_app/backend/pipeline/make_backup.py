"""Verified project backup: python -m pipeline.make_backup  (never overwrites; name includes date and time)."""
import zipfile
from collections import Counter
from datetime import date, datetime
from pathlib import Path

NLP = Path(r"D:\NLP")
APP = NLP / "sentiment_app"
ZIP = NLP / "backups" / f"sentiment_app_final_{datetime.now():%Y-%m-%d_%H%M}.zip"
if ZIP.exists():
    raise SystemExit(f"{ZIP} already exists; not overwriting.")

EXCL_DIRS = {"node_modules", "__pycache__", ".pytest_cache", "dist", "_checkpoints", ".vite"}
EXCL_SUFFIX = {".log", ".tmp", ".pyc", ".pyo"}
EXCL_NAMES = {".env"}

members = {}
for p in sorted(APP.rglob("*")):
    rel = p.relative_to(APP)
    if not p.is_file() or set(rel.parts) & EXCL_DIRS or p.suffix in EXCL_SUFFIX or p.name in EXCL_NAMES:
        continue
    if any("smoke" in part.lower() for part in rel.parts):
        continue
    members[f"sentiment_app/{rel.as_posix()}"] = p

# Telugu-only result references (small files; the 1.1 GB model is in telugu_run_outputs_2026-10-01.zip)
TEL = NLP / "outputs"
tel_files = ["metrics_baseline.json", "metrics_no_context.json", "metrics_context_aware.json", "model_comparison.csv",
             "per_language_results.csv", "confusion_matrix_baseline.csv", "confusion_matrix_baseline.png",
             "confusion_matrix_no_context.csv", "confusion_matrix_no_context.png",
             "confusion_matrices_telugu_side_by_side.png", "misclassified_baseline.csv", "misclassified_no_context.csv",
             "misclassified_no_context_30.csv", "label_mapping.json", "audit_report.json",
             "xlm_roberta_no_context/validation_history.csv", "xlm_roberta_no_context/config.json"]
for f in tel_files:
    members[f"telugu_only_reference/outputs/{f}"] = TEL / f
members["telugu_only_reference/telugu_results_report.ipynb"] = NLP / "telugu_results_report.ipynb"
missing = [k for k, p in members.items() if not p.exists()]
if missing:
    raise SystemExit(f"missing files: {missing}")

MODEL_B_LINE = ("Multilingual Model B: trained on a Colab GPU; real results in sentiment_app/artifacts/multilingual/model_b."
                if (APP / "artifacts" / "multilingual" / "model_b" / "metrics_model_b.json").exists()
                else "Multilingual Model B: Not yet trained — GPU Colab run required. (No Model B metrics exist.)")
note = f"""Final project backup ({date.today().isoformat()})

sentiment_app/            backend, frontend source, tests, docs, start scripts, Docker files,
                          multilingual data + real Model A artifacts, Colab Model B package.
telugu_only_reference/    real Telugu-only (TeSent) Model A/B metrics, tables, plots, errors and report.
                          The Telugu-only XLM-R weights (1.1 GB) are in D:\\NLP\\backups\\telugu_run_outputs_2026-10-01.zip.

Excluded: node_modules, __pycache__, .pytest_cache, frontend/dist (rebuild with `npm run build`), .env, logs, temp files.
{MODEL_B_LINE}
Model C: Skipped — no real conversation_id/message_order available.
Restore: unzip, then `cd frontend && npm install`, `pip install -r backend/requirements.txt`, `start_all.bat`.
"""

with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for name, p in members.items():
        z.write(p, name)
    z.writestr("BACKUP_CONTENTS.txt", note)

with zipfile.ZipFile(ZIP) as z:
    bad = z.testzip()
    infos = z.infolist()
    sizes_ok = all(z.getinfo(n).file_size == p.stat().st_size for n, p in members.items())
    names = z.namelist()

def group(n):
    parts = n.split("/")
    if parts[0] == "sentiment_app":
        return "/".join(parts[:3]) if parts[1] in ("artifacts", "backend", "frontend") and len(parts) > 3 else "/".join(parts[:2]) if len(parts) > 2 else "sentiment_app (root files)"
    return "/".join(parts[:1]) if len(parts) > 1 else "(root)"

summary = Counter(group(i.filename) for i in infos)
raw = Counter()
for i in infos:
    raw[group(i.filename)] += i.file_size
print(f"ZIP: {ZIP}")
print(f"size: {ZIP.stat().st_size / 1e6:.1f} MB compressed | {sum(i.file_size for i in infos) / 1e6:.1f} MB uncompressed | files: {len(infos)}")
print(f"CRC test: {'OK' if bad is None else 'FAILED at ' + bad} | sizes match sources: {sizes_ok}")
for k in sorted(summary):
    print(f"  {summary[k]:4d} files {raw[k] / 1e6:8.2f} MB  {k}")
leaks = [n for n in names if any(x in n for x in ("node_modules", "__pycache__", ".pytest_cache", "/dist/", ".log", "smoke", "_checkpoints"))]
if bad is not None or not sizes_ok or leaks:
    raise SystemExit("BACKUP VERIFICATION FAILED")
print("excluded-pattern check:", leaks or "none found")
print("model_b present:", any("/model_b/" in n for n in names))
