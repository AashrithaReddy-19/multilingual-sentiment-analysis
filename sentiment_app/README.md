# Sentiment Lab: Sentiment Analysis for Telugu, English, and Telugu-English Code-Mixed Chat

Three-class sentiment (0 = Negative, 1 = Neutral, 2 = Positive) for Telugu, English and Telugu-English code-mixed text.
The project has a FastAPI backend, a React + TypeScript + Vite + Tailwind web app, a reproducible data and
experiment pipeline, and a resumable Colab GPU package for the Transformer models.

Every metric shown anywhere comes from saved, executed results. Pending experiments show no numbers.

## Status at a glance
| Run | Status |
|---|---|
| Multilingual A (word TF-IDF + LR) | trained — test macro F1 0.6637 |
| **Multilingual A+** (word + char TF-IDF + LR) | **trained — best completed multilingual model**, test macro F1 0.7001, accuracy 0.6994 |
| Classical search (177 validation-only configs) | done — nothing beat A+ on validation; classical phase closed |
| multilingual-e5-small, XLM-R Model B, MuRIL / IndicBERTv2 / mDeBERTa, pair/triple and language-aware ensembles | **Not trained / pending GPU run** (Colab package phase3-v5 ready: staged per-family search) |
| Model C (context-aware) | Skipped — no real conversation_id/message_order available. |
| Telugu-only run (`D:\NLP\outputs`, different test split) | A 0.6232 / XLM-R 0.6985 test macro F1 |

Report: [docs/final_report_phase3.md](docs/final_report_phase3.md) (regenerated from the saved artifacts).

## Environment (one environment for everything)
- **Python 3.11**: `C:\Users\sresh\AppData\Local\Programs\Python\Python311\python.exe`. The other installed Pythons
  (3.12, 3.13) do **not** have torch/transformers; do not use them for this project.
- VS Code / Pylance: `D:\NLP\.vscode\settings.json` pins this interpreter.
- Jupyter: kernel **Python (NLP Sentiment)** (`nlp-sentiment`), registered with
  `py -3.11 -m ipykernel install --user --name nlp-sentiment --display-name "Python (NLP Sentiment)"`.
- Install (only if setting up a new machine): `py -3.11 -m pip install -r backend\requirements.txt jupyter ipykernel openpyxl evaluate sentencepiece`
- Node 20+ (verified with Node 25.9, npm 11).

## Run the app (Windows)
```bat
cd /d D:\NLP\sentiment_app
start_all.bat              :: both servers + browser
start_backend.bat          :: API on http://127.0.0.1:8000 (docs: /docs) — uses py -3.11
start_frontend.bat         :: web app on http://localhost:5173 (npm install on first run)
```
**Which model answers predictions:** the model named in `artifacts\multilingual\final_selection.json` if its files are
on this machine; otherwise **Multilingual A+**. Telugu-only models are only a last resort and are labelled
"Telugu-only model currently loaded". Every prediction shows the model, all three class probabilities, a heuristic
language guess and a low-confidence warning (< 50%). Predictions use the current message only — they are **not**
context-aware.

## Tests
```bat
run_tests.bat              :: pytest (backend) + frontend type-check and build
```

## Pipeline (backend\pipeline, run with `py -3.11 -m pipeline.<name>` from `backend\`)
| Step | Script | Notes |
|---|---|---|
| data | `prepare_data` | checksums, cleaning, documented label mapping, dedupe, leakage-safe 70/15/15 split |
| A / A+ | `train_model_a`, `run_experiments`, `finalize_model_a_plus` | frozen; refuse to overwrite |
| metrics | `recompute_metrics` | exact F1 etc. from saved predictions |
| error analysis | `error_analysis_a_plus` | A+ test errors (analysis only) |
| classical search | `classical_search` | S0–S6, validation only, `experiments\experiment_leaderboard.csv` |
| Colab package | `build_colab_transformers` | `artifacts\colab_package_phase3\` (staged plan in `colab_runner.PLAN`) |
| import + final evaluation | `finalize_phase3 --import` | ensembles, calibration, McNemar/bootstrap, final tables, `final_selection.json` |
| app weights | `install_phase3_model` | installs downloaded Transformer weights for the web app |
| notebook | `build_status_notebook` | rebuilds and executes `D:\NLP\context_aware_sentiment_analysis.ipynb` |
| report | `write_phase3_report` | `docs\final_report_phase3.md` |

## Transformer training (GPU, Colab)
See [docs/COLAB_PHASE3.md](docs/COLAB_PHASE3.md). In short: open
`artifacts\colab_package_phase3\colab_phase3_transformers.ipynb` in Colab with a T4 GPU, Run all, upload
`transformer_package.zip`, then copy the downloaded `phase3_results_*.zip` to `artifacts\colab_results\` and run
`py -3.11 -m pipeline.finalize_phase3 --import`.

## Layout
```
sentiment_app/
  backend/app/            FastAPI: main.py, model_service.py (model registry + fallback), artifacts.py, schemas.py
  backend/sentiment_core/ labels, cleaning, context (Model C, ready for real data), language (heuristic), evaluation, experiment_io
  backend/pipeline/       data, classical, Colab, finalisation and report scripts (see table above)
  frontend/               React app: Dashboard, Predict, Batch, Comparison, Audit, Errors, Context
  tests/                  pytest suite
  artifacts/multilingual/ data, model_a, model_a_plus, experiments/, final_selection.json, phase3_models/ (installed weights)
  artifacts/colab_package_phase3/  Colab notebook + package;  artifacts/colab_results/  downloaded Colab zips
```
The original Phase-1 notebook is archived unchanged in `D:\NLP\archive\`.
