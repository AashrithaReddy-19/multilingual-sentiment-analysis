> **Superseded (2026-10-12).** Current status and results: `docs/final_report_phase3.md`; Colab training: `docs/COLAB_PHASE3.md`. Kept for history; numbers below may be out of date.

# Training multilingual Model B on Google Colab

Why Colab: this PC has no GPU. Telugu-only Model B alone took about 24 hours on its CPU, and the multilingual training set is about 4.8 times larger.

**Blocker:** Colab and Google Drive need your Google sign-in in a browser. That can't be automated from this machine. Everything else is automatic.

## Steps
1. Go to https://colab.research.google.com → **Upload** → `D:\NLP\sentiment_app\artifacts\colab_package\colab_train_model_b.ipynb`. The notebook is set to open on a T4 GPU.
2. Press **Ctrl+F9** (Run all).
3. Allow Google Drive access.
4. Select `D:\NLP\sentiment_app\artifacts\colab_package\model_b_package.zip` when asked. This happens only on the first run; the file is kept on Drive afterwards.

The notebook then:
1. verifies all 12 package files against `MANIFEST.json`
2. trains XLM-RoBERTa with the exact project settings
3. evaluates on the test split: overall, per class, per language, confusion matrix, confidence and 30 misclassified examples
4. saves everything to `MyDrive/NLP_sentiment/model_b_runs/run_<time>/`
5. downloads `model_b_results_<time>.zip` to your Downloads folder, and tries `model_b_full_<time>.zip`, which includes the 1.1 GB model

Model C is reported as "Skipped — no real conversation_id/message_order available."

## Bring the results back
```bat
cd /d D:\NLP\sentiment_app\backend
python -m pipeline.import_colab_results
```
The importer only accepts a real run (not a smoke test) whose split checksums match `artifacts\multilingual\data`. It never overwrites existing results.

To serve the multilingual model in the app, the full zip (with weights) is needed. If its browser download fails, download it from Drive and run the importer with its path:

```bat
python -m pipeline.import_colab_results C:\path\to\model_b_full_<time>.zip
```
