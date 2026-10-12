# Phase 3 – Transformer study on Colab (GPU only)

Full Transformer training is GPU-only (`train_transformer.py` refuses to run on CPU except for smoke tests).

## 1. Files
- Notebook: `D:\NLP\sentiment_app\artifacts\colab_package_phase3\colab_phase3_transformers.ipynb`
- Package (upload when asked): `D:\NLP\sentiment_app\artifacts\colab_package_phase3\transformer_package.zip`
  (the exact local train/validation/test CSVs + code, checked against `MANIFEST.json`; `VERSION.txt` = phase3-v3)
- If you uploaded an older package before, delete `MyDrive/NLP_sentiment/transformer_package.zip` so the new one is used.

## 2. Run
1. https://colab.research.google.com → File → Upload notebook → the notebook above.
2. Runtime → Change runtime type → **T4 GPU** → Save.
3. Runtime → **Run all**. Allow Google Drive access; choose `transformer_package.zip` when asked.
4. Default `STAGES = colab_runner.TIER1` (~12 runs, ~6 h on a T4), run in this order: learning rate
   {1e-5, 2e-5, 3e-5} → epochs (3, and 8 with patience 3; 10 only if epoch 8 was still improving) → loss (weighted CE
   from train frequencies, focal γ 1.5/2.0 + 1.0 or 2.5) → language tokens → MuRIL.
   Later sessions: `STAGES = colab_runner.TIER1 + colab_runner.TIER2` (lr 1.5e-5/2.5e-5, label smoothing 0.05/0.1,
   IndicBERTv2), then `+ colab_runner.TIER3` (batch 32/8, weight decay 0/0.02, warmup 0.05, mDeBERTa-v3 in fp32,
   lower-layer freezing, max_len 192). Finished runs are skipped, so every *Run all* continues where it stopped.
5. Each stage changes one factor of the current best XLM-R recipe; a change is kept only if validation macro F1
   (then macro recall, then accuracy) improves. Per-epoch validation loss, accuracy, macro P/R/F1, weighted F1, per-class
   P/R/F1 and training loss are saved in `validation_history.csv`; `config.json` flags overfitting
   (validation macro F1 falling while training loss falls).
6. *Final test* scores only the best validation run of each model family on the test split.
7. The next cell downloads **`phase3_results_<time>.zip`** (metrics, predictions, plots — no weights).
8. Only if a Transformer or the ensemble is later selected for the web app: set `DOWNLOAD_MODELS = ["xlmr"]`
   (add e.g. `"muril"` for an ensemble member) and run the last cell to download `phase3_model_<family>_<time>.zip` (~1.1 GB each).

## 3. Bring results back
Copy the zip(s) to **`D:\NLP\sentiment_app\artifacts\colab_results\`** (or leave them in Downloads), then:
```
cd D:\NLP\sentiment_app\backend
py -3.11 -m pipeline.finalize_phase3 --import      :: import (real runs on this split only), ensembles, final test tables, selection
py -3.11 -m pipeline.install_phase3_model          :: only if model zips were downloaded; checks + CPU load test
py -3.11 -m pipeline.write_phase3_report           :: docs\final_report_phase3.md
py -3.11 -m pipeline.build_status_notebook         :: refresh D:\NLP\context_aware_sentiment_analysis.ipynb
```
Restart the backend afterwards. If the selected model's weights are not installed, the app keeps serving A+ and says so.
