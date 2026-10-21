# Phase 3 – Transformer study on Colab (GPU only)

Full Transformer training is GPU-only (`train_transformer.py` refuses to run on CPU except for smoke tests).

## 1. Files
- Notebook: `D:\NLP\sentiment_app\artifacts\colab_package_phase3\colab_phase3_transformers.ipynb`
- Package (upload when asked): `D:\NLP\sentiment_app\artifacts\colab_package_phase3\transformer_package.zip`
  (the exact local train/validation/test CSVs + code, checked against `MANIFEST.json`; `VERSION.txt` = phase3-v5)
- If an older package is on Drive, delete `MyDrive/NLP_sentiment/transformer_package.zip` so the new one is uploaded.
- Google Drive is optional. If Drive authorization fails, the notebook uses `/content/NLP_sentiment` and downloads
  results directly with `files.download()`.

## 2. Run
1. https://colab.research.google.com → File → Upload notebook → the notebook above.
2. Runtime → Change runtime type → **GPU** (L4 or A100 preferred; T4 works but is ~2–3× slower) → Save.
3. Runtime → **Run all**. Allow Google Drive access; choose `transformer_package.zip` when asked. The first cell prints
   the GPU name, VRAM, CUDA version and `torch.cuda.is_available()`.
4. `PLAN = colab_runner.PLAN` runs, in order (each step = one family + tier, see the table in the notebook):
   E5 core → XLM-R core → MuRIL core → IndicBERTv2 core → E5 extra → XLM-R refine → MuRIL/IndicBERTv2 refine →
   mDeBERTa core → XLM-R extra. Finished runs are skipped, so every *Run all* continues where it stopped; use e.g.
   `PLAN = colab_runner.PLAN[:2]` for a shorter session.
5. Inside a family every stage changes ONE factor of its current best run (validation macro F1, then macro recall, then
   accuracy): learning rate → epochs (4 → 5 → 6 → 8 only while still improving; early stopping patience 2, 3 from 6
   epochs) → max length → loss (CE / weighted CE / focal γ 1.0–2.5) → class-weight scheme (inverse, sqrt-inverse,
   effective number) → label smoothing (0.03/0.05/0.10) → language tokens → language-reweighted training sampler
   (sqrt / balanced; validation and test are never resampled) → text variants → weight decay / warmup / cosine →
   layer freezing (3/6 permanent, 6 gradual) → batch 32. MuRIL / IndicBERTv2 / mDeBERTa start from the best XLM-R
   recipe and search their own learning rate and epochs. Per-epoch validation metrics (incl. per-language accuracy and
   macro F1) are in `validation_history.csv`; the best epoch's weights are kept, not the last.
6. Runs with language tokens also save validation/test predictions where the token comes from the app's heuristic
   language detector (`detected_lang/`), because the web app has no dataset metadata. Both numbers are reported.
7. *Final test* scores only the best validation run of each model family on the test split.
8. The next cell downloads **`phase3_results_<time>.zip`** (metrics, predictions, plots — no weights).
9. The last cell downloads `phase3_model_<family>_<time>.zip` for every family whose best validation macro F1 beats
   A+ (0.6962) — chosen on validation only (`DOWNLOAD_MODELS = None`), or set a list / `[]`.

## 3. Bring results back
Copy the zip(s) to **`D:\NLP\sentiment_app\artifacts\colab_results\`** (or leave them in Downloads), then:
```
cd D:\NLP\sentiment_app\backend
py -3.11 -m pipeline.finalize_phase3 --import      :: import (real runs on this split only), validation leaderboard, pair/triple + language-aware
                                                   :: ensembles (validation), calibration, final test, significance, selection
py -3.11 -m pipeline.install_phase3_model          :: only if model zips were downloaded; checks + CPU load test
py -3.11 -m pipeline.write_phase3_report           :: docs\final_report_phase3.md
py -3.11 -m pipeline.build_status_notebook         :: refresh D:\NLP\context_aware_sentiment_analysis.ipynb
```
Restart the backend afterwards. If the selected model's weights are not installed, the app keeps serving A+ and says so.
