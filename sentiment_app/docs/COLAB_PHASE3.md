# Phase 3 – Transformer study on a GPU (Colab)

`train_transformer.py` refuses full runs without CUDA unless `--allow-cpu` is given (the E5 winner was trained that way,
~6 h on this PC). XLM-R, MuRIL, IndicBERTv2 and mDeBERTa are 2–6× larger and need a GPU.

## 1. Files (package phase3-v6)
- **VS Code route (recommended):** `D:\NLP\sentiment_app\artifacts\colab_package_phase3\colab_vscode_phase3.ipynb` —
  self-contained (the frozen split + code are embedded and SHA-256 checked; the file is gitignored because it contains
  the datasets).
- Browser route: `colab_phase3_transformers.ipynb` + `transformer_package.zip` (upload when asked).
- If an older package is on Drive, delete `MyDrive/NLP_sentiment/transformer_package.zip` so the new one is used.

## 2. Run (VS Code)
1. Open `colab_vscode_phase3.ipynb` in VS Code → **Select Kernel** → **Colab** → **New Colab Server** → **GPU**
   (A100 or L4 preferred; T4 works, ~2–3× slower) → sign in to Google once.
2. **Run All**. Allow Google Drive (finished runs and best weights survive disconnects; Run All again resumes).
   The first cell prints `torch.cuda.is_available()` and the GPU name and stops if there is no GPU.
3. `PLAN` runs, in order: XLM-R core → MuRIL core → IndicBERTv2 core → XLM-R refine → MuRIL / IndicBERTv2 refine →
   mDeBERTa core → XLM-R extra. **E5 is not re-run**: its winner `e5_lr5e-05_ce_ep3p2_bs32_len128_cpu` is already
   imported (`colab_runner.PLAN_WITH_E5` restores a wider E5 search).
4. Inside a family every stage changes ONE factor of its current best run (validation macro F1, then macro recall, then
   accuracy): learning rate (XLM-R 1e-5/2e-5/3e-5 then 1.5e-5/2.5e-5; others 2e-5/3e-5, 3 epochs) → epochs 4 → 5 → 6 → 8
   only while still improving (early stopping on validation macro F1) → loss (CE, weighted CE, focal γ 1.0/1.5/2.0/2.5)
   → class weights from TRAINING counts (inverse, sqrt-inverse, effective number) → label smoothing → language tokens →
   language-reweighted training sampler (sqrt / balanced; validation and test are never resampled) → text variants
   (repeat normalisation; letters only, punctuation untouched; + emoticon tokens; emoticon tokens only) → weight decay /
   warmup / cosine → layer freezing → batch 32. MuRIL / IndicBERTv2 / mDeBERTa start from the best XLM-R recipe and
   search their own learning rate and epochs (then loss / language tokens / sampler for MuRIL and IndicBERTv2).
5. Per-epoch validation history has overall and per-class metrics; per-language metrics are computed once for the best
   epoch from row-aligned predictions (`validation_metrics.json`).
6. Runs with language tokens also save predictions with the token from the app's heuristic detector (`detected_lang/`,
   "as served"). Both numbers are reported.
7. *Final test* scores only the best validation run of each family, once. It then writes
   `MyDrive/NLP_sentiment/phase3_model_<family>.zip` for each family winner.

Every step's results are saved into the notebook's outputs, so VS Code keeps them in the local `.ipynb`.

## 3. Bring results back
```
cd D:\NLP\sentiment_app\backend
py -3.11 -m pipeline.import_vscode_results         :: notebook outputs -> artifacts\colab_results\phase3_results_vscode_*.zip
py -3.11 -m pipeline.finalize_phase3 --import      :: validation leaderboard, pair -> triple and language-aware ensembles
                                                   :: (validation only), calibration, test of the finalists, significance,
                                                   :: selection
```
Selection: best validation macro F1 → simplicity gate (a more complex model needs ≥ 0.005 validation macro F1; cost = one
point per Transformer, +0.5 ensembling, +0.5 language routing) → it must also be **significantly better on test than
the current production model** (`experiments/production_incumbent/`: A+ 0.50 / E5 0.50; McNemar p < 0.05 and bootstrap
95% CI of the macro F1 difference > 0). Otherwise production stays unchanged.

If a new model is selected, `final_selection.json` lists the families it needs (`requires_transformer_weights`). Download
those `phase3_model_<family>.zip` files from Drive into `artifacts\colab_results\`, then:
```
py -3.11 -m pipeline.install_phase3_model          :: checks split / smoke / selected run, CPU load test
py -3.11 -m pipeline.write_phase3_report
py -3.11 -m pipeline.build_status_notebook
```
Restart the backend. If the selected model's weights are missing, the app serves A+ and says so.
