# Per-model improvement round (CPU only) — pipeline.improve_all

Selection on validation only (5-fold CV inside validation, parameters refitted per fold; pass = mean gain >= 0.003
macro F1 and better in >= 4/5 folds). Only frozen finalists were scored on test, once.

| Study | Result |
|---|---|
| Inventory (inventory.csv) | test acc / macro F1: A .6635/.6637, A+ .6994/.7001, E5 (earlier) .7438/.7453, E5 .7445/.7458, Previous prod .7483/.7499, Optimized .7510/.7524, Production lang_aware_v2 .7537/.7553; Telugu A .6285/.6232, A+ .6597/.6558, B .6979/.6985 |
| Class bias per model (postproc.json) | no pass for A, A+, E5, E5-new, Telugu A+, Telugu B; Telugu A +0.004 (4/5) but still below Telugu A+ -> pruned. Temperature improves calibration only (argmax unchanged) |
| Dual-E5 (dual_e5.json) | 94.4% agreement, oracle .7652; best blend +0.0005 CV -> no |
| Production refinement (prod_refine.json) | finer weights, language x class bias (any shrinkage), Neutral multiplier, adding A: all <= 0 CV gain -> production unchanged |
| Telugu leakage (telugu_leakage.json) | ~70% of Telugu-only val/test rows are in the multilingual TRAIN split -> multilingual models must not be evaluated/mixed on the Telugu-only split |
| New Telugu-only models (telugu_models.json) | 43 configs; best new architecture: char TF-IDF + frozen e5-small embeddings + LR, val .6811 |
| Telugu test (artifacts/telugu_only_improve/test_result.json) | Telugu Hybrid acc .6840 / F1 .6835 (vs A+ .6558, McNemar p=.0004); Telugu Ensemble (B + char TF-IDF) acc .7011 / F1 .7015 (vs B .6985, p=.49, not significant) |

75% target: met by Production, Optimized and Language-Aware (v1) ensembles. Single multilingual E5 is 0.4-0.6 points
short; TF-IDF and Telugu-only models are 5-9 points short and no CPU method found here closes that gap.
