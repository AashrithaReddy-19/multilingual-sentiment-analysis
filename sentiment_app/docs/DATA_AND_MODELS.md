# Data and model files not stored in Git

The repository contains all code, notebooks, reports, metric files, summaries and the small trained TF-IDF models
(A, A+, Telugu-only A). The items below are **excluded** (see the root `.gitignore`) and must be obtained or rebuilt.

## Raw datasets (not redistributed)
| Dataset | Where to get it | Local folder expected | Licence note |
|---|---|---|---|
| TeSent (Telugu) | Hugging Face `DSL-13-SRMAP/TeSent_Benchmark-Dataset` | `D:\NLP\multilingual_inputs\telugu_tesent\` (`Tesent.csv`, `README.md`) | CC BY 4.0 |
| TweetEval sentiment (English) | https://github.com/cardiffnlp/tweeteval (`datasets/sentiment`, commit `4fbd22c`) | `D:\NLP\multilingual_inputs\english_tweeteval\` | Twitter/SemEval-2017 terms — do not redistribute tweet text |
| CMTET (Telugu-English code-mixed) | https://github.com/ksubbu199/cmtet-sentiment (commit `ae383ee`) | `D:\NLP\multilingual_inputs\code_mixed_cmtet\` | no licence stated — research use with citation |

Exact file checksums are recorded in `sentiment_app/artifacts/multilingual/data/provenance.json` and checked by the
pipeline. The hate-speech files (`Tel_Eng_dataset`) and the fairness dataset were never used for sentiment training.

## Rebuild the data split and Colab package
```bat
cd /d D:\NLP\sentiment_app\backend
py -3.11 -m pipeline.prepare_data              :: verifies checksums, rebuilds train/validation/test (seed 42)
py -3.11 -m pipeline.build_colab_transformers  :: rebuilds transformer_package.zip
```
The rebuilt split must reproduce the SHA-256 values in the package `MANIFEST.json`
(train `e0835f56…`, validation `427fe31b…`, test `935b3185…`); results are only imported for that exact split.

## Excluded derived / large files
- Split CSVs, `cleaned_all_data.csv`, and per-example prediction / misclassification / ambiguity CSVs: they contain
  dataset text (or are large, ~200 MB of per-experiment validation predictions). Metrics, confusion matrices,
  classification reports and summary tables are kept.
- Transformer weights (`*.safetensors`, ~1.1 GB each): Telugu-only XLM-R in `D:\NLP\outputs\xlm_roberta_no_context\`,
  Phase-3 models in `sentiment_app\artifacts\multilingual\phase3_models\` (installed with
  `py -3.11 -m pipeline.install_phase3_model` from the Colab download). Without them the app serves Multilingual A+.
- `backups\` (local zip snapshots), Colab package/result zips, caches, `node_modules`, build output, `.env`.
