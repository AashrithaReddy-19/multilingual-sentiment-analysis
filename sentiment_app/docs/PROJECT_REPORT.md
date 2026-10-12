> **Superseded (2026-10-12).** Current status and results: `docs/final_report_phase3.md`; Colab training: `docs/COLAB_PHASE3.md`. Kept for history; numbers below may be out of date.

# Project report: Context-Aware Sentiment Analysis for Telugu, English, and Telugu-English Code-Mixed Conversational Chat

## 1. Goal
Classify chat messages as **Negative (0), Neutral (1) or Positive (2)** in three settings:
- Telugu script
- English
- Telugu-English code-mixed text, including Romanized Telugu

A context-aware variant would read the two previous messages of the same conversation.

## 2. Datasets and verified label mappings
| Dataset | Language | Source (pinned) | Rows used* | Documented mapping | Licence / use |
|---|---|---|---|---|---|
| TeSent | Telugu | user's copy of DSL-13-SRMAP/TeSent_Benchmark-Dataset | 21,118 | labels already `Negative/Neutral/Positive` (README) | CC BY 4.0 |
| TweetEval sentiment | English | github.com/cardiffnlp/tweeteval @ 4fbd22c | 59,873 | `mapping.txt`: 0 = negative, 1 = neutral, 2 = positive | "released without any restrictions"; SemEval-2017 terms and Twitter/X rules apply |
| CMTET | Telugu-English code-mixed | github.com/ksubbu199/cmtet-sentiment @ ae383ee | 19,794 | README tags `NEG/NTL/POS`; classes defined in Kusampudi et al., RANLP 2021, Sec. 3.3 | no licence stated: research use with citation |

\*After cleaning and duplicate removal.

Every input file is checked against its audited SHA-256 checksum before use. No other mapping is accepted; unknown labels raise an error.

**Excluded during the audit:**
- The originally supplied English file (`sentiment_data.csv`): its numeric labels are undocumented.
- The Telugu-English `.xlsx` files: their labels are hate/non-hate, not sentiment.

## 3. Data pipeline (`backend/pipeline/prepare_data.py`)
1. **Cleaning** removes only URLs, HTML tags and entities, control characters and extra whitespace.
   - It keeps Telugu script (including zero-width joiners), Romanized Telugu, English, emojis, `!`, `?` and negation words (not, no, never, ledu, kaadu, avvale, vaddu).
   - It does no lowercasing, stemming or stop-word removal.
2. **Rows removed:** missing text or label (none were found), and **101 exact duplicate text-label pairs** (TeSent 1, TweetEval 26, CMTET 74).
3. **Conflicting labels:** 55 texts (129 rows) carry more than one label. They are **kept**, because the disagreement is genuine annotation ambiguity, and **grouped**: each identical cleaned text is one split group, so it can never appear in two splits.
4. **Schema:** `raw_text, cleaned_text, label, language_type, conversation_id, message_order, context_text, source_dataset`, plus `source_label` (the original label) and provenance columns.
5. **Context:** `context_text = cleaned_text` for every row, because no dataset has conversation metadata.
6. **Split:** 70/15/15, reproducible (seed 42), stratified jointly by label × language_type, at the group level.
   - Sizes: train 70,540 · validation 15,121 · test 15,124.
   - Leakage check: zero cleaned-text overlap between any two splits.
7. **Saved files** (`artifacts/multilingual/data/`): `cleaned_all_data.csv`, the three split files, `final_audit.json`, `cleaning_report.json`, `label_mapping.json`, `class_distributions.csv`, `provenance.json` and `leakage_checks.json`.

## 4. Models
- **A: TF-IDF + Logistic Regression.** A fast, transparent baseline. Word 1–2-grams with max_features 20,000 feed a logistic regression (`class_weight="balanced"`, `max_iter=1000`, `random_state=42`). The tokenizer splits on whitespace (`\S+`) so Telugu words and emojis stay intact; scikit-learn's default pattern would break Telugu words apart and drop emojis.
- **B: XLM-RoBERTa (`xlm-roberta-base`) without context.** A multilingual Transformer pretrained on about 100 languages, including Telugu, which suits Telugu script, English and Romanized or mixed text.
  - **Settings:** max_length 128 · 5 epochs · learning rate 2e-5 · weight decay 0.01 · warmup ratio 0.1.
  - **Training procedure:** evaluated and checkpointed every epoch, with the best checkpoint chosen by validation macro F1. Early stopping patience is 2, fp16 is used only on CUDA, and the seed is 42.
- **C: Context-aware XLM-RoBERTa.** **Skipped — no real conversation_id/message_order available.** Without conversations, `context_text` equals `cleaned_text` for every row, so Model C would have the same input and settings as Model B: a duplicate experiment.

What the parameters mean:
- `max_length`: the maximum number of tokens read per input.
- `epochs`: the maximum number of full passes over the training data.
- A small `learning_rate` (2e-5) suits fine-tuning.
- `weight_decay` reduces overfitting.
- `warmup` ramps the learning rate up during the first 10% of steps.
- Early stopping ends training after two evaluations without improvement.
- `class_weight="balanced"` counteracts unequal class sizes in Model A.
- **Macro F1** weights the three classes equally, **weighted F1** follows class frequency, and **accuracy** is the share of correct predictions.

## 5. Real results
All numbers come from saved metric files. Full tables, per-class scores and confusion matrices are in [final_report.md](final_report.md).

**Telugu-only run** (TeSent test set, 3,168 sentences; completed earlier on CPU):

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| A: TF-IDF + LogReg | 0.6285 | 0.6232 | 0.6287 |
| B: XLM-RoBERTa | 0.6979 | 0.6985 | 0.6979 |
| C | Skipped — no real conversation_id/message_order available. | | |

**Multilingual run** (test set of 15,124 rows):

| Model | Accuracy | Macro F1 | Weighted F1 | Telugu F1 | English F1 | Code-mixed F1 |
|---|---|---|---|---|---|---|
| A: TF-IDF + LogReg (original baseline) | 0.6635 | 0.6637 | 0.6639 | 0.6008 | 0.6281 | 0.7660 |
| A+: word + char TF-IDF + LogReg | 0.6994 | 0.7001 | 0.6996 | 0.6710 | 0.6617 | 0.7774 |
| B: XLM-RoBERTa | Not yet trained — GPU Colab run required. | | | | | |
| C | Skipped — no real conversation_id/message_order available. | | | | | |

The two runs use different data and splits, so they shouldn't be compared directly.

On the Telugu-only data, XLM-RoBERTa beat TF-IDF on every class. Its largest gain was Positive recall, and it confused Negative with Positive about half as often.

**Heuristic error reasons:** each misclassified example is tagged by simple rules (negation words, emojis, mixed script, low confidence, annotator disagreement). These tags are guesses to guide inspection, **not ground truth**.

## 5b. CPU experiments and Model A+
**Error analysis** (validation split, original Model A; the test split was not used):
- **Confusions involving Neutral** cause 85% of errors. Positive→Neutral is 25% of errors, Neutral→Negative 22%, Neutral→Positive 21%, Negative→Neutral 17%.
- **Telugu-script text is hardest:** a 39.5% error rate, against 21.6% for code-mixed. Word TF-IDF treats each inflected Telugu word as unrelated to the others.
- **Text with negations** has a 41.9% error rate, against 33.1% elsewhere.
- **Romanized and code-mixed text** is easiest (about 20% errors).
- **Short texts** (1.1% of rows) and **label-conflict texts** (0.2%) are too rare to matter.
- **Saved to** `artifacts/multilingual/experiments/error_analysis_model_a/`.

**79 variants** were trained on `train.csv` and scored on `validation.csv` only:
- 72 word TF-IDF + Logistic Regression variants (n-grams 1–2/1–3, sublinear on/off, min_df 1/2/3, 20k/50k features, C 0.5/1/2)
- 2 character (`char_wb` 3–5, 3–6) + LinearSVC
- 2 word+char + LinearSVC
- 3 word+char + Logistic Regression

All use `class_weight="balanced"`. The character vectorizers use `min_df=2` to bound memory.

**Selection and result:**
- **Rule:** highest validation macro F1, then validation accuracy.
- **Winner:** word (1,2, sublinear, 20k) + `char_wb` (3,5) TF-IDF with Logistic Regression C=1.0. Validation macro F1 was 0.6962, against 0.6609 for the baseline.
- **Test:** the winner was refit, its validation score was verified, and it was evaluated **once** on the untouched test split. Test macro F1 was 0.7001, against 0.6637 for the original (+0.0364). Telugu improved the most (+0.0702).
- **Saved as Model A+** in `artifacts/multilingual/model_a_plus/`. The original Model A is unchanged and shown next to it in the app.
- **Full table:** `artifacts/multilingual/experiments/experiment_comparison.csv`. It holds validation results for every candidate, plus test results for the original baseline and the selected model only.

Note: the 72 word variants ran in parallel worker processes. Floating-point differences there can shift validation macro F1 by about 0.0005; for example, the baseline config scored 0.6614 in a worker and 0.6609 single-process. The selected model ran single-process and was re-verified before testing.

## 6. Limitations
- **Multilingual Model B has not been trained.** Doing it on this CPU would take days. The Colab package is ready, but running it needs a Google sign-in, which can't be automated here.
- **There is no conversational data,** so the "context-aware" part of the title isn't realised yet. Model C is skipped rather than faked.
- **TweetEval is 59% of the multilingual data.** Overall scores lean toward English, so the per-language columns are the fairer view.
- **Some TweetEval tweets contain literal `’`-style escape sequences** from the source files. They were kept, because cleaning is deliberately limited to URLs, HTML, control characters and whitespace.
- **Each result is a single run with one seed.** No confidence intervals or significance tests were computed.
- **CMTET has no stated licence,** so redistribution terms are unclear.
- **The Telugu-only XLM-R that the app currently serves** was trained on Telugu script only. The UI labels it "Telugu-only model currently loaded"; its outputs for English or code-mixed text are not multilingual results.

## 7. Next steps
1. Run `colab_train_model_b.ipynb` on a GPU, then `python -m pipeline.import_colab_results`. The app then serves the multilingual model automatically.
2. Collect and annotate real conversations (`conversation_id, message_order, text, label, language_type`) and train Model C with the existing context builder.
3. Run several seeds and report the variance. Consider balancing languages, for example by capping TweetEval.
