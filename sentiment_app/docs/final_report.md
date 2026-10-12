> **Superseded (2026-10-12).** Current status and results: `docs/final_report_phase3.md`; Colab training: `docs/COLAB_PHASE3.md`. Kept for history; numbers below may be out of date.

# Final report: real results

_Generated 2026-10-03 11:12 from saved metric files._

Only models that were actually trained and evaluated appear with numbers.

## 1. Multilingual run (TeSent + TweetEval + CMTET)

Split: train 70540, validation 15121, test 15124 (cleaned-text leakage across splits: none).

| Model | Input | Accuracy | Macro F1 | Weighted F1 | Telugu F1 | English F1 | Code-mixed F1 | Best val. F1 | Status |
|---|---|---|---|---|---|---|---|---|---|
| A: TF-IDF + Logistic Regression (original baseline) | cleaned_text | 0.6635 | 0.6637 | 0.6639 | 0.6008 | 0.6281 | 0.7660 | 0.6609 | trained |
| A+: word + char TF-IDF + Logistic Regression | cleaned_text | 0.6994 | 0.7001 | 0.6996 | 0.6710 | 0.6617 | 0.7774 | 0.6962 | trained |
| B: XLM-RoBERTa (no context) | cleaned_text | — | — | — | — | — | — | — | Not yet trained — GPU Colab run required. |
| C: XLM-RoBERTa (context-aware) | context_text | — | — | — | — | — | — | — | Skipped — no real conversation_id/message_order available. |

Model A+ was chosen from 79 CPU experiment variants by validation macro F1 (then validation accuracy); the test split was used once, after selection. The original Model A baseline is unchanged. Details: artifacts/multilingual/experiments/experiment_comparison.csv.

### Model A (multilingual, original baseline): per-class (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.6224 | 0.6784 | 0.6492 | 3741 |
| Neutral | 0.6488 | 0.6426 | 0.6457 | 6190 |
| Positive | 0.7160 | 0.6776 | 0.6963 | 5193 |

Confusion matrix (rows = actual, columns = predicted Negative / Neutral / Positive):

| Actual | Negative | Neutral | Positive |
|---|---|---|---|
| Negative | 2538 | 904 | 299 |
| Neutral | 1115 | 3978 | 1097 |
| Positive | 425 | 1249 | 3519 |

Mean confidence 0.6368 (correct 0.6783, wrong 0.5549). Misclassified: 5089 (30 saved with heuristic reasons, which are not ground truth).

### Model A+ (multilingual): per-class (test)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | 0.6792 | 0.7068 | 0.6927 | 3741 |
| Neutral | 0.6800 | 0.6850 | 0.6825 | 6190 |
| Positive | 0.7394 | 0.7113 | 0.7251 | 5193 |

Confusion matrix (rows = actual, columns = predicted Negative / Neutral / Positive):

| Actual | Negative | Neutral | Positive |
|---|---|---|---|
| Negative | 2644 | 822 | 275 |
| Neutral | 923 | 4240 | 1027 |
| Positive | 326 | 1173 | 3694 |

Mean confidence 0.6912 (correct 0.7324, wrong 0.5954). Misclassified: 4546 (30 saved with heuristic reasons, which are not ground truth).

## 2. Telugu-only run (TeSent only, completed earlier on CPU)

| Model | Accuracy | Macro F1 | Weighted F1 | Best val. F1 |
|---|---|---|---|---|
| A: TF-IDF + Logistic Regression | 0.6285 | 0.6232 | 0.6287 | 0.6077 |
| B: XLM-RoBERTa (no context) | 0.6979 | 0.6985 | 0.6979 | 0.7126 |
| C: XLM-RoBERTa (context-aware) | — | — | — | Skipped — no real conversation_id/message_order available. |

The two runs use different splits and data, so their numbers are not directly comparable.

## 3. Model C

Skipped — no real conversation_id/message_order available. None of the datasets contain conversation_id/message_order, so context_text equals cleaned_text for every row; training Model C would duplicate Model B.
