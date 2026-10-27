# Data-centric improvement study (training data only)

validation.csv and test.csv were not modified; test.csv was not read. Every variant trains the SAME A+ pipeline (word (1,2) 20k + char_wb (3,5) TF-IDF, balanced logistic regression) and is scored on validation.

## Training-data audit

- train_rows: 70540
- exact_dup: 77
- norm_dup: 304
- conflict_groups: 60
- conflict_group: 141
- near_dup_conflict: 28
- cross_language_dup: 0
- empty_after_cleaning: 0
- url_only: 0
- mention_or_hashtag_only: 1
- punct_only: 0
- emoji_only: 0
- very_short_<=2_words: 36
- encoding_corruption: 16
- neutral_with_strong_pos: 1067
- neutral_with_strong_neg: 312
- neutral_mixed_cues: 63
- negative_with_pos_only_cues: 642
- positive_with_neg_only_cues: 271

Conflict groups by deterministic policy class: {'single disagreeing pair (ambiguous)': 43, 'repeated text with split labels (likely noise or ambiguity)': 17}

## Train-CV quality categories (5-fold out-of-fold A+ probabilities + char LinearSVC)

{'clean_consensus': 34228, 'normal': 18993, 'hard_legitimate': 12814, 'potentially_noisy': 4505}

Potential label noise / ambiguity (rows both OOF models get wrong with A+ confidence >= 0.7, or in a conflict group). Disagreement is NOT proof of a wrong label:

- overall: 6.39%
- lang_english: 6.72%
- lang_telugu: 6.39%
- lang_code_mixed: 5.37%
- class_Negative: 5.18%
- class_Neutral: 7.29%
- class_Positive: 6.18%

## Training variants (validation)

| variant                                   |   train_rows |   val_accuracy |   val_macro_f1 |   gain_vs_original |   Neutral_f1 |   english_macro_f1 |   telugu_macro_f1 |   code_mixed_macro_f1 |
|:------------------------------------------|-------------:|---------------:|---------------:|-------------------:|-------------:|-------------------:|------------------:|----------------------:|
| A original                                |        70540 |         0.6958 |         0.6962 |             0      |       0.6814 |             0.6614 |            0.6552 |                0.768  |
| B same-label duplicates collapsed         |        70540 |         0.6958 |         0.6962 |             0      |       0.6814 |             0.6614 |            0.6552 |                0.768  |
| C B + conflict groups removed             |        70399 |         0.696  |         0.6965 |             0.0003 |       0.6802 |             0.6616 |            0.6577 |                0.7675 |
| D B + conflict groups weight 0.5          |        70540 |         0.6959 |         0.6964 |             0.0002 |       0.6808 |             0.6614 |            0.6567 |                0.7679 |
| E quality weights noisy 0.5 / hard 1.1    |        70540 |         0.6938 |         0.6942 |            -0.002  |       0.6801 |             0.6593 |            0.651  |                0.7701 |
| E quality weights noisy 0.25 / hard 1.0   |        70540 |         0.6939 |         0.6942 |            -0.002  |       0.681  |             0.6594 |            0.6475 |                0.7741 |
| E quality weights noisy 0.75 / hard 1.2   |        70540 |         0.6949 |         0.6955 |            -0.0007 |       0.6794 |             0.6625 |            0.6522 |                0.7686 |
| F potentially-noisy rows removed          |        66035 |         0.6936 |         0.6937 |            -0.0025 |       0.6806 |             0.6582 |            0.6494 |                0.7711 |
| G Neutral weight x1.05                    |        70540 |         0.6969 |         0.6975 |             0.0013 |       0.6817 |             0.663  |            0.6569 |                0.7679 |
| G Neutral weight x1.1                     |        70540 |         0.6959 |         0.6963 |             0.0001 |       0.6814 |             0.6617 |            0.6546 |                0.7678 |
| G Neutral weight x1.2                     |        70540 |         0.6954 |         0.6959 |            -0.0003 |       0.6803 |             0.6615 |            0.6552 |                0.7669 |
| H sqrt language balancing                 |        70540 |         0.6963 |         0.6961 |            -0.0001 |       0.6849 |             0.6588 |            0.6543 |                0.7712 |
| H Telugu weight x1.2                      |        70540 |         0.6961 |         0.6966 |             0.0004 |       0.681  |             0.6616 |            0.6553 |                0.7696 |
| H Telugu weight x1.3                      |        70540 |         0.6963 |         0.6968 |             0.0006 |       0.68   |             0.6631 |            0.6545 |                0.7674 |
| I matrix EN-neu 1.1 TE-neu 1.1 TE-pol 1.0 |        70540 |         0.6965 |         0.697  |             0.0007 |       0.682  |             0.6632 |            0.6541 |                0.7675 |
| I matrix EN-neu 1.0 TE-neu 1.2 TE-pol 1.1 |        70540 |         0.6956 |         0.6959 |            -0.0003 |       0.6808 |             0.6622 |            0.6511 |                0.7688 |
| I matrix EN-neu 1.1 TE-neu 1.2 TE-pol 1.2 |        70540 |         0.6965 |         0.6968 |             0.0006 |       0.6818 |             0.6609 |            0.6562 |                0.7702 |
| J Neutral format augmentation             |        99411 |         0.6892 |         0.6898 |            -0.0064 |       0.6652 |             0.6558 |            0.6471 |                0.7659 |
| K Telugu Unicode normalisation            |        70540 |         0.6958 |         0.696  |            -0.0002 |       0.6806 |             0.6615 |            0.6558 |                0.7663 |

## Learning curve (A+, stratified by language x label)

|   train_fraction |   rows |   val_macro_f1 |   val_accuracy |   english_macro_f1 |   telugu_macro_f1 |   code_mixed_macro_f1 |
|-----------------:|-------:|---------------:|---------------:|-------------------:|------------------:|----------------------:|
|             0.25 |  17635 |         0.6568 |         0.6583 |             0.611  |            0.6063 |                0.7513 |
|             0.5  |  35270 |         0.6748 |         0.6755 |             0.6288 |            0.6359 |                0.7715 |
|             0.75 |  52905 |         0.6876 |         0.6873 |             0.6503 |            0.6463 |                0.7652 |
|             1    |  70540 |         0.6962 |         0.6958 |             0.6614 |            0.6552 |                0.768  |
