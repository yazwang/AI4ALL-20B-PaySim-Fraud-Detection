# Key Findings and Next Steps

These findings are based on the saved outputs from the real PaySim development sample run. The test set had 15,000 transactions, including 19 fraud cases.

1. Random Forest performed best on this development sample. At both the default threshold of 0.50 and the validation-selected threshold of 0.30, it found all 19 fraud cases and made no false positive fraud flags. Its precision, recall, F1, and PR-AUC / Average Precision were all 1.000000, with a false-positive rate of 0.000000. It flagged 19 transactions, or 0.126667% of the test set.

2. Logistic Regression found some fraud, but missed 8 of the 19 fraud cases at both thresholds. At the default 0.50 threshold, it had precision 0.916667, recall 0.578947, F1 0.709677, and PR-AUC / Average Precision 0.671791. It flagged 12 transactions, or 0.080000% of the test set, with a very low false-positive rate of 0.000067.

3. Lowering Logistic Regression to the validation-selected threshold of 0.15 did not improve recall on the test set. Recall stayed at 0.578947, but precision dropped from 0.916667 to 0.846154 and F1 dropped from 0.709677 to 0.687500. It flagged one extra transaction, 13 instead of 12, raising the flagged percentage from 0.080000% to 0.086667% and the false-positive rate from 0.000067 to 0.000134.

4. The original isFlaggedFraud baseline did not catch any fraud cases in this test set. It flagged 0 transactions, so its precision, recall, and F1 were all 0.000000. Its false-positive rate was also 0.000000, but only because it made no fraud flags at all.

5. The validation-selected thresholds did not change the Random Forest test results, but they did make Logistic Regression slightly less precise. For this sample, Random Forest was the strongest model because it had perfect fraud recall without increasing false positives, while Logistic Regression traded a small number of extra flags for no additional fraud cases found.

XGBoost is a boosted-tree extension under review. It is excluded from the final metrics above until it runs reproducibly on the same real-PaySim evaluation pipeline.

These results are promising but are based on only 19 fraud cases in the test set; they do not prove real-world perfect performance.
