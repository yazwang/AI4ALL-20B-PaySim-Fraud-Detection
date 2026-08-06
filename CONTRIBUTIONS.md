# Contributions

This is a team-reviewed contribution record. The items below reflect the confirmed work of each team member. Team members are listed alphabetically by last name. The order does not reflect contribution level.

## Abdullahi Ali

- Contributed to the Logistic Regression implementation.

## Arjya Misra

- Owned and maintained the shared project proposal document and contributed to the team proposal.
- Contributed to the Random Forest implementation.

## Emmanuel A. Opoku

- Implemented and integrated the XGBoost fraud-detection model using the project's leakage-safe engineered features.
- Added class-imbalance handling, validation-based threshold selection, and early stopping for XGBoost.
- Produced XGBoost evaluation outputs, including the confusion matrix, precision-recall curve, feature-importance plot, and final comparison metrics.
- Resolved notebook execution and integration issues, restored XGBoost to the final comparison table, and aligned the Key Findings with the saved notebook outputs.
- Pinned the required package versions and documented the macOS `libomp` requirement for running XGBoost.
- Contributed the XGBoost work through PR #2, which was reviewed and merged into `main`.

## Malvee Vasan

- Continued developing the initial Colab workflow created by Esther Wang by organizing, running, and testing the project code.
- Contributed to the stratified sampling and train-validation-test split workflow.
- Organized and consolidated the team's shared Google Drive files and project documents.
- Helped coordinate task assignments and followed up with team members on project progress.
- Owned and managed the shared `PaySim_DS` dataset file.

## Esther Wang

### Project Proposal & Planning

- Served as the primary author of the team project proposal by drafting the initial full version, writing the majority of the substantive content, incorporating available team input, and completing the final revision.
- Wrote the topic summary, project motivation, potential impact section, and research question to define the fraud-detection problem and its relevance to safer mobile-payment systems.
- Developed the PaySim dataset rationale and planned the model-comparison approach using logistic regression, random forest, and gradient boosting or XGBoost.
- Wrote the training and testing considerations, class-imbalance strategy, and feature-engineering plan for transaction amount, balance-change, and account-balance features.
- Selected fraud-detection evaluation metrics, including precision, recall, F1-score, confusion matrix, false-positive rate, and PR-AUC.
- Identified synthetic-data bias, geographic or contextual bias, and class-imbalance or label bias; developed mitigation strategies; and gathered supporting citations and sources.

### Notebook Development & Evaluation

- Designed the initial Colab workflow to load data, structure analysis steps, and support team development.
- Implemented leakage-aware feature-engineering components, including transaction-derived predictors and intentional exclusion of identifier and leakage-prone features.
- Reorganized and cleaned the shared notebook, integrated teammates' separate model sections into a coherent final workflow, and preserved attribution for their original logistic regression and random forest model-training code.
- Built the shared fraud-flagging and evaluation framework, implemented validation-based threshold selection, and evaluated the original `isFlaggedFraud` baseline against the same test set used for model comparison.
- Created the unified model-comparison table for baseline, default-threshold, and validation-selected-threshold results.
- Wrote the data-supported Key Findings from the project evaluation outputs.

### Repository & Final Integration

- Created the initial GitHub repository structure with notebook, documentation, data, app, and source-code folders.
- Prepared project documentation, including README content, data handling notes, repository structure notes, and contribution records.
- Combined, reviewed, and integrated separate team deliverables into the final notebook and submission-ready repository structure.

## Submission Notes

- Do not add unconfirmed work to another team member's section.
- Do not describe assigned work as completed unless completion has been confirmed.
- Keep model-performance claims consistent with the notebook outputs and the development-sample limitation.

## Acknowledgments

The team acknowledges the AI4ALL Ignite Fellowship for supporting the project.
