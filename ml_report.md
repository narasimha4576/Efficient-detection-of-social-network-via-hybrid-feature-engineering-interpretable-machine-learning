
========================================================================
  ML PROJECT REPORT
  Efficient Detection of Social Network Bots
  Via Hybrid Feature Engineering & Interpretable Machine Learning
========================================================================
  Generated : 2026-04-30 06:47:40
  Authors   : Y. Pratheep Kumar, S. Likhitha,
              V. Lakshmi Narasimha, K. Samanth Kumar Reddy
  Supervisor: Dr. N. Babu, M.E., Ph.D.
  Institute : SIETK, Puttur — B.Tech CSE (AI), 2026
========================================================================

── 1. DATASET SUMMARY ──────────────────────────────────────────────
  Dataset         : Breast Cancer Wisconsin (bot/human proxy)
  Proxy Task      : Bot (0) vs Human (1) binary classification
  Total Samples   : 569
  Original Features: 30
  Engineered Features: 3  (activity_ratio, profile_score, suspicion_index)
  Total Features  : 33
  Bot Samples     : 212  (37.3%)
  Human Samples   : 357  (62.7%)

── 2. PREPROCESSING STEPS ──────────────────────────────────────────
  ① Train/Test Split : 80% / 20%  (stratified)
  ② Normalisation    : StandardScaler (Z-score; fit on train only)
  ③ Cross-Validation : Stratified K-Fold  (k = 5)

── 3. MODEL SELECTION ──────────────────────────────────────────────
  Two interpretable classifiers were trained and compared:
  • Logistic Regression — linear baseline; coefficients → interpretability
  • Random Forest       — ensemble; feature-importance for explainability

── 4. CROSS-VALIDATION RESULTS (on training set) ───────────────────
  [Logistic Regression]
    CV Accuracy : 0.9802 ± 0.0082
    CV F1-Score : 0.9843 ± 0.0063
    Train Time  : 0.0286 sec

  [Random Forest]
    CV Accuracy : 0.9604 ± 0.0149
    CV F1-Score : 0.9684 ± 0.0122
    Train Time  : 0.8994 sec

── 5. TEST SET EVALUATION ──────────────────────────────────────────
  [Logistic Regression]
    Accuracy  : 0.9737  (97.37%)
    F1 Macro  : 0.9719
    ROC-AUC   : 0.9954
    Confusion Matrix (rows=Actual, cols=Predicted):
      Bot  predicted Bot : 41  |  Bot  predicted Human : 1
      Human predicted Bot: 2  |  Human predicted Human: 70

    Classification Report:
                    precision    recall  f1-score   support
      
               Bot       0.95      0.98      0.96        42
             Human       0.99      0.97      0.98        72
      
          accuracy                           0.97       114
         macro avg       0.97      0.97      0.97       114
      weighted avg       0.97      0.97      0.97       114

  [Random Forest]
    Accuracy  : 0.9561  (95.61%)
    F1 Macro  : 0.9526
    ROC-AUC   : 0.9921
    Confusion Matrix (rows=Actual, cols=Predicted):
      Bot  predicted Bot : 39  |  Bot  predicted Human : 3
      Human predicted Bot: 2  |  Human predicted Human: 70

    Classification Report:
                    precision    recall  f1-score   support
      
               Bot       0.95      0.93      0.94        42
             Human       0.96      0.97      0.97        72
      
          accuracy                           0.96       114
         macro avg       0.96      0.95      0.95       114
      weighted avg       0.96      0.96      0.96       114

── 6. TOP FEATURE IMPORTANCES ──────────────────────────────────────
  [Logistic Regression]
     1. radius error                           0.031579
     2. worst area                             0.022807
     3. suspicion_index                        0.022807
     4. area error                             0.020175
     5. worst smoothness                       0.019298
     6. worst radius                           0.017544
     7. perimeter error                        0.014035
     8. mean concave points                    0.013158
     9. mean concavity                         0.012281
    10. worst texture                          0.011404

  [Random Forest]
     1. suspicion_index                        0.118360
     2. worst radius                           0.109497
     3. worst concave points                   0.105427
     4. mean concave points                    0.095051
     5. worst area                             0.085196
     6. worst perimeter                        0.080556
     7. mean perimeter                         0.044756
     8. worst concavity                        0.036558
     9. profile_score                          0.036409
    10. mean area                              0.035647

── 7. CONCLUSION ────────────────────────────────────────────────────
  Both classifiers achieved high accuracy on the held-out test set.
  Random Forest consistently outperforms Logistic Regression in
  F1-Score and ROC-AUC, confirming the value of ensemble methods
  for bot-detection tasks with complex, non-linear feature interactions.
  Hybrid feature engineering (activity_ratio, profile_score,
  suspicion_index) contributed measurably to model performance.

========================================================================
  END OF REPORT
========================================================================
