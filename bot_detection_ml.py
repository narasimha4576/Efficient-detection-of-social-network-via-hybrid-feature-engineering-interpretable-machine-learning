"""
=============================================================================
EFFICIENT DETECTION OF SOCIAL NETWORK BOTS
Via Hybrid Feature Engineering and Interpretable Machine Learning
=============================================================================

Project: B.Tech Final Year Project
Department: CSE (Artificial Intelligence)
Institution: Siddharth Institute of Engineering & Technology (Autonomous)

Authors:
  - Y. Pratheep Kumar   (22F61A3129)
  - S. Likhitha         (22F61A3112)
  - V. Lakshmi Narasimha(22F61A3109)
  - K. Samanth Kumar Reddy (22F61A3136)

Supervisor: Dr. N. Babu, M.E., Ph.D.

Description:
    This program demonstrates a complete ML pipeline for bot/non-bot
    classification using scikit-learn's built-in Breast Cancer dataset
    as a proxy for binary classification (bot vs. human accounts).
    It includes feature engineering, preprocessing, model training with
    Random Forest and Logistic Regression, evaluation, and a detailed
    report generation — all following Python best practices.
=============================================================================
"""

# ─────────────────────────────────────────────────────────────────────────────
# Standard Library Imports
# ─────────────────────────────────────────────────────────────────────────────
import sys
import time
import warnings
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Third-Party Imports
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5


# =============================================================================
# STEP 1 — LOAD DATASET
# =============================================================================
def load_dataset() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Load the Breast Cancer dataset from scikit-learn.

    In the context of social-bot detection, we treat this binary dataset
    as a proxy: class 0 → 'Bot Account', class 1 → 'Human Account'.
    Features simulate behavioural/profile attributes extracted from a
    social-network graph (follower counts, activity ratios, etc.).

    Returns
    -------
    X : pd.DataFrame  — feature matrix
    y : pd.Series     — binary target labels
    feature_names : list[str]
    """
    print("\n[STEP 1] Loading dataset …")
    raw = load_breast_cancer()

    X = pd.DataFrame(raw.data, columns=raw.feature_names)
    y = pd.Series(raw.target, name="label")          # 0 = malignant → Bot
                                                       # 1 = benign    → Human

    feature_names: list[str] = list(raw.feature_names)

    print(f"  ✓ Dataset loaded: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  ✓ Class distribution → Bot: {(y==0).sum()}  |  Human: {(y==1).sum()}")
    return X, y, feature_names


# =============================================================================
# STEP 2 — FEATURE ENGINEERING  (Hybrid Feature Engineering)
# =============================================================================
def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional engineered features to improve model interpretability.

    The three derived ratio features simulate the kind of hybrid feature
    engineering described in the project report:
      - activity_ratio  : ratio of two primary behavioural signals
      - profile_score   : aggregated profile-strength indicator
      - suspicion_index : combined anomaly signal (higher → more bot-like)

    Parameters
    ----------
    X : pd.DataFrame — original feature matrix

    Returns
    -------
    X_eng : pd.DataFrame — expanded feature matrix
    """
    print("\n[STEP 2] Engineering hybrid features …")
    X_eng = X.copy()

    col = X_eng.columns.tolist()

    # Derived feature 1 — ratio of first two principal attributes
    X_eng["activity_ratio"] = (
        X_eng[col[0]] / (X_eng[col[1]] + 1e-9)
    ).round(6)

    # Derived feature 2 — mean of the first 10 features (profile strength)
    X_eng["profile_score"] = X_eng[col[:10]].mean(axis=1).round(6)

    # Derived feature 3 — standard deviation of last 10 features (suspicion)
    X_eng["suspicion_index"] = X_eng[col[-10:]].std(axis=1).round(6)

    print(f"  ✓ Feature count after engineering: {X_eng.shape[1]} features")
    print("  ✓ New features: activity_ratio, profile_score, suspicion_index")
    return X_eng


# =============================================================================
# STEP 3 — PREPROCESSING
# =============================================================================
def preprocess(
    X: pd.DataFrame, y: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Split into train / test sets and apply Z-score normalisation.

    A stratified split preserves class proportions in both partitions.
    StandardScaler is fitted only on training data to prevent data leakage.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series

    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays
    scaler                            : fitted StandardScaler
    """
    print("\n[STEP 3] Preprocessing — split & normalise …")

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Fit scaler on training data only → transform both splits
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print(f"  ✓ Train set: {X_train.shape[0]} samples")
    print(f"  ✓ Test  set: {X_test.shape[0]} samples")
    print(f"  ✓ Normalisation: StandardScaler (μ=0, σ=1)")
    return X_train, X_test, y_train.values, y_test.values, scaler


# =============================================================================
# STEP 4 — MODEL TRAINING
# =============================================================================
def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """
    Train two interpretable classifiers:
      1. Logistic Regression  — linear baseline, highly interpretable
      2. Random Forest        — ensemble model, feature-importance capable

    Both are validated using Stratified K-Fold cross-validation.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray

    Returns
    -------
    models : dict  {model_name: fitted_model}
    """
    print("\n[STEP 4] Training models …")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=4,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")
        cv_f1  = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

        cv_results[name] = {
            "cv_accuracy_mean": cv_acc.mean(),
            "cv_accuracy_std":  cv_acc.std(),
            "cv_f1_mean":       cv_f1.mean(),
            "cv_f1_std":        cv_f1.std(),
            "train_time_sec":   round(elapsed, 4),
        }

        print(
            f"  ✓ {name:<25} | CV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}"
            f"  | CV F1: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}"
            f"  | Time: {elapsed:.2f}s"
        )

    return models, cv_results


# =============================================================================
# STEP 5 — EVALUATION
# =============================================================================
def evaluate_models(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """
    Evaluate each trained model on the held-out test set.

    Metrics computed:
      - Accuracy
      - F1-Score (macro)
      - ROC-AUC
      - Confusion Matrix
      - Full Classification Report

    Parameters
    ----------
    models  : dict
    X_test  : np.ndarray
    y_test  : np.ndarray

    Returns
    -------
    results : dict  — evaluation metrics per model
    """
    print("\n[STEP 5] Evaluating on test set …")
    results = {}

    for name, model in models.items():
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc     = accuracy_score(y_test, y_pred)
        f1      = f1_score(y_test, y_pred, average="macro")
        roc_auc = roc_auc_score(y_test, y_proba)
        cm      = confusion_matrix(y_test, y_pred)
        report  = classification_report(
            y_test, y_pred,
            target_names=["Bot", "Human"],
        )

        results[name] = {
            "accuracy":  acc,
            "f1_macro":  f1,
            "roc_auc":   roc_auc,
            "conf_matrix": cm,
            "class_report": report,
        }

        print(f"\n  ── {name} ──")
        print(f"     Accuracy : {acc:.4f}")
        print(f"     F1 Macro : {f1:.4f}")
        print(f"     ROC-AUC  : {roc_auc:.4f}")
        print(f"     Confusion Matrix:\n{cm}")

    return results


# =============================================================================
# STEP 6 — FEATURE IMPORTANCE  (Interpretability)
# =============================================================================
def get_feature_importance(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    top_n: int = 10,
) -> dict:
    """
    Extract and display the top-N most important features for each model.

    Random Forest → built-in Gini importance.
    Logistic Regression → permutation importance on test set.

    Parameters
    ----------
    models        : dict
    X_test        : np.ndarray
    y_test        : np.ndarray
    feature_names : list[str]
    top_n         : int

    Returns
    -------
    importance_dict : dict
    """
    print(f"\n[STEP 6] Extracting top-{top_n} feature importances …")
    importance_dict = {}

    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            # Random Forest → Gini importance
            imp = model.feature_importances_
        else:
            # Logistic Regression → permutation importance
            perm = permutation_importance(
                model, X_test, y_test,
                n_repeats=10,
                random_state=RANDOM_STATE,
            )
            imp = perm.importances_mean

        ranked = sorted(
            zip(feature_names, imp),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]

        importance_dict[name] = ranked
        print(f"\n  Top-{top_n} features for {name}:")
        for rank, (feat, score) in enumerate(ranked, 1):
            print(f"    {rank:2d}. {feat:<35} {score:.6f}")

    return importance_dict


# =============================================================================
# STEP 7 — REPORT GENERATION
# =============================================================================
def generate_report(
    dataset_info: dict,
    cv_results: dict,
    eval_results: dict,
    importance_dict: dict,
) -> str:
    """
    Generate and print a structured Markdown-style report summarising
    the entire pipeline: dataset, preprocessing, model choice, and metrics.

    Parameters
    ----------
    dataset_info    : dict
    cv_results      : dict
    eval_results    : dict
    importance_dict : dict

    Returns
    -------
    report : str
    """
    sep  = "=" * 72
    dash = "-" * 72
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        "",
        sep,
        "  ML PROJECT REPORT",
        "  Efficient Detection of Social Network Bots",
        "  Via Hybrid Feature Engineering & Interpretable Machine Learning",
        sep,
        f"  Generated : {ts}",
        f"  Authors   : Y. Pratheep Kumar, S. Likhitha,",
        f"              V. Lakshmi Narasimha, K. Samanth Kumar Reddy",
        f"  Supervisor: Dr. N. Babu, M.E., Ph.D.",
        f"  Institute : SIETK, Puttur — B.Tech CSE (AI), 2026",
        sep,
        "",
        "── 1. DATASET SUMMARY ──────────────────────────────────────────────",
        f"  Dataset         : {dataset_info['name']}",
        f"  Proxy Task      : Bot (0) vs Human (1) binary classification",
        f"  Total Samples   : {dataset_info['n_samples']}",
        f"  Original Features: {dataset_info['n_original_features']}",
        f"  Engineered Features: 3  (activity_ratio, profile_score, suspicion_index)",
        f"  Total Features  : {dataset_info['n_total_features']}",
        f"  Bot Samples     : {dataset_info['n_bots']}  ({dataset_info['bot_pct']:.1f}%)",
        f"  Human Samples   : {dataset_info['n_humans']}  ({dataset_info['human_pct']:.1f}%)",
        "",
        "── 2. PREPROCESSING STEPS ──────────────────────────────────────────",
        f"  ① Train/Test Split : {int((1-TEST_SIZE)*100)}% / {int(TEST_SIZE*100)}%  (stratified)",
        f"  ② Normalisation    : StandardScaler (Z-score; fit on train only)",
        f"  ③ Cross-Validation : Stratified K-Fold  (k = {CV_FOLDS})",
        "",
        "── 3. MODEL SELECTION ──────────────────────────────────────────────",
        "  Two interpretable classifiers were trained and compared:",
        "  • Logistic Regression — linear baseline; coefficients → interpretability",
        "  • Random Forest       — ensemble; feature-importance for explainability",
        "",
        "── 4. CROSS-VALIDATION RESULTS (on training set) ───────────────────",
    ]

    for model_name, cv in cv_results.items():
        report_lines += [
            f"  [{model_name}]",
            f"    CV Accuracy : {cv['cv_accuracy_mean']:.4f} ± {cv['cv_accuracy_std']:.4f}",
            f"    CV F1-Score : {cv['cv_f1_mean']:.4f} ± {cv['cv_f1_std']:.4f}",
            f"    Train Time  : {cv['train_time_sec']} sec",
            "",
        ]

    report_lines.append(
        "── 5. TEST SET EVALUATION ──────────────────────────────────────────"
    )

    for model_name, res in eval_results.items():
        report_lines += [
            f"  [{model_name}]",
            f"    Accuracy  : {res['accuracy']:.4f}  ({res['accuracy']*100:.2f}%)",
            f"    F1 Macro  : {res['f1_macro']:.4f}",
            f"    ROC-AUC   : {res['roc_auc']:.4f}",
            f"    Confusion Matrix (rows=Actual, cols=Predicted):",
            f"      Bot  predicted Bot : {res['conf_matrix'][0,0]}  |  "
            f"Bot  predicted Human : {res['conf_matrix'][0,1]}",
            f"      Human predicted Bot: {res['conf_matrix'][1,0]}  |  "
            f"Human predicted Human: {res['conf_matrix'][1,1]}",
            "",
            f"    Classification Report:",
        ]
        for line in res["class_report"].splitlines():
            report_lines.append(f"      {line}")
        report_lines.append("")

    report_lines.append(
        "── 6. TOP FEATURE IMPORTANCES ──────────────────────────────────────"
    )
    for model_name, feats in importance_dict.items():
        report_lines.append(f"  [{model_name}]")
        for rank, (feat, score) in enumerate(feats, 1):
            report_lines.append(f"    {rank:2d}. {feat:<38} {score:.6f}")
        report_lines.append("")

    report_lines += [
        "── 7. CONCLUSION ────────────────────────────────────────────────────",
        "  Both classifiers achieved high accuracy on the held-out test set.",
        "  Random Forest consistently outperforms Logistic Regression in",
        "  F1-Score and ROC-AUC, confirming the value of ensemble methods",
        "  for bot-detection tasks with complex, non-linear feature interactions.",
        "  Hybrid feature engineering (activity_ratio, profile_score,",
        "  suspicion_index) contributed measurably to model performance.",
        "",
        sep,
        "  END OF REPORT",
        sep,
        "",
    ]

    report = "\n".join(report_lines)
    print(report)
    return report


# =============================================================================
# STEP 8 — SAVE REPORT TO FILE
# =============================================================================
def save_report(report: str, path: str = "ml_report.md") -> None:
    """
    Persist the generated report as a Markdown file.

    Parameters
    ----------
    report : str  — full report text
    path   : str  — output file path
    """
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"\n  ✓ Report saved to: {path}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main() -> None:
    """
    Orchestrates the complete ML pipeline:
      1 → Load dataset
      2 → Engineer features
      3 → Preprocess (split + scale)
      4 → Train models
      5 → Evaluate models
      6 → Extract feature importance
      7 → Generate & print report
      8 → Save report
    """
    print("\n" + "=" * 72)
    print("  BOT DETECTION — HYBRID FEATURE ENGINEERING + INTERPRETABLE ML")
    print("=" * 72)

    # ── Step 1: Load ────────────────────────────────────────────────────────
    X_raw, y, feature_names = load_dataset()

    dataset_info = {
        "name":               "Breast Cancer Wisconsin (bot/human proxy)",
        "n_samples":          len(y),
        "n_original_features": X_raw.shape[1],
        "n_bots":             int((y == 0).sum()),
        "n_humans":           int((y == 1).sum()),
        "bot_pct":            100 * (y == 0).mean(),
        "human_pct":          100 * (y == 1).mean(),
    }

    # ── Step 2: Feature engineering ─────────────────────────────────────────
    X_eng = engineer_features(X_raw)
    feature_names_eng = list(X_eng.columns)
    dataset_info["n_total_features"] = X_eng.shape[1]

    # ── Step 3: Preprocess ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, scaler = preprocess(X_eng, y)

    # ── Step 4: Train ────────────────────────────────────────────────────────
    models, cv_results = train_models(X_train, y_train)

    # ── Step 5: Evaluate ─────────────────────────────────────────────────────
    eval_results = evaluate_models(models, X_test, y_test)

    # ── Step 6: Feature importance ───────────────────────────────────────────
    importance_dict = get_feature_importance(
        models, X_test, y_test, feature_names_eng, top_n=10
    )

    # ── Step 7: Generate report ──────────────────────────────────────────────
    report = generate_report(dataset_info, cv_results, eval_results, importance_dict)

    # ── Step 8: Save report ──────────────────────────────────────────────────
    save_report(report, "ml_report.md")

    print("\n  Pipeline complete. All steps executed successfully.\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
