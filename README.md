# 🤖 Bot Detection ML — Hybrid Feature Engineering & Interpretable ML

**B.Tech Final Year Project — SIETK, 2026**
Department of CSE (Artificial Intelligence)

| Roll No     | Name                    |
|-------------|-------------------------|
| 22F61A3109  | V LAKSHMI NARASIMHA     |

**Supervisor:** Dr. N. Babu, M.E., Ph.D. — Associate Professor, Dept. of CSE (AIMD)

---

## 📌 Overview

This project implements a complete ML pipeline for **Social Network Bot Detection**
using **Hybrid Feature Engineering** and two interpretable classifiers:

- **Logistic Regression** — linear baseline with high interpretability
- **Random Forest** — ensemble model with built-in feature importance

A built-in scikit-learn dataset is used as a binary
classification proxy (Bot vs Human), with three engineered features added
to simulate real social-network behavioural signals.

---

## 🗂️ Project Structure

```
bot-detection-ml/
├── bot_detection_ml.py   # Main ML pipeline (all steps)
├── ml_report.md          # Auto-generated evaluation report
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## ⚙️ Requirements

- Python 3.9+
- scikit-learn
- numpy
- pandas

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Program

```bash
python bot_detection_ml.py
```

The program will:
1. Load the dataset
2. Engineer hybrid features
3. Preprocess (split + normalize)
4. Train Logistic Regression & Random Forest
5. Evaluate with Accuracy, F1, ROC-AUC
6. Print top-10 feature importances
7. Print and save the full Markdown report → `ml_report.md`

---

## 📊 Sample Results

| Model                | CV Accuracy     | Test Accuracy | ROC-AUC |
|----------------------|-----------------|---------------|---------|
| Logistic Regression  | 98.02% ± 0.82%  | 97.37%        | 0.9954  |
| Random Forest        | 96.04% ± 1.49%  | 95.61%        | 0.9921  |

---

## 📤 Pushing to GitHub

Follow these steps to push this project to your GitHub repository:

```bash
# 1. Initialize a local Git repository
git init

# 2. Stage all project files
git add .

# 3. Commit with a descriptive message
git commit -m "Initial ML project — Bot Detection via Hybrid Feature Engineering"

# 4. Create a new repo on GitHub (github.com → New Repository)
#    Then link it as the remote origin:
git remote add origin https://github.com/<your-username>/bot-detection-ml.git

# 5. Rename the default branch to 'main' (if not already)
git branch -M main

# 6. Push to GitHub
git push -u origin main

---

## 📁 requirements.txt

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
```

---

## 📄 License

Academic project — Siddharth Institute of Engineering & Technology (Autonomous), 2026.
