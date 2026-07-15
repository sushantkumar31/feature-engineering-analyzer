# 📊 Feature Engineering Impact Analyzer

A Streamlit app that proves feature engineering matters — with numbers, not claims.

Upload any CSV, pick a target column, and watch the app train a **baseline model on raw data**, apply real feature engineering and selection, retrain the **same algorithm on the engineered data**, and show you the exact improvement — across accuracy, precision, recall, F1, and more.

**🔗 Live app:** https://feature-engineering-analyzer-slr4etpc88hhfg6b3ouahx.streamlit.app/

---

## Why this exists

Most ML tutorials tell you "feature engineering improves your model" without ever showing you the before/after number on the *same* model, the *same* split, the *same* everything — so you never actually see the effect isolated. This app does that in one interactive session, on any dataset you upload.

## What it does

**Data loading & validation**
1. Upload a CSV, preview it, and see column dtypes, missing counts, and unique-value counts at a glance
2. Target column selection with validation against bad targets (constant columns, ID-like columns, continuous/regression-style targets, mostly-missing columns)
3. Automatic problem-type detection (currently classification only)
4. Small dataset and class imbalance warnings

**Baseline model**
5. Trained naively on raw data (drop missing rows, numeric columns only) — deliberately dumb, to expose what a model can do *without* any engineering

**Data cleaning & feature engineering**
6. Duplicate row detection and removal
7. Missing value handling: drop rows / mean / median (numeric) + mode (categorical)
8. Outlier detection (IQR or Z-score) with treatment options: remove, cap, or winsorize
9. Feature scaling: Standardization, Min-Max, or Robust Scaling — with before/after distribution plots
10. Categorical encoding: One-Hot or Label encoding, with a transformed-column preview and a high-cardinality warning
11. Correlation heatmap of the engineered dataset, plus top features correlated with the target
12. Constant/zero-variance feature detection

**Feature selection**
13. Choice of Variance Threshold, Correlation-based removal, SelectKBest (ANOVA F-test), Mutual Information, or RFE — all optional, all show exactly which features were kept or dropped

**Model training & evaluation**
14. Choice of algorithm — Logistic Regression, Decision Tree, Random Forest, KNN, SVM, or Naive Bayes — applied identically to both baseline and final model, so the comparison isolates the effect of engineering, not a different algorithm
15. Final model retrained on the fully engineered + selected data
16. **Before vs After comparison** — the whole point of the app
17. Full evaluation suite: Precision, Recall, F1, ROC-AUC (binary targets), Confusion Matrix, Classification Report, 5-fold Cross-Validation

**Model interpretation & comparison**
18. Feature importance (tree-based models) or coefficients (linear models)
19. One-click comparison across all 6 algorithms on the same engineered data, ranked by accuracy, with a best-model callout

**Export**
20. Download the cleaned, engineered CSV
21. Download the trained model as a `.pkl` file (via `joblib`)

## Tech stack

- **Streamlit** — UI, single-file app, no frontend code
- **pandas / numpy** — data handling
- **scikit-learn** — models, scaling, encoding, feature selection, evaluation metrics
- **matplotlib / seaborn** — distribution plots, correlation heatmap, confusion matrix
- **joblib** — trained model export
- **uv** — dependency management

No paid APIs, no database, no environment secrets.

## Running locally

**1. Install `uv`** (if you don't already have it):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip (any OS)
pip install uv
```

**2. Clone and run:**

```bash
git clone https://github.com/sushantkumar31/feature-engineering-analyzer.git
cd feature-engineering-analyzer
uv venv --python 3.12
uv pip install -r requirements.txt
uv run streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Try it with

- [Titanic dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) — target column: `Survived`. Good mix of missing values (`Age`, `Cabin`), categorical features (`Sex`, `Embarked`), and class balance worth exploring.

## Design notes

- The baseline model is intentionally naive (drops NaNs, drops non-numeric columns) — this isn't a limitation, it's the point: it shows what happens with zero engineering effort.
- Baseline and final models use the identical algorithm, split ratio, and random seed, so any difference in results is attributable to the engineering, not randomness or a different model.
- Model choice applies to *both* baseline and final training for the same reason — changing the algorithm and the data at the same time would make the comparison meaningless.
- Scaling matters more for algorithms like Logistic Regression/KNN/SVM than tree-based models — don't expect a big shift if you swap in a tree-based model.
- No automatic feature dropping based on correlation — feature selection is a deliberate, visible choice the user makes, not something the app decides silently.
- RFE falls back to a Decision Tree internally when the chosen model doesn't expose feature weights (e.g. KNN, Naive Bayes) — this is flagged to the user, not silently swapped.
- Deployment is pinned to **Python 3.12** (via Streamlit Cloud's Advanced Settings and `runtime.txt`/`.python-version` locally) since newer Python versions can lag behind on prebuilt wheels for packages like Pillow and PyArrow.

## Deliberately not included

- **XGBoost, SHAP, PDF report export** — each adds heavier, more fragile dependencies; skipped to keep the app easy to deploy and maintain reliably.
- **Light/Dark mode, tooltips, experiment history** — UI polish rather than feature-engineering substance.
- **Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)** — not yet implemented; a reasonable next extension if the app's scope grows further.

## Possible future extensions

- Support for regression targets (currently classification only)
- Per-column missing value strategy instead of one global strategy
- Hyperparameter tuning for the selected model
- Ordinal encoding option (for genuinely ordered categories)
- Date feature extraction (year/month/day/weekday) for datetime columns
