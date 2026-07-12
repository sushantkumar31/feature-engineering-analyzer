# 📊 Feature Engineering Impact Analyzer

A Streamlit app that proves feature engineering matters — with numbers, not claims.

Upload any CSV, pick a target column, and watch the app train a **baseline model on raw data**, apply real feature engineering (missing value imputation, scaling, categorical encoding), retrain the **same model on the engineered data**, and show you the exact accuracy improvement.

**🔗 Live app:** https://feature-engineering-analyzer-slr4etpc88hhfg6b3ouahx.streamlit.app/

---

## Why this exists

Most ML tutorials tell you "feature engineering improves your model" without ever showing you the before/after number on the *same* model, the *same* split, the *same* everything — so you never actually see the effect isolated. This app does that in one interactive session, on any dataset you upload.

## What it does

1. **Upload** a CSV and select a target column (with validation against bad targets — constant columns, ID-like columns, mostly-missing columns)
2. **Baseline model**: trained naively on raw data (drop missing rows, numeric columns only) — this is deliberately dumb, to expose what a model can do *without* any engineering
3. **Missing value handling**: choose drop rows / mean / median (numeric) + mode (categorical)
4. **Feature scaling**: Standardization or Min-Max normalization, with before/after distribution plots
5. **Categorical encoding**: One-Hot or Label encoding, with a preview of transformed columns and a high-cardinality warning
6. **Correlation heatmap** of the fully engineered dataset, plus top features correlated with the target
7. **Final model**: same algorithm (Logistic Regression), same train/test split logic, trained on engineered data
8. **Before vs After accuracy comparison** — the whole point of the app
9. **Download button** for the cleaned, engineered CSV

## Tech stack

- **Streamlit** — UI, single-file app, no frontend code
- **pandas** — data handling
- **scikit-learn** — model training, scaling, encoding
- **matplotlib / seaborn** — distribution plots and correlation heatmap
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
uv sync
uv run streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Try it with

- [Titanic dataset](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) — target column: `Survived`. Good mix of missing values (`Age`, `Cabin`) and categorical features (`Sex`, `Embarked`).

## Design notes

- The baseline model is intentionally naive (drops NaNs, drops non-numeric columns) — this isn't a limitation, it's the point: it shows what happens with zero engineering effort.
- Baseline and final models use the identical algorithm, split ratio, and random seed, so any accuracy difference is attributable to the engineering, not randomness or a different model.
- Scaling matters more for algorithms like Logistic Regression/KNN/SVM than tree-based models — don't expect a big shift if you swap in a tree-based model later.
- No automatic feature dropping based on correlation — that's left as a diagnostic for the user to interpret, not something the app decides silently on your behalf.

## Possible future extensions

- Support for regression targets (currently classification only)
- Let the user pick the model algorithm (Random Forest, SVM, etc.)
- Per-column missing value strategy instead of one global strategy
