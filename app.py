"""Feature Engineering Impact Analyzer - Streamlit interface.

This module is a thin presentation layer. All data science logic lives in the
``fea`` package (src/fea) so it stays testable and reusable outside the UI.
"""

from __future__ import annotations

import io
import logging
import os
import sys

# Ensure the `fea` package (under `src/`) is importable when the app runs from
# the repository root without an editable install (e.g. Streamlit Cloud).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import joblib
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

from fea import data, modeling
from fea import pipeline as pl
from fea import visualization as viz
from fea.config import (
    DEFAULT_CORR_THRESHOLD,
    DEFAULT_K_SELECT,
    DEFAULT_VARIANCE_THRESHOLD,
    MIN_ROWS_FOR_MODEL,
    MODEL_NAMES,
    EncodingMethod,
    FeatureSelectionMethod,
    MissingValueStrategy,
    OutlierDetection,
    OutlierTreatment,
    PipelineConfig,
    ScalingMethod,
)
from fea.pipeline import split_features_target

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Feature Engineering Impact Analyzer", layout="wide")


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Reading CSV...")
def load_dataframe_cached(raw: bytes) -> pd.DataFrame:
    """Parse uploaded CSV bytes into a DataFrame, cached by content."""
    return pd.read_csv(io.BytesIO(raw))


# ---------------------------------------------------------------------------
# Theme (light / dark) via CSS injection
# ---------------------------------------------------------------------------
DARK_THEME_CSS = """
/* App surfaces */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #0e1117 !important;
    color: #fafafa !important;
}
[data-testid="stHeader"] { background-color: rgba(14,17,23,0.9) !important; }
[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #262c36 !important;
}

/* Text */
[data-testid="stMarkdownContainer"] p,
[data-testid="stHeading"] h1,
[data-testid="stHeading"] h2,
[data-testid="stHeading"] h3,
[data-testid="stCaptionContainer"] p,
[data-testid="stWidgetLabel"] p,
[data-testid="stText"],
[data-testid="stHeaderActionElements"] p { color: #fafafa !important; }

/* Metrics */
[data-testid="stMetric"] { background-color: #1a202a !important; }
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] p { color: #fafafa !important; }

/* Alerts / info boxes */
[data-testid="stAlert"] { background-color: #1a202a !important; color: #fafafa !important; }
[data-testid="stAlert"] p { color: #fafafa !important; }

/* DataFrames & tables */
[data-testid="stDataFrame"] { background-color: #161b22 !important; }
[data-testid="stDataFrame"] * { color: #fafafa !important; }
[data-testid="stDataFrame"] [data-testid="stTable"] {
    background-color: #161b22 !important;
}

/* Widgets / inputs / buttons */
[data-testid="stFileUploaderDropzone"] {
    background-color: #161b22 !important;
    border-color: #3a4354 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] p { color: #9aa0aa !important; }
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-secondary"] {
    border-color: #3a4354 !important;
    color: #fafafa !important;
    background-color: #222a37 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: #161b22 !important;
    color: #fafafa !important;
}
[data-testid="stRadio"] [data-baseweb="radio"] label,
[data-testid="stCheckbox"] label span { color: #fafafa !important; }
[data-testid="stExpander"] { color: #fafafa !important; }

/* Code blocks */
[data-testid="stMarkdownContainer"] code {
    background-color: #1a202a !important;
    color: #f0c674 !important;
}
"""

LIGHT_THEME_CSS = """
/* Revert surfaces to Streamlit's native light theme */
.stApp, [data-testid="stAppViewContainer"] {
    background-color: #ffffff !important;
    color: #31333f !important;
}
[data-testid="stHeader"] { background-color: rgba(255,255,255,0.9) !important; }
[data-testid="stSidebar"] {
    background-color: #f0f2f6 !important;
    border-right: 1px solid #e6e9ef !important;
}
[data-testid="stMarkdownContainer"] p,
[data-testid="stHeading"] h1,
[data-testid="stHeading"] h2,
[data-testid="stHeading"] h3,
[data-testid="stCaptionContainer"] p,
[data-testid="stWidgetLabel"] p,
[data-testid="stText"] { color: #31333f !important; }
[data-testid="stMetric"] { background-color: #f0f2f6 !important; }
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p,
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] p { color: #31333f !important; }
[data-testid="stAlert"] { background-color: #f0f2f6 !important; color: #31333f !important; }
[data-testid="stAlert"] p { color: #31333f !important; }
[data-testid="stDataFrame"] { background-color: #ffffff !important; }
[data-testid="stDataFrame"] * { color: #31333f !important; }
[data-testid="stDataFrame"] [data-testid="stTable"] { background-color: #ffffff !important; }
[data-testid="stFileUploaderDropzone"] {
    background-color: #ffffff !important;
    border-color: #d0d4da !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] p { color: #6c7178 !important; }
[data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-secondary"] {
    border-color: #d0d4da !important;
    color: #31333f !important;
    background-color: #ffffff !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #31333f !important;
}
[data-testid="stRadio"] [data-baseweb="radio"] label,
[data-testid="stCheckbox"] label span { color: #31333f !important; }
[data-testid="stExpander"] { color: #31333f !important; }
[data-testid="stMarkdownContainer"] code {
    background-color: #f0f2f6 !important;
    color: #c7254e !important;
}
"""


def render_theme_toggle() -> None:
    """Show a Light/Dark toggle and inject the matching theme CSS."""
    st.sidebar.markdown("### Appearance")
    theme = st.sidebar.radio(
        "Theme",
        options=["Light", "Dark"],
        horizontal=True,
        label_visibility="collapsed",
        index=1,
    )
    css = DARK_THEME_CSS if theme == "Dark" else LIGHT_THEME_CSS
    if css:
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------------------------
def render_sidebar_config() -> tuple[str, PipelineConfig]:
    """Render sidebar controls and return ``(model_name, config)``."""
    st.sidebar.header("⚙️ Configuration")

    model_name = st.sidebar.selectbox(
        "Model algorithm",
        options=list(MODEL_NAMES),
        help=(
            "Used for BOTH baseline and engineered models so any difference is caused "
            "by the data, not the algorithm."
        ),
    )

    st.sidebar.markdown("### Preprocessing")
    missing_strategy = st.sidebar.radio(
        "Missing values",
        options=[m.value for m in MissingValueStrategy],
        index=1,
    )
    detect_outliers = st.sidebar.checkbox("Detect & treat outliers", value=True)
    outlier_detection = OutlierDetection.IQR
    outlier_treatment = OutlierTreatment.NONE
    if detect_outliers:
        outlier_detection = OutlierDetection(
            st.sidebar.selectbox("Outlier detection", options=[m.value for m in OutlierDetection])
        )
        outlier_treatment = OutlierTreatment(
            st.sidebar.selectbox("Outlier treatment", options=[m.value for m in OutlierTreatment])
        )

    scale = st.sidebar.checkbox("Scale numeric features", value=True)
    scaling_method = ScalingMethod(
        st.sidebar.selectbox("Scaling method", options=[m.value for m in ScalingMethod])
    )
    encoding_method = EncodingMethod(
        st.sidebar.selectbox("Categorical encoding", options=[m.value for m in EncodingMethod])
    )

    st.sidebar.markdown("### Feature selection")
    fs_method = FeatureSelectionMethod(
        st.sidebar.selectbox(
            "Method",
            options=[m.value for m in FeatureSelectionMethod],
            help="Applied inside the engineered model pipeline (fit on training data only).",
        )
    )
    variance_threshold = DEFAULT_VARIANCE_THRESHOLD
    correlation_threshold = DEFAULT_CORR_THRESHOLD
    k_best = DEFAULT_K_SELECT
    n_features_rfe = DEFAULT_K_SELECT
    if fs_method == FeatureSelectionMethod.VARIANCE:
        variance_threshold = st.sidebar.slider(
            "Min variance", 0.0, 1.0, DEFAULT_VARIANCE_THRESHOLD, 0.01
        )
    elif fs_method == FeatureSelectionMethod.CORRELATION:
        correlation_threshold = st.sidebar.slider(
            "Correlation threshold", 0.5, 1.0, DEFAULT_CORR_THRESHOLD, 0.05
        )
    elif fs_method in (FeatureSelectionMethod.SELECT_K_BEST, FeatureSelectionMethod.MUTUAL_INFO):
        k_best = st.sidebar.slider("Number of features (K)", 1, 50, DEFAULT_K_SELECT)
    elif fs_method == FeatureSelectionMethod.RFE:
        n_features_rfe = st.sidebar.slider("Number of features", 1, 50, DEFAULT_K_SELECT)

    config = PipelineConfig(
        missing_strategy=MissingValueStrategy(missing_strategy),
        detect_outliers=detect_outliers,
        outlier_detection=outlier_detection,
        outlier_treatment=outlier_treatment,
        scale=scale,
        scaling_method=scaling_method,
        encoding_method=encoding_method,
        feature_selection=fs_method,
        variance_threshold=variance_threshold,
        correlation_threshold=correlation_threshold,
        k_best=k_best,
        n_features_rfe=n_features_rfe,
    )
    return model_name, config


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------
def run_analysis(df: pd.DataFrame, target: str, model_name: str, config: PipelineConfig) -> None:
    """Execute the full workflow: clean, split, baseline vs engineered, evaluate."""
    st.header("1️⃣ Data Cleaning")
    df_work, n_duplicates = data.remove_duplicates(df)
    if n_duplicates:
        st.warning(f"Removed **{n_duplicates}** duplicate row(s).")
    else:
        st.caption("✅ No duplicate rows found.")

    total_missing = int(df_work.isnull().sum().sum())
    st.caption(f"Total missing values: **{total_missing}** · Rows: {df_work.shape[0]}")

    # Outlier handling is a row-level operation and must run before splitting.
    if config.detect_outliers:
        numeric_feats, _ = split_features_target(df_work, target)
        if numeric_feats:
            outlier_mask = preprocessing_detect_outliers(df_work, numeric_feats, config.outlier_detection)
            rows_with_outliers = int(outlier_mask.any(axis=1).sum())
            st.caption(
                f"Outliers ({config.outlier_detection.value}): **{rows_with_outliers}** row(s) across "
                f"{len(numeric_feats)} numeric column(s)."
            )
            if config.outlier_treatment != OutlierTreatment.NONE:
                df_work, note = preprocessing_apply_treatment(
                    df_work, numeric_feats, config.outlier_treatment, outlier_mask
                )
                st.caption(note)
        else:
            st.info("No numeric feature columns to check for outliers.")

    # Encode the target label once, fit on training data only.
    y_all = df_work[target]
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_all.astype(str))

    X_all = df_work.drop(columns=[target])
    numeric_cols, categorical_cols = split_features_target(df_work, target)

    if X_all.shape[0] < MIN_ROWS_FOR_MODEL or len(numeric_cols) + len(categorical_cols) < 1:
        st.error("Not enough usable data to train a model after cleaning.")
        return

    X_train, X_test, y_train, y_test = modeling.split_data(X_all, y_encoded, config)

    # --- Baseline (naive) model ---
    st.header("2️⃣ Baseline Model (Raw / Naive)")
    baseline_pipeline = pl.build_baseline_pipeline(numeric_cols, config.missing_strategy, model_name)
    baseline_result = modeling.evaluate_model(
        baseline_pipeline, X_train, y_train, X_test, y_test,
        feature_names=numeric_cols, config=config,
    )
    st.metric(f"Baseline Accuracy ({model_name})", f"{baseline_result.accuracy:.2%}")
    st.caption("Baseline = numeric columns only, imputed, no scaling/encoding/feature-selection.")

    # --- Engineered model ---
    st.header("3️⃣ Engineered Model (Full Pipeline)")
    engineered_pipeline = pl.build_model_pipeline(config, numeric_cols, categorical_cols, model_name)
    feature_names = None
    try:
        engineered_pipeline.fit(X_train, y_train)
        feature_names = pl.final_feature_names(engineered_pipeline, X_train)
    except Exception as exc:  # pragma: no cover - surface errors to the user
        logger.exception("Engineered pipeline failed to fit")
        st.error(f"❌ The engineered pipeline could not be trained: {exc}")
        return

    engineered_result = modeling.evaluate_model(
        engineered_pipeline, X_train, y_train, X_test, y_test,
        feature_names=feature_names or numeric_cols, config=config,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{engineered_result.accuracy:.2%}")
    m2.metric("Precision", f"{engineered_result.precision:.2%}")
    m3.metric("Recall", f"{engineered_result.recall:.2%}")
    m4.metric("F1 Score", f"{engineered_result.f1:.2%}")
    if engineered_result.roc_auc is not None:
        st.metric("ROC-AUC", f"{engineered_result.roc_auc:.3f}")

    # --- Comparison ---
    st.header("4️⃣ Before vs After")
    col_x, col_y, col_z = st.columns(3)
    col_x.metric("Baseline Accuracy", f"{baseline_result.accuracy:.2%}")
    col_y.metric("Engineered Accuracy", f"{engineered_result.accuracy:.2%}")
    delta = engineered_result.accuracy - baseline_result.accuracy
    col_z.metric("Improvement", f"{delta:+.2%}")

    if delta > 0:
        st.success(f"✅ Feature engineering improved accuracy by {delta:.2%}.")
    elif delta < 0:
        st.warning(
            f"⚠️ Engineered accuracy is {abs(delta):.2%} lower than baseline. This can happen with small "
            f"datasets or sparse one-hot columns — try a different encoding/scaling/selection combination."
        )
    else:
        st.info("No change in accuracy — try a different configuration.")

    # --- Detailed evaluation ---
    st.header("5️⃣ Detailed Evaluation (Engineered)")
    fig_cm = viz.confusion_matrix_figure(pd.DataFrame(engineered_result.confusion_matrix))
    st.pyplot(fig_cm)
    with st.expander("Full classification report"):
        st.text(engineered_result.report)
    if engineered_result.cv_mean is not None:
        st.write(
            f"5-fold CV Accuracy: **{engineered_result.cv_mean:.2%}** "
            f"(± {engineered_result.cv_std:.2%}) · "
            f"{', '.join(f'{s:.2%}' for s in engineered_result.cv_scores)}"
        )

    # --- Correlation heatmap on engineered features ---
    engineered_train = pl.transform_features(engineered_pipeline, X_train)
    if engineered_train.shape[1] >= 2:
        corr = engineered_train.corr()
        st.subheader("Correlation Heatmap (Engineered Features)")
        st.pyplot(viz.correlation_heatmap(corr))
        st.caption(
            f"Correlations across {corr.shape[1]} engineered numeric columns "
            "(annotations hidden above 12 columns for readability)."
        )
    else:
        st.info("Not enough engineered numeric columns to compute correlations.")

    # --- Feature importance ---
    st.header("6️⃣ Feature Importance")
    classifier = engineered_pipeline.named_steps["classifier"]
    importance = modeling.feature_importance(classifier, feature_names or numeric_cols)
    if not importance.empty:
        st.dataframe(importance)
    else:
        st.caption(
            f"{model_name} does not expose feature importances or coefficients directly."
        )

    # --- Model comparison across algorithms ---
    st.header("7️⃣ Model Comparison")
    if st.button("Run comparison across all algorithms"):
        avg = "binary" if len(label_encoder.classes_) == 2 else "weighted"
        engineered_train = pl.transform_features(engineered_pipeline, X_train)
        engineered_test = pl.transform_features(engineered_pipeline, X_test)
        comp = modeling.compare_models(
            list(MODEL_NAMES), engineered_train, y_train, engineered_test, y_test, avg=avg
        )
        display_comp = comp.copy()
        for col in ("Accuracy", "F1 Score"):
            display_comp[col] = display_comp[col].apply(
                lambda v: f"{v:.2%}" if pd.notna(v) else "—"
            )
        st.dataframe(display_comp)
        best = comp.dropna()
        if not best.empty:
            st.success(
                f"🏆 Best model: **{best.iloc[0]['Model']}** "
                f"({best.iloc[0]['Accuracy']:.2%} accuracy)."
            )

    # --- Downloads ---
    st.header("8️⃣ Export")
    download_pipeline(engineered_pipeline)
    download_engineered_csv(df, target, model_name, config)


def download_pipeline(engineered_pipeline) -> None:
    buffer = io.BytesIO()
    joblib.dump(engineered_pipeline, buffer)
    buffer.seek(0)
    st.download_button(
        "⬇️ Download deployable pipeline (.pkl)",
        data=buffer,
        file_name="feature_engineering_pipeline.pkl",
        mime="application/octet-stream",
    )
    st.caption(
        "This single artefact bundles preprocessing + model. Load and predict with: "
        "`pipe = joblib.load('feature_engineering_pipeline.pkl'); pipe.predict(new_df)`."
    )


def download_engineered_csv(df: pd.DataFrame, target: str, model_name: str, config: PipelineConfig) -> None:
    try:
        result = pl.engineer_features(df, target, config)
        st.download_button(
            "⬇️ Download engineered dataset (CSV)",
            data=result.df.to_csv(index=False).encode("utf-8"),
            file_name="engineered_dataset.csv",
            mime="text/csv",
        )
        st.caption(
            f"Engineered shape: {result.df.shape[0]} rows × {result.df.shape[1]} columns "
            f"(started as {df.shape[0]} rows)."
        )
    except Exception as exc:  # pragma: no cover - graceful degradation
        st.warning(f"Could not produce the engineered CSV export: {exc}")


# ---------------------------------------------------------------------------
# Thin wrappers so the UI does not import preprocessing internals directly
# ---------------------------------------------------------------------------
def preprocessing_detect_outliers(df, cols, method):
    from fea.preprocessing import detect_outliers

    return detect_outliers(df, cols, method)


def preprocessing_apply_treatment(df, cols, treatment, mask):
    from fea.preprocessing import apply_outlier_treatment

    return apply_outlier_treatment(df, cols, treatment, mask)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    render_theme_toggle()

    st.title("📊 Feature Engineering Impact Analyzer")
    st.caption(
        "Upload a CSV, pick a target column, and see exactly how much feature engineering "
        "improves model performance — evaluated on a held-out test set, leak-free."
    )

    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded is None:
        st.info("👆 Upload a CSV to get started.")
        return

    df = load_dataframe_cached(uploaded.getvalue())

    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10))
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing values", int(df.isnull().sum().sum()))

    st.subheader("Column Data Types")
    st.dataframe(data.summarize_dataframe(df))

    st.subheader("Select Target Column")
    target = st.selectbox("Which column do you want to predict?", options=list(df.columns))

    validation = data.validate_target(df, target)
    for warning in validation.warnings:
        st.warning(f"⚠️ {warning}")
    if not validation.is_valid:
        for message in validation.messages:
            st.error(f"❌ {message}")
        st.stop()

    st.info(f"🔍 Detected problem type: **{validation.problem_type.value}** "
            f"({validation.n_classes} distinct classes)")

    model_name, config = render_sidebar_config()
    run_analysis(df, target, model_name, config)


if __name__ == "__main__":
    main()
