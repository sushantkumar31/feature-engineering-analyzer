import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Feature Engineering Impact Analyzer", layout="wide")

st.title("📊 Feature Engineering Impact Analyzer")
st.write("Upload a CSV, pick a target column, and see exactly how much feature engineering improves model accuracy.")

# ------------------------------------------------------------------
# STEP 1: Upload + preview + target selection
# ------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head(10))

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing values", int(df.isnull().sum().sum()))

    st.subheader("Column Data Types")
    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": df.dtypes.astype(str).values,
        "Missing": df.isnull().sum().values,
        "Unique values": [df[c].nunique() for c in df.columns]
    })
    st.dataframe(dtype_df)

    st.subheader("Select Target Column")
    target_col = st.selectbox("Which column do you want to predict?", options=df.columns)

    # ------------------------------------------------------------------
    # Target validation
    # ------------------------------------------------------------------
    n_unique = df[target_col].nunique()
    n_missing = df[target_col].isnull().sum()
    n_rows = df.shape[0]

    valid_target = True

    if n_unique <= 1:
        st.error("❌ This column has only one unique value — can't train a model on a constant target.")
        valid_target = False
    elif n_unique > 20 and n_unique > 0.5 * n_rows:
        st.warning(
            f"⚠️ This column has {n_unique} unique values out of {n_rows} rows — "
            f"it looks like an ID column, not a label. Pick a different target."
        )
        valid_target = False
    elif pd.api.types.is_numeric_dtype(df[target_col]) and n_unique > 20:
        st.error(
            f"❌ This column has {n_unique} unique numeric values — it looks like a continuous "
            f"(regression) target, not a set of classes. This app only supports classification "
            f"targets right now (a column with a small number of distinct categories, like 0/1 "
            f"or Win/Loss). Pick a different target column."
        )
        valid_target = False
    elif n_missing > 0.5 * n_rows:
        st.warning(f"⚠️ This column is {n_missing / n_rows:.0%} missing — results will be unreliable.")

    if valid_target:
        problem_type = "Classification"
        st.info(f"🔍 Detected problem type: **{problem_type}** ({n_unique} distinct classes)")

        if n_rows < 100:
            st.warning(f"⚠️ Small dataset ({n_rows} rows) — accuracy numbers may be unstable/unreliable "
                        f"across different train/test splits.")

        class_counts = df[target_col].value_counts()
        if len(class_counts) > 1 and (class_counts.min() / class_counts.max()) < 0.3:
            st.warning(f"⚠️ Class imbalance detected — largest class has {class_counts.max()} rows, "
                        f"smallest has {class_counts.min()}. Accuracy alone can be misleading here; "
                        f"check Precision/Recall/F1 in the final model section.")

        st.session_state["df"] = df
        st.session_state["target_col"] = target_col
        st.success(f"Target set to: **{target_col}**")

        st.subheader("Choose Model Algorithm")
        model_choice = st.selectbox(
            "This model will be used for BOTH the baseline and engineered comparison, "
            "so the improvement you see is caused only by the data, not a different algorithm.",
            options=["Logistic Regression", "Decision Tree", "Random Forest", "K-Nearest Neighbors",
                     "Support Vector Machine", "Naive Bayes"],
            index=0,
        )

        def build_model(name):
            if name == "Logistic Regression":
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(max_iter=1000)
            elif name == "Decision Tree":
                from sklearn.tree import DecisionTreeClassifier
                return DecisionTreeClassifier(random_state=42)
            elif name == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(random_state=42)
            elif name == "K-Nearest Neighbors":
                from sklearn.neighbors import KNeighborsClassifier
                return KNeighborsClassifier()
            elif name == "Support Vector Machine":
                from sklearn.svm import SVC
                return SVC(probability=True, random_state=42)
            elif name == "Naive Bayes":
                from sklearn.naive_bayes import GaussianNB
                return GaussianNB()

        # ------------------------------------------------------------------
        # Duplicate row detection
        # ------------------------------------------------------------------
        n_duplicates = int(df.duplicated().sum())
        if n_duplicates > 0:
            st.warning(f"⚠️ Found **{n_duplicates}** duplicate row(s) out of {df.shape[0]} total.")
            remove_duplicates = st.checkbox("Remove duplicate rows before engineering", value=True)
            if remove_duplicates:
                df = df.drop_duplicates().reset_index(drop=True)
                st.caption(f"Removed {n_duplicates} duplicate row(s). New row count: {df.shape[0]}")
        else:
            st.caption("✅ No duplicate rows found.")

        # ------------------------------------------------------------------
        # STEP 2: Baseline model (naive, no engineering)
        # ------------------------------------------------------------------
        st.header("1️⃣ Baseline Model (Raw Data, No Engineering)")

        baseline_df = df.dropna()  # naive: drop any row with missing values

        target_is_numeric = pd.api.types.is_numeric_dtype(baseline_df[target_col])

        if not target_is_numeric:
            # Target itself is categorical (e.g. Win/Loss). Label-encode ONLY the target
            # so we can still get a baseline number. Features stay untouched/naive.
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            baseline_df = baseline_df.copy()
            baseline_df[target_col] = le.fit_transform(baseline_df[target_col].astype(str))
            st.caption(
                f"ℹ️ Target '{target_col}' was categorical ({', '.join(map(str, le.classes_))}) — "
                f"label-encoded to numbers ({', '.join(f'{c}={i}' for i, c in enumerate(le.classes_))}) "
                f"so a baseline model can run. Feature columns are still untouched/naive at this stage."
            )

        baseline_df = baseline_df.select_dtypes(include="number")  # naive: keep only numeric feature columns

        if target_col not in baseline_df.columns:
            st.error(
                f"❌ Target column '{target_col}' could not be made numeric. Pick a different target."
            )
        elif baseline_df.shape[0] < 20 or baseline_df.shape[1] < 2:
            st.error(
                "❌ Not enough usable data left after naively dropping NaNs/non-numeric columns "
                "to train a baseline model. This itself proves why feature engineering matters!"
            )
        else:
            X = baseline_df.drop(columns=[target_col])
            y = baseline_df[target_col]

            stratify_arg = y if (y.nunique() < 20 and y.value_counts().min() >= 2) else None

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_arg
            )

            baseline_model = build_model(model_choice)
            baseline_model.fit(X_train, y_train)
            baseline_preds = baseline_model.predict(X_test)
            baseline_acc = accuracy_score(y_test, baseline_preds)

            st.metric(f"Baseline Accuracy ({model_choice}, raw/naive)", f"{baseline_acc:.2%}")
            st.caption(
                f"Trained on {baseline_df.shape[0]} rows (after dropping NaNs) "
                f"using {X.shape[1]} numeric columns only."
            )

            st.session_state["baseline_acc"] = baseline_acc

        # ------------------------------------------------------------------
        # STEP 3: Missing value handling
        # ------------------------------------------------------------------
        st.header("2️⃣ Missing Value Handling")

        total_missing_before = int(df.isnull().sum().sum())
        st.write(f"Total missing values in raw data: **{total_missing_before}**")

        strategy = st.radio(
            "Choose how to handle missing values:",
            options=["Drop rows with any missing value", "Mean (numeric) / Mode (categorical)",
                      "Median (numeric) / Mode (categorical)"],
            index=1,
        )

        eng_df = df.copy()

        if strategy == "Drop rows with any missing value":
            eng_df = eng_df.dropna()
        else:
            numeric_cols = eng_df.select_dtypes(include="number").columns
            categorical_cols = eng_df.select_dtypes(exclude="number").columns

            for col in numeric_cols:
                if eng_df[col].isnull().any():
                    fill_value = eng_df[col].mean() if "Mean" in strategy else eng_df[col].median()
                    eng_df[col] = eng_df[col].fillna(fill_value)

            for col in categorical_cols:
                if eng_df[col].isnull().any():
                    mode_value = eng_df[col].mode(dropna=True)
                    if len(mode_value) > 0:
                        eng_df[col] = eng_df[col].fillna(mode_value[0])
                    else:
                        eng_df = eng_df.dropna(subset=[col])

        total_missing_after = int(eng_df.isnull().sum().sum())

        if eng_df.shape[0] == 0:
            st.error(
                "❌ This strategy removed every row (every row had at least one missing value "
                "somewhere). Try 'Mean/Median + Mode' instead of dropping rows."
            )
            st.stop()

        col_a, col_b = st.columns(2)
        col_a.metric("Missing values before", total_missing_before)
        col_b.metric("Missing values after", total_missing_after)

        st.caption(f"Rows: {df.shape[0]} → {eng_df.shape[0]} after handling missing values.")

        st.session_state["eng_df"] = eng_df

        # ------------------------------------------------------------------
        # STEP 4: Outlier detection & treatment
        # ------------------------------------------------------------------
        st.header("3️⃣ Outlier Detection & Treatment")

        import numpy as np

        outlier_numeric_cols = [c for c in eng_df.select_dtypes(include="number").columns if c != target_col]

        if len(outlier_numeric_cols) == 0:
            st.info("No numeric feature columns to check for outliers.")
        else:
            detect_method = st.radio(
                "Choose outlier detection method:",
                options=["IQR (Interquartile Range)", "Z-score"],
                index=0,
            )

            outlier_mask = pd.DataFrame(False, index=eng_df.index, columns=outlier_numeric_cols)

            if detect_method == "IQR (Interquartile Range)":
                for col in outlier_numeric_cols:
                    q1 = eng_df[col].quantile(0.25)
                    q3 = eng_df[col].quantile(0.75)
                    iqr = q3 - q1
                    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    outlier_mask[col] = (eng_df[col] < lower) | (eng_df[col] > upper)
            else:
                for col in outlier_numeric_cols:
                    mean, std = eng_df[col].mean(), eng_df[col].std()
                    if std > 0:
                        z_scores = (eng_df[col] - mean) / std
                        outlier_mask[col] = z_scores.abs() > 3

            rows_with_outliers = outlier_mask.any(axis=1).sum()
            st.write(f"Detected **{rows_with_outliers}** row(s) with at least one outlier "
                      f"across {len(outlier_numeric_cols)} numeric column(s) using {detect_method}.")

            if rows_with_outliers > 0:
                treatment = st.radio(
                    "Choose outlier treatment:",
                    options=["Do nothing", "Remove rows", "Cap (clip to bounds)", "Winsorize (5th/95th percentile)"],
                    index=0,
                )

                if treatment == "Remove rows":
                    eng_df = eng_df[~outlier_mask.any(axis=1)].reset_index(drop=True)
                    st.caption(f"Removed {rows_with_outliers} row(s). New row count: {eng_df.shape[0]}")
                elif treatment == "Cap (clip to bounds)":
                    for col in outlier_numeric_cols:
                        q1 = eng_df[col].quantile(0.25)
                        q3 = eng_df[col].quantile(0.75)
                        iqr = q3 - q1
                        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                        eng_df[col] = eng_df[col].clip(lower, upper)
                    st.caption("Capped outlier values to IQR bounds (values beyond the bound are set to the bound).")
                elif treatment == "Winsorize (5th/95th percentile)":
                    for col in outlier_numeric_cols:
                        lower = eng_df[col].quantile(0.05)
                        upper = eng_df[col].quantile(0.95)
                        eng_df[col] = eng_df[col].clip(lower, upper)
                    st.caption("Winsorized values below 5th percentile / above 95th percentile.")
                else:
                    st.caption("No treatment applied — outliers left as-is.")

            if eng_df.shape[0] == 0:
                st.error("❌ Outlier removal eliminated all rows. Try Cap or Winsorize instead of Remove.")
                st.stop()

        st.session_state["eng_df"] = eng_df

        # ------------------------------------------------------------------
        # STEP 5: Feature scaling
        # ------------------------------------------------------------------
        st.header("4️⃣ Feature Scaling")

        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        import matplotlib.pyplot as plt

        numeric_feature_cols = [c for c in eng_df.select_dtypes(include="number").columns if c != target_col]

        if len(numeric_feature_cols) == 0:
            st.info("No numeric feature columns to scale (target excluded).")
        else:
            scale_method = st.radio(
                "Choose a scaling method:",
                options=["Standardization (Z-score)", "Normalization (Min-Max)", "Robust Scaling (median/IQR)"],
                index=0,
            )

            if scale_method == "Standardization (Z-score)":
                scaler = StandardScaler()
            elif scale_method == "Normalization (Min-Max)":
                scaler = MinMaxScaler()
            else:
                scaler = RobustScaler()

            before_values = eng_df[numeric_feature_cols].copy()
            eng_df[numeric_feature_cols] = scaler.fit_transform(eng_df[numeric_feature_cols])

            plot_col = st.selectbox("Preview distribution for column:", options=numeric_feature_cols)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
            ax1.hist(before_values[plot_col], bins=30, color="#4C78A8")
            ax1.set_title(f"{plot_col} — Before")
            ax2.hist(eng_df[plot_col], bins=30, color="#F58518")
            ax2.set_title(f"{plot_col} — After ({scale_method.split()[0]})")
            st.pyplot(fig)

            st.caption(f"Scaled {len(numeric_feature_cols)} numeric feature columns using {scale_method}.")

        st.session_state["eng_df"] = eng_df

        # ------------------------------------------------------------------
        # STEP 5: Categorical encoding
        # ------------------------------------------------------------------
        st.header("5️⃣ Categorical Encoding")

        categorical_feature_cols = [c for c in eng_df.select_dtypes(exclude="number").columns if c != target_col]

        if len(categorical_feature_cols) == 0:
            st.info("No categorical feature columns to encode.")
        else:
            encode_method = st.radio(
                "Choose an encoding method:",
                options=["One-Hot Encoding", "Label Encoding"],
                index=0,
            )

            HIGH_CARDINALITY_THRESHOLD = 15
            high_card_cols = [c for c in categorical_feature_cols if eng_df[c].nunique() > HIGH_CARDINALITY_THRESHOLD]

            if encode_method == "One-Hot Encoding" and high_card_cols:
                st.warning(
                    f"⚠️ These columns have more than {HIGH_CARDINALITY_THRESHOLD} unique values: "
                    f"{', '.join(high_card_cols)}. One-hot encoding them will explode column count "
                    f"(e.g. a column with 200 unique values → 200 new columns). Consider Label Encoding "
                    f"instead for these, or dropping them."
                )

            if encode_method == "One-Hot Encoding":
                eng_df = pd.get_dummies(eng_df, columns=categorical_feature_cols, drop_first=True)
            else:
                from sklearn.preprocessing import LabelEncoder
                for col in categorical_feature_cols:
                    le_col = LabelEncoder()
                    eng_df[col] = le_col.fit_transform(eng_df[col].astype(str))

            st.write(f"Encoded {len(categorical_feature_cols)} categorical column(s) using {encode_method}.")
            st.caption(f"Column count: {df.shape[1]} (raw) → {eng_df.shape[1]} (after encoding)")

            st.subheader("Preview of Transformed Columns")
            st.dataframe(eng_df.head(10))

        st.session_state["eng_df"] = eng_df

        # ------------------------------------------------------------------
        # Encode target column in eng_df (needed for heatmap + final model)
        # ------------------------------------------------------------------
        if not pd.api.types.is_numeric_dtype(eng_df[target_col]):
            from sklearn.preprocessing import LabelEncoder
            le_target = LabelEncoder()
            eng_df[target_col] = le_target.fit_transform(eng_df[target_col].astype(str))
            st.session_state["target_classes"] = le_target.classes_

        st.session_state["eng_df"] = eng_df

        # ------------------------------------------------------------------
        # STEP 6: Correlation heatmap
        # ------------------------------------------------------------------
        st.header("6️⃣ Correlation Heatmap (Engineered Data)")

        import seaborn as sns

        corr_df = eng_df.select_dtypes(include="number")

        if corr_df.shape[1] < 2:
            st.info("Not enough numeric columns to compute correlations.")
        else:
            corr_matrix = corr_df.corr()

            fig_size = max(6, min(0.4 * corr_matrix.shape[1], 20))
            fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))
            sns.heatmap(corr_matrix, cmap="coolwarm", center=0, ax=ax,
                        annot=corr_matrix.shape[1] <= 12, fmt=".2f",
                        cbar_kws={"shrink": 0.8})
            st.pyplot(fig)

            st.caption(
                f"Showing correlations across {corr_matrix.shape[1]} numeric columns "
                f"(annotations hidden above 12 columns for readability)."
            )

            if target_col in corr_matrix.columns:
                target_corr = corr_matrix[target_col].drop(target_col).sort_values(key=abs, ascending=False)
                st.subheader(f"Top correlated features with '{target_col}'")
                st.dataframe(target_corr.head(10).rename("correlation"))

        # ------------------------------------------------------------------
        # STEP 7: Final model + before/after comparison
        # ------------------------------------------------------------------
        st.header("7️⃣ Feature Selection")

        low_var_check_cols = [c for c in eng_df.select_dtypes(include="number").columns if c != target_col]
        constant_cols = [c for c in low_var_check_cols if eng_df[c].nunique() <= 1]
        if constant_cols:
            st.warning(f"⚠️ Constant feature(s) detected (no variation, zero predictive value): "
                        f"{', '.join(constant_cols)}. Consider removing them via Variance Threshold below.")


        from sklearn.feature_selection import (VarianceThreshold, SelectKBest,
                                                 f_classif, mutual_info_classif, RFE)

        fs_feature_cols = [c for c in eng_df.select_dtypes(include="number").columns if c != target_col]

        if len(fs_feature_cols) < 2:
            st.info("Not enough numeric feature columns to run feature selection.")
        else:
            fs_method = st.radio(
                "Choose a feature selection method:",
                options=["None (keep all features)", "Variance Threshold", "Correlation-based removal",
                          "SelectKBest (ANOVA F-test)", "Mutual Information", "RFE (Recursive Feature Elimination)"],
                index=0,
            )

            X_fs = eng_df[fs_feature_cols]
            y_fs = eng_df[target_col]
            selected_cols = fs_feature_cols  # default: keep everything

            if fs_method == "Variance Threshold":
                threshold = st.slider("Minimum variance to keep a feature", 0.0, 1.0, 0.01, 0.01)
                vt = VarianceThreshold(threshold=threshold)
                try:
                    vt.fit(X_fs)
                    selected_cols = list(X_fs.columns[vt.get_support()])
                except Exception as e:
                    st.warning(f"Could not apply Variance Threshold: {e}")

            elif fs_method == "Correlation-based removal":
                corr_threshold = st.slider("Correlation threshold (drop one of any pair above this)", 0.5, 1.0, 0.9, 0.05)
                corr_matrix_fs = X_fs.corr().abs()
                upper = corr_matrix_fs.where(np.triu(np.ones(corr_matrix_fs.shape), k=1).astype(bool))
                to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
                selected_cols = [c for c in fs_feature_cols if c not in to_drop]
                if to_drop:
                    st.caption(f"Dropped {len(to_drop)} redundant column(s): {', '.join(to_drop)}")

            elif fs_method == "SelectKBest (ANOVA F-test)":
                k = st.slider("Number of top features to keep (K)", 1, len(fs_feature_cols), min(5, len(fs_feature_cols)))
                try:
                    skb = SelectKBest(score_func=f_classif, k=k)
                    skb.fit(X_fs, y_fs)
                    selected_cols = list(X_fs.columns[skb.get_support()])
                except Exception as e:
                    st.warning(f"Could not apply SelectKBest: {e}")

            elif fs_method == "Mutual Information":
                k = st.slider("Number of top features to keep (K)", 1, len(fs_feature_cols), min(5, len(fs_feature_cols)))
                try:
                    skb_mi = SelectKBest(score_func=mutual_info_classif, k=k)
                    skb_mi.fit(X_fs, y_fs)
                    selected_cols = list(X_fs.columns[skb_mi.get_support()])
                except Exception as e:
                    st.warning(f"Could not apply Mutual Information selection: {e}")

            elif fs_method == "RFE (Recursive Feature Elimination)":
                n_features = st.slider("Number of features to keep", 1, len(fs_feature_cols), min(5, len(fs_feature_cols)))
                try:
                    rfe_model = build_model(model_choice)
                    if not hasattr(rfe_model, "coef_") and not hasattr(rfe_model, "feature_importances_"):
                        st.warning(
                            f"{model_choice} doesn't expose feature weights natively, so RFE will use a "
                            f"Decision Tree internally as a stand-in for ranking features."
                        )
                        from sklearn.tree import DecisionTreeClassifier
                        rfe_model = DecisionTreeClassifier(random_state=42)
                    rfe = RFE(estimator=rfe_model, n_features_to_select=n_features)
                    rfe.fit(X_fs, y_fs)
                    selected_cols = list(X_fs.columns[rfe.support_])
                except Exception as e:
                    st.warning(f"Could not apply RFE: {e}")

            if fs_method != "None (keep all features)":
                st.write(f"Selected **{len(selected_cols)}** of {len(fs_feature_cols)} feature(s): "
                          f"{', '.join(selected_cols) if selected_cols else '(none)'}")
                if len(selected_cols) == 0:
                    st.error("❌ No features survived selection — try a less aggressive threshold/K.")
                else:
                    eng_df = eng_df[selected_cols + [target_col]]
                    st.caption(f"eng_df now has {eng_df.shape[1]} columns (features + target).")

        st.session_state["eng_df"] = eng_df

        # ------------------------------------------------------------------
        # STEP 8: Final model + before/after comparison
        # ------------------------------------------------------------------
        st.header("8️⃣ Final Model (Engineered Data)")

        final_df = eng_df.copy()

        if final_df.shape[0] < 20 or final_df.shape[1] < 2:
            st.error("❌ Not enough usable data to train the final model.")
        else:
            X_final = final_df.drop(columns=[target_col])
            y_final = final_df[target_col]

            stratify_arg = y_final if (y_final.nunique() < 20 and y_final.value_counts().min() >= 2) else None

            X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
                X_final, y_final, test_size=0.2, random_state=42, stratify=stratify_arg
            )

            final_model = build_model(model_choice)
            final_model.fit(X_train_f, y_train_f)
            final_preds = final_model.predict(X_test_f)
            final_acc = accuracy_score(y_test_f, final_preds)

            st.session_state["final_acc"] = final_acc

            # --------------------------------------------------------------
            # Detailed evaluation metrics (final model only)
            # --------------------------------------------------------------
            from sklearn.metrics import (precision_score, recall_score, f1_score,
                                          confusion_matrix, classification_report, roc_auc_score)
            from sklearn.model_selection import cross_val_score

            n_classes = y_final.nunique()
            avg_method = "binary" if n_classes == 2 else "weighted"

            precision = precision_score(y_test_f, final_preds, average=avg_method, zero_division=0)
            recall = recall_score(y_test_f, final_preds, average=avg_method, zero_division=0)
            f1 = f1_score(y_test_f, final_preds, average=avg_method, zero_division=0)

            st.subheader("📊 Detailed Evaluation Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{final_acc:.2%}")
            m2.metric("Precision", f"{precision:.2%}")
            m3.metric("Recall", f"{recall:.2%}")
            m4.metric("F1 Score", f"{f1:.2%}")

            if n_classes == 2 and hasattr(final_model, "predict_proba"):
                try:
                    y_proba = final_model.predict_proba(X_test_f)[:, 1]
                    roc_auc = roc_auc_score(y_test_f, y_proba)
                    st.metric("ROC-AUC", f"{roc_auc:.3f}")
                except Exception:
                    pass

            st.write("**Confusion Matrix**")
            cm = confusion_matrix(y_test_f, final_preds)
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

            with st.expander("Full Classification Report"):
                st.text(classification_report(y_test_f, final_preds, zero_division=0))

            with st.expander("5-Fold Cross-Validation Score"):
                try:
                    cv_scores = cross_val_score(build_model(model_choice), X_final, y_final, cv=5)
                    st.write(f"Mean CV Accuracy: **{cv_scores.mean():.2%}** (± {cv_scores.std():.2%})")
                    st.write(f"Individual folds: {', '.join(f'{s:.2%}' for s in cv_scores)}")
                except Exception as e:
                    st.caption(f"Cross-validation not available for this data/model combination: {e}")

            st.subheader("📈 Before vs After: Accuracy Comparison")

            baseline_acc = st.session_state.get("baseline_acc")

            if baseline_acc is None:
                st.warning(
                    "Baseline accuracy wasn't computed (likely because the naive baseline had no usable "
                    "numeric columns). Showing engineered accuracy only."
                )
                st.metric("Engineered Model Accuracy", f"{final_acc:.2%}")
            else:
                col_x, col_y, col_z = st.columns(3)
                col_x.metric(f"Baseline Accuracy ({model_choice})", f"{baseline_acc:.2%}")
                col_y.metric(f"Engineered Accuracy ({model_choice})", f"{final_acc:.2%}")
                delta = final_acc - baseline_acc
                col_z.metric("Improvement", f"{delta:+.2%}")

                if delta > 0:
                    st.success(
                        f"✅ Feature engineering improved accuracy by {delta:.2%} — "
                        f"from {baseline_acc:.2%} to {final_acc:.2%}."
                    )
                elif delta < 0:
                    st.warning(
                        f"⚠️ Engineered accuracy is actually {abs(delta):.2%} lower than baseline. "
                        f"This can happen — e.g. one-hot encoding adding noisy sparse columns, or the "
                        f"baseline getting lucky on a small test split. Try a different encoding method "
                        f"or scaling method above and see if it changes."
                    )
                else:
                    st.info("No change in accuracy — try a different scaling/encoding combination.")

            st.session_state["eng_df"] = eng_df

        # ------------------------------------------------------------------
        # STEP 9: Model interpretation, comparison, and export
        # ------------------------------------------------------------------
        st.header("9️⃣ Model Interpretation & Comparison")

        if final_df.shape[0] >= 20 and final_df.shape[1] >= 2:
            st.subheader("Feature Importance")
            feature_names = X_final.columns.tolist()

            if hasattr(final_model, "feature_importances_"):
                importances = final_model.feature_importances_
                imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}) \
                    .sort_values("Importance", ascending=False)
                st.dataframe(imp_df)
            elif hasattr(final_model, "coef_"):
                coefs = final_model.coef_[0] if final_model.coef_.ndim > 1 else final_model.coef_
                coef_df = pd.DataFrame({"Feature": feature_names, "Coefficient": coefs}) \
                    .sort_values("Coefficient", key=abs, ascending=False)
                st.dataframe(coef_df)
                st.caption("Positive coefficients push toward the positive class; negative push away from it.")
            else:
                st.caption(f"{model_choice} doesn't expose feature importances or coefficients directly.")

            st.subheader("Model Comparison (All Algorithms, Same Engineered Data)")
            if st.button("Run comparison across all models"):
                comparison_rows = []
                all_model_names = ["Logistic Regression", "Decision Tree", "Random Forest",
                                    "K-Nearest Neighbors", "Support Vector Machine", "Naive Bayes"]
                progress = st.progress(0)
                for i, name in enumerate(all_model_names):
                    try:
                        m = build_model(name)
                        m.fit(X_train_f, y_train_f)
                        preds = m.predict(X_test_f)
                        comparison_rows.append({
                            "Model": name,
                            "Accuracy": accuracy_score(y_test_f, preds),
                            "F1 Score": f1_score(y_test_f, preds, average=avg_method, zero_division=0),
                        })
                    except Exception:
                        comparison_rows.append({"Model": name, "Accuracy": None, "F1 Score": None})
                    progress.progress((i + 1) / len(all_model_names))

                comp_df = pd.DataFrame(comparison_rows).sort_values("Accuracy", ascending=False, na_position="last")
                st.dataframe(comp_df.style.format({"Accuracy": "{:.2%}", "F1 Score": "{:.2%}"}))

                best_valid = comp_df.dropna()
                if not best_valid.empty:
                    best_row = best_valid.iloc[0]
                    st.success(f"🏆 Best performing model on this data: **{best_row['Model']}** "
                                f"({best_row['Accuracy']:.2%} accuracy)")

            st.subheader("Download Trained Model")
            import joblib
            import io
            model_buffer = io.BytesIO()
            joblib.dump(final_model, model_buffer)
            model_buffer.seek(0)
            st.download_button(
                label="⬇️ Download trained model (.pkl)",
                data=model_buffer,
                file_name=f"{model_choice.lower().replace(' ', '_')}_model.pkl",
                mime="application/octet-stream",
            )
            st.caption(
                "Load it later with: `import joblib; model = joblib.load('model.pkl')`. "
                "You'll need to apply the same preprocessing (scaling/encoding) to new data "
                "before calling `model.predict()`."
            )

        # ------------------------------------------------------------------
        # STEP 10: Download engineered CSV
        # ------------------------------------------------------------------
        st.header("🔟 Download Engineered Dataset")

        csv_bytes = eng_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️ Download engineered CSV",
            data=csv_bytes,
            file_name="engineered_dataset.csv",
            mime="text/csv",
        )

        st.caption(
            f"Final shape: {eng_df.shape[0]} rows × {eng_df.shape[1]} columns "
            f"(started as {df.shape[0]} rows × {df.shape[1]} columns raw)."
        )

    else:
        st.stop()

else:
    st.info("👆 Upload a CSV to get started.")
