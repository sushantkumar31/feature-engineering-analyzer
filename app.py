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
    elif n_missing > 0.5 * n_rows:
        st.warning(f"⚠️ This column is {n_missing / n_rows:.0%} missing — results will be unreliable.")

    if valid_target:
        st.session_state["df"] = df
        st.session_state["target_col"] = target_col
        st.success(f"Target set to: **{target_col}**")

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

            stratify_arg = y if y.nunique() < 20 else None

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=stratify_arg
            )

            baseline_model = LogisticRegression(max_iter=1000)
            baseline_model.fit(X_train, y_train)
            baseline_preds = baseline_model.predict(X_test)
            baseline_acc = accuracy_score(y_test, baseline_preds)

            st.metric("Baseline Accuracy (raw, naive)", f"{baseline_acc:.2%}")
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

        col_a, col_b = st.columns(2)
        col_a.metric("Missing values before", total_missing_before)
        col_b.metric("Missing values after", total_missing_after)

        st.caption(f"Rows: {df.shape[0]} → {eng_df.shape[0]} after handling missing values.")

        st.session_state["eng_df"] = eng_df

        # ------------------------------------------------------------------
        # STEP 4: Feature scaling
        # ------------------------------------------------------------------
        st.header("3️⃣ Feature Scaling")

        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        import matplotlib.pyplot as plt

        numeric_feature_cols = [c for c in eng_df.select_dtypes(include="number").columns if c != target_col]

        if len(numeric_feature_cols) == 0:
            st.info("No numeric feature columns to scale (target excluded).")
        else:
            scale_method = st.radio(
                "Choose a scaling method:",
                options=["Standardization (Z-score)", "Normalization (Min-Max)"],
                index=0,
            )

            scaler = StandardScaler() if scale_method == "Standardization (Z-score)" else MinMaxScaler()

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
        st.header("4️⃣ Categorical Encoding")

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
        st.header("5️⃣ Correlation Heatmap (Engineered Data)")

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
        st.header("6️⃣ Final Model (Engineered Data)")

        final_df = eng_df.copy()

        if final_df.shape[0] < 20 or final_df.shape[1] < 2:
            st.error("❌ Not enough usable data to train the final model.")
        else:
            X_final = final_df.drop(columns=[target_col])
            y_final = final_df[target_col]

            stratify_arg = y_final if y_final.nunique() < 20 else None

            X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
                X_final, y_final, test_size=0.2, random_state=42, stratify=stratify_arg
            )

            final_model = LogisticRegression(max_iter=1000)
            final_model.fit(X_train_f, y_train_f)
            final_preds = final_model.predict(X_test_f)
            final_acc = accuracy_score(y_test_f, final_preds)

            st.session_state["final_acc"] = final_acc

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
                col_x.metric("Baseline Accuracy", f"{baseline_acc:.2%}")
                col_y.metric("Engineered Accuracy", f"{final_acc:.2%}")
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
        # STEP 8: Download engineered CSV
        # ------------------------------------------------------------------
        st.header("7️⃣ Download Engineered Dataset")

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
