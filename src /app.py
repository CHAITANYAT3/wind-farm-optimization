import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(page_title="Wind Farm Optimization", page_icon="🌬️", layout="wide")

TARGET = "power_29_avg"
PREFERRED_FEATURES = [
    "wind_speed_3_avg",
    "wind_speed_4_avg",
    "sensor_15_avg",
    "sensor_16_avg",
    "sensor_23_avg",
    "sensor_24_avg",
    "sensor_25_avg",
    "sensor_45",
    "sensor_50",
]

st.title("🌬️ Wind Farm Renewable Energy Optimization")
st.write(
    "Upload a wind-farm dataset to explore its data, compare four regression models, "
    "and inspect power-generation patterns. The default target column is "
    f"`{TARGET}`."
)

uploaded_file = st.file_uploader(
    "Upload your wind-farm dataset", type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info("Upload a CSV or Excel dataset to begin. The dataset must contain the target column `power_29_avg`.")
    st.stop()

try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()

df.columns = [str(c).strip() for c in df.columns]
st.subheader("1. Dataset overview")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{len(df.columns):,}")
c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
st.dataframe(df.head(20), use_container_width=True)

if TARGET not in df.columns:
    st.error(f"Required target column `{TARGET}` was not found. Check your dataset's column names.")
    st.write("Available columns:", df.columns.tolist())
    st.stop()

# Convert known numeric values; preserve the original timestamp separately for line plots.
numeric_df = df.copy()
for col in numeric_df.columns:
    if col != "time_stamp":
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="ignore")

numeric_columns = numeric_df.select_dtypes(include=np.number).columns.tolist()
if TARGET not in numeric_columns:
    numeric_df[TARGET] = pd.to_numeric(numeric_df[TARGET], errors="coerce")
    numeric_columns = numeric_df.select_dtypes(include=np.number).columns.tolist()

st.subheader("2. Exploratory data analysis (EDA)")
if not numeric_columns:
    st.warning("No numeric columns were found in the uploaded dataset.")
else:
    default_target_idx = numeric_columns.index(TARGET) if TARGET in numeric_columns else 0

    left, right = st.columns(2)
    with left:
        st.markdown("**Histogram**")
        hist_col = st.selectbox("Variable for histogram", numeric_columns, index=default_target_idx, key="hist")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(pd.to_numeric(numeric_df[hist_col], errors="coerce").dropna(), bins=30, edgecolor="black")
        ax.set(title=f"Histogram of {hist_col}", xlabel=hist_col, ylabel="Frequency")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.markdown("**Box plot**")
        box_col = st.selectbox("Variable for box plot", numeric_columns, index=default_target_idx, key="box")
        fig, ax = plt.subplots(figsize=(7, 4))
        vals = pd.to_numeric(numeric_df[box_col], errors="coerce").dropna()
        ax.boxplot(vals, vert=True)
        ax.set(title=f"Box plot of {box_col}", ylabel=box_col)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    feature_candidates = [c for c in numeric_columns if c != TARGET]
    if feature_candidates:
        left, right = st.columns(2)
        with left:
            st.markdown("**Bar chart**")
            bar_col = st.selectbox(
                "Variable to group for bar chart",
                feature_candidates,
                index=feature_candidates.index("wind_speed_3_avg") if "wind_speed_3_avg" in feature_candidates else 0,
                key="bar",
            )
            bar_data = numeric_df[[bar_col, TARGET]].copy()
            bar_data[bar_col] = pd.to_numeric(bar_data[bar_col], errors="coerce")
            bar_data[TARGET] = pd.to_numeric(bar_data[TARGET], errors="coerce")
            bar_data = bar_data.dropna()
            if len(bar_data) >= 2 and bar_data[bar_col].nunique() >= 2:
                try:
                    bar_data["Range"] = pd.qcut(bar_data[bar_col], q=min(10, bar_data[bar_col].nunique()), duplicates="drop")
                    summary = bar_data.groupby("Range", observed=True)[TARGET].mean()
                    fig, ax = plt.subplots(figsize=(8, 4))
                    summary.plot(kind="bar", ax=ax)
                    ax.set(title=f"Average {TARGET} by {bar_col} range", xlabel=f"{bar_col} range", ylabel=f"Mean {TARGET}")
                    plt.xticks(rotation=45, ha="right")
                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                except ValueError:
                    st.info("Not enough distinct values to create bar-chart ranges.")
            else:
                st.info("Not enough valid data to create the bar chart.")

        with right:
            st.markdown("**Scatter plot**")
            scatter_col = st.selectbox(
                "X-axis variable for scatter plot",
                feature_candidates,
                index=feature_candidates.index("wind_speed_3_avg") if "wind_speed_3_avg" in feature_candidates else 0,
                key="scatter",
            )
            scatter_data = numeric_df[[scatter_col, TARGET]].apply(pd.to_numeric, errors="coerce").dropna()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(scatter_data[scatter_col], scatter_data[TARGET], alpha=0.5)
            ax.set(title=f"{scatter_col} vs {TARGET}", xlabel=scatter_col, ylabel=TARGET)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.markdown("**Line graph**")
    line_data = df.copy()
    line_data[TARGET] = pd.to_numeric(line_data[TARGET], errors="coerce")
    if "time_stamp" in line_data.columns:
        line_data["time_stamp"] = pd.to_datetime(line_data["time_stamp"], errors="coerce")
        line_data = line_data.dropna(subset=["time_stamp", TARGET]).sort_values("time_stamp").tail(1000)
        x_values = line_data["time_stamp"]
        x_label = "Time"
    else:
        line_data = line_data.dropna(subset=[TARGET]).tail(1000)
        x_values = np.arange(len(line_data))
        x_label = "Record number"
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x_values, line_data[TARGET])
    ax.set(title="Wind power generation over time/records", xlabel=x_label, ylabel=TARGET)
    plt.xticks(rotation=30)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("**Correlation heatmap**")
    heatmap_candidates = [c for c in PREFERRED_FEATURES + [TARGET] if c in numeric_columns]
    if len(heatmap_candidates) < 2:
        heatmap_candidates = numeric_columns[: min(12, len(numeric_columns))]
    if len(heatmap_candidates) >= 2:
        corr = numeric_df[heatmap_candidates].apply(pd.to_numeric, errors="coerce").corr()
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation heatmap")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("At least two numeric columns are needed for a correlation heatmap.")

st.subheader("3. Train and compare four regression models")

available_preferred = [c for c in PREFERRED_FEATURES if c in df.columns and c != TARGET]
if available_preferred:
    default_features = available_preferred
else:
    # Avoid obvious identifiers, timestamps, and power-derived columns to reduce leakage.
    default_features = [
        c for c in numeric_columns
        if c != TARGET
        and c.lower() not in {"id", "asset_id", "train_test"}
        and not c.lower().startswith("power_")
    ][:10]

features = st.multiselect(
    "Select input features (power-derived columns are excluded by default to reduce target leakage)",
    options=[c for c in numeric_columns if c != TARGET],
    default=default_features,
)
test_size = st.slider("Test data percentage", min_value=10, max_value=40, value=20, step=5)
random_state = st.number_input("Random seed", min_value=0, max_value=99999, value=42, step=1)

if st.button("Train models", type="primary"):
    if not features:
        st.error("Select at least one input feature.")
        st.stop()

    modeling = df[features + [TARGET]].copy()
    for col in features + [TARGET]:
        modeling[col] = pd.to_numeric(modeling[col], errors="coerce")
    modeling = modeling.replace([np.inf, -np.inf], np.nan).dropna()

    if len(modeling) < 10:
        st.error("At least 10 complete rows are required to train and evaluate the models.")
        st.stop()
    if modeling[TARGET].nunique() < 2:
        st.error("The target column must contain at least two distinct numeric values.")
        st.stop()

    X = modeling[features].to_numpy(dtype=float)
    y = modeling[TARGET].to_numpy(dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size / 100, random_state=int(random_state)
    )

    # Single-feature Linear Regression; Multiple Linear Regression uses all selected features.
    one_feature_idx = features.index("wind_speed_3_avg") if "wind_speed_3_avg" in features else 0
    X_single = X[:, [one_feature_idx]]
    X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(
        X_single, y, test_size=test_size / 100, random_state=int(random_state)
    )

    models = {
        "Linear Regression": (LinearRegression(), X_train_single, X_test_single, y_train_single, y_test_single),
        "Multiple Linear Regression": (LinearRegression(), X_train, X_test, y_train, y_test),
        "Decision Tree Regression": (DecisionTreeRegressor(max_depth=15, random_state=int(random_state)), X_train, X_test, y_train, y_test),
        "Random Forest Regression": (RandomForestRegressor(n_estimators=100, max_depth=15, random_state=int(random_state), n_jobs=-1), X_train, X_test, y_train, y_test),
    }

    # Median threshold defines high generation for Accuracy/F1. This is a derived
    # binary classification view of the regression output, not regression accuracy.
    threshold = float(np.median(y_train))
    results = []
    prediction_frames = []
    fitted_models = {}

    with st.spinner("Training and evaluating models..."):
        for name, (model, xtr, xte, ytr, yte) in models.items():
            model.fit(xtr, ytr)
            pred = model.predict(xte)
            actual_class = (yte >= threshold).astype(int)
            predicted_class = (pred >= threshold).astype(int)

            mse = mean_squared_error(yte, pred)
            results.append({
                "Model": name,
                "MAE": mean_absolute_error(yte, pred),
                "MSE": mse,
                "RMSE": float(np.sqrt(mse)),
                "R2 Score": r2_score(yte, pred),
                "Accuracy": accuracy_score(actual_class, predicted_class),
                "F1 Score": f1_score(actual_class, predicted_class, zero_division=0),
            })
            prediction_frames.append(pd.DataFrame({
                "Model": name,
                "Actual Power": yte,
                "Predicted Power": pred,
                "Actual Class": actual_class,
                "Predicted Class": predicted_class,
            }))
            fitted_models[name] = model

    results_df = pd.DataFrame(results).sort_values("R2 Score", ascending=False)
    st.session_state["model_results"] = results_df
    st.session_state["predictions"] = pd.concat(prediction_frames, ignore_index=True)
    st.session_state["feature_importance"] = pd.DataFrame()
    st.session_state["threshold"] = threshold
    st.session_state["fitted_models"] = fitted_models
    st.session_state["features"] = features
    st.session_state["train_data"] = modeling
    st.success("All four models trained successfully.")

if "model_results" in st.session_state:
    st.subheader("4. Model comparison")
    results_df = st.session_state["model_results"]
    st.caption(
        f"High-generation threshold (median of training target): {st.session_state['threshold']:.4f}. "
        "Accuracy and F1 Score are calculated after classifying power as high (1) or normal/low (0)."
    )
    st.dataframe(results_df.round(4), use_container_width=True)
    st.download_button(
        "Download model comparison CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison_results.csv",
        mime="text/csv",
    )

    best_r2 = results_df.iloc[0]
    best_accuracy = results_df.loc[results_df["Accuracy"].idxmax()]
    best_f1 = results_df.loc[results_df["F1 Score"].idxmax()]
    a, b, c = st.columns(3)
    a.metric("Best regression model (R²)", str(best_r2["Model"]), f"R² {best_r2['R2 Score']:.3f}")
    b.metric("Best by Accuracy", str(best_accuracy["Model"]), f"{best_accuracy['Accuracy']:.1%}")
    c.metric("Best by F1 Score", str(best_f1["Model"]), f"{best_f1['F1 Score']:.3f}")

    st.subheader("5. Actual vs predicted power")
    prediction_data = st.session_state["predictions"]
    selected_model = st.selectbox("Select model to inspect", prediction_data["Model"].unique(), key="pred_model")
    chosen = prediction_data[prediction_data["Model"] == selected_model]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(chosen["Actual Power"], chosen["Predicted Power"], alpha=0.6)
    low = min(chosen["Actual Power"].min(), chosen["Predicted Power"].min())
    high = max(chosen["Actual Power"].max(), chosen["Predicted Power"].max())
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set(title=f"Actual vs predicted power: {selected_model}", xlabel="Actual power", ylabel="Predicted power")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.download_button(
        "Download all test predictions CSV",
        data=prediction_data.to_csv(index=False).encode("utf-8"),
        file_name="model_predictions.csv",
        mime="text/csv",
    )

    rf = st.session_state["fitted_models"]["Random Forest Regression"]
    selected_features = st.session_state["features"]
    if hasattr(rf, "feature_importances_"):
        importance = pd.DataFrame({
            "Feature": selected_features,
            "Importance": rf.feature_importances_,
        }).sort_values("Importance", ascending=False)
        st.subheader("6. Random Forest feature importance")
        st.dataframe(importance.round(4), use_container_width=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        importance.sort_values("Importance").plot.barh(x="Feature", y="Importance", legend=False, ax=ax)
        ax.set(title="Feature importance", xlabel="Importance", ylabel="Feature")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.download_button(
            "Download feature importance CSV",
            data=importance.to_csv(index=False).encode("utf-8"),
            file_name="feature_importance.csv",
            mime="text/csv",
        )

    st.subheader("7. Wind-generation optimization insights")
    rf_importance = pd.DataFrame({
        "Feature": selected_features,
        "Importance": st.session_state["fitted_models"]["Random Forest Regression"].feature_importances_,
    }).sort_values("Importance", ascending=False)
    top_features = rf_importance.head(3)["Feature"].tolist()
    st.write(f"Most influential input features in this Random Forest model: **{', '.join(top_features)}**.")
    st.write(
        "Use these results to investigate which operating conditions are associated with higher predicted power. "
        "These are observational model insights, not proof that changing a sensor or turbine setting will increase generation. "
        "Validate any operational change with engineering constraints and real-world testing."
    )

st.caption("Prototype for academic research. Model results depend on dataset quality, feature selection, and operating conditions.")
