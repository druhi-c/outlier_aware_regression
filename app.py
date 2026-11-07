from __future__ import annotations

from pathlib import Path
import math
import re
from io import BytesIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.covariance import MinCovDet
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

ROOT = Path(__file__).resolve().parent
ASSET_ROOT = ROOT / "report_assets"
FIG_DIR = ASSET_ROOT / "figures"
TABLE_DIR = ASSET_ROOT / "tables"

st.set_page_config(
    page_title="Outlier-Aware House Price Regression",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 1.4rem; padding-bottom: 2.2rem; max-width: 1500px; }
[data-testid="stMetric"] { background: rgba(128, 128, 128, 0.12); border: 1px solid rgba(128, 128, 128, 0.28); padding: 0.9rem 1rem; border-radius: 12px; color: inherit; }
[data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] { color: inherit !important; }
[data-testid="stMetricLabel"] p, [data-testid="stMetricValue"] div { color: inherit !important; }
.small-note { color: #5f6b7a; font-size: 0.92rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_table(filename: str) -> pd.DataFrame:
    path = TABLE_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def has_table(filename: str) -> bool:
    return (TABLE_DIR / filename).exists()


def _rank01(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(len(values))
    fill = float(np.median(values[finite]))
    values = np.nan_to_num(values, nan=fill, posinf=float(np.max(values[finite])), neginf=float(np.min(values[finite])))
    return (rankdata(values, method="average") - 1) / max(len(values) - 1, 1)


def _training_reliability_weights(X: pd.DataFrame, y_log: np.ndarray, seed: int = 42):
    """Fit the notebook-style influence signals using training rows only."""
    numeric = X.select_dtypes(include=np.number).copy()
    numeric = numeric.loc[:, numeric.nunique(dropna=False) > 1]
    if numeric.shape[1] == 0:
        return np.ones(len(X)), {"mean_weight": 1.0, "floor_fraction": 0.0}
    Xi = SimpleImputer(strategy="median").fit_transform(numeric)
    Xs = RobustScaler().fit_transform(Xi)
    n = len(y_log)
    raw = {}
    iso = IsolationForest(n_estimators=150, contamination="auto", random_state=seed, n_jobs=-1).fit(Xs)
    raw["isolation_forest"] = -iso.score_samples(Xs)
    k = min(20, n - 1)
    lof = LocalOutlierFactor(n_neighbors=k, contamination="auto")
    lof.fit_predict(Xs)
    raw["lof"] = -lof.negative_outlier_factor_
    try:
        cov = MinCovDet(random_state=seed, support_fraction=0.75).fit(Xs)
        raw["robust_mahalanobis"] = cov.mahalanobis(Xs)
    except Exception:
        centered = Xs - np.median(Xs, axis=0)
        inv = np.linalg.pinv(np.cov(Xs, rowvar=False) + np.eye(Xs.shape[1]) * 1e-6)
        raw["robust_mahalanobis"] = np.maximum(np.einsum("ij,jk,ik->i", centered, inv, centered), 0)
    inner = KFold(n_splits=min(5, n), shuffle=True, random_state=seed)
    oof_pred = cross_val_predict(LinearRegression(), Xs, y_log, cv=inner, n_jobs=None)
    resid = y_log - oof_pred
    mad = max(float(1.4826 * np.median(np.abs(resid - np.median(resid)))), 1e-9)
    raw["oof_residual"] = np.abs(resid) / mad
    design = np.column_stack([np.ones(n), Xs])
    hat = np.sum(design * (design @ np.linalg.pinv(design.T @ design)), axis=1)
    ols = LinearRegression().fit(Xs, y_log)
    ols_resid = y_log - ols.predict(Xs)
    mse = np.sum(ols_resid ** 2) / max(n - design.shape[1], 1)
    p = design.shape[1]
    raw["leverage"] = hat
    raw["cooks_distance"] = (ols_resid**2 / (p * max(mse, 1e-12))) * (hat / np.maximum((1-hat)**2, 1e-12))
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(Xs).kneighbors(Xs, return_distance=False)[:, 1:]
    raw["knn_target"] = np.abs(y_log - y_log[neighbors].mean(axis=1))
    components = np.column_stack([_rank01(v) for v in raw.values()])
    score = pd.DataFrame(components).rank(pct=True, method="average").mean(axis=1).to_numpy()
    weights = np.clip(np.exp(-3.0 * score), 0.05, 1.0)
    return weights, {"mean_weight": float(weights.mean()),
                     "floor_fraction": float(np.isclose(weights, 0.05).mean()),
                     "mean_score": float(score.mean())}


@st.cache_resource(show_spinner="Training the selected model…")
def train_house_model(csv_bytes: bytes, model_kind: str):
    """Train a new estimator from an uploaded Ames-style CSV; cache it per input."""
    df = pd.read_csv(BytesIO(csv_bytes), low_memory=False)
    if "SalePrice" not in df.columns:
        raise ValueError("Training CSV must include a SalePrice column.")
    y = pd.to_numeric(df["SalePrice"], errors="coerce")
    valid = y.notna() & (y >= 0)
    df, y = df.loc[valid].copy(), y.loc[valid].astype(float)
    X = df.drop(columns=["SalePrice", "LogSalePrice", "Id"], errors="ignore")
    X = X.dropna(axis=1, how="all")
    if len(X) < 100:
        raise ValueError("Please provide at least 100 labeled training rows.")
    y_log = np.log1p(y.to_numpy())
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols))
    prep = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)
    weights = None
    weight_summary = None
    if model_kind == "Influence-weighted Gradient Boosting":
        weights, weight_summary = _training_reliability_weights(X, y_log, seed=42)
        estimator = GradientBoostingRegressor(
            loss="squared_error", learning_rate=0.05, n_estimators=200,
            max_depth=2, min_samples_leaf=3, random_state=42)
    else:
        estimator = HistGradientBoostingRegressor(
            loss="absolute_error", max_iter=300, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=15, random_state=42)
    model = Pipeline([("prep", prep), ("model", estimator)])
    if weights is None:
        model.fit(X, y_log)
    else:
        model.fit(X, y_log, model__sample_weight=weights)
    return model, list(X.columns), {"rows": int(len(X)), "features": int(X.shape[1]),
        "target_median": float(y.median()), "model_kind": model_kind, "weights": weight_summary,
        "neighborhoods": sorted(X["Neighborhood"].dropna().astype(str).unique().tolist()) if "Neighborhood" in X else []}


@st.cache_data(show_spinner=False)
def fetch_ames_csv_bytes():
    from sklearn.datasets import fetch_openml
    ames = fetch_openml(name="house_prices", as_frame=True, parser="auto")
    df = ames.data.copy()
    df["SalePrice"] = pd.to_numeric(ames.target, errors="coerce")
    return df.to_csv(index=False).encode("utf-8")


def show_table(filename: str, *, height: int = 360, use_container_width: bool = True):
    data = load_table(filename)
    if data.empty:
        st.info(f"Saved result table not found: {filename}")
        return data
    st.dataframe(data, height=height, use_container_width=use_container_width, hide_index=True)
    return data


def show_figure(filename: str, caption: str | None = None):
    path = FIG_DIR / filename
    if path.exists():
        st.image(str(path), caption=caption or filename, use_container_width=True)
    else:
        st.info(f"Saved figure not found: {filename}")


def nice_label(value: str) -> str:
    return str(value).replace("—", "—").replace("_", " ")


def money(value: float) -> str:
    return f"${value:,.0f}"


with st.sidebar:
    st.title("🏠 Outlier-Aware Regression")
    st.caption("Ames Housing · saved experimental results")
    page = st.radio(
        "Explore",
        ["Overview", "Price predictions", "Model comparison", "Contamination lab", "Influence & weights",
         "Ablations & error analysis", "Figures & tables", "Protocol & caveats"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("This app reads the notebook’s existing report assets. It does not rerun CV or modify the locked test results.")


if page == "Overview":
    st.title("Outlier-Aware Regression for House Prices")
    st.markdown("""
    **Research question:** Can modelling the reliability or influence of each training observation
    improve the accuracy and robustness of tabular regression?

    This project compares standard regression, removal and winsorization, robust losses,
    sample-weighted models, and iterative reweighting on Ames Housing. The target is
    `log1p(SalePrice)`. Use the pages in the sidebar to explore reported results and diagnostics.
    """)

    cv = load_table("table_07_01_paired_cv_ranked_comparison.csv")
    test = load_table("table_07_05_one_time_holdout_results.csv")
    if not cv.empty and not test.empty:
        best_cv = cv.sort_values("RMSE_log").iloc[0]
        weighted = test.loc[test["strategy"] == "Weighted — Gradient Boosting"]
        baseline = test.loc[test["strategy"] == "No handling — HistGBDT"]
        best_test = test.sort_values("test_RMSE_log").iloc[0]
        cols = st.columns(4)
        cols[0].metric("Development CV leader", str(best_cv["strategy"]), f"RMSE {best_cv['RMSE_log']:.4f}")
        if not weighted.empty:
            cols[1].metric("Preselected weighted method", "Weighted Gradient Boosting", f"Test RMSE {weighted.iloc[0]['test_RMSE_log']:.4f}")
        if not baseline.empty and not weighted.empty:
            delta_pct = 100 * (baseline.iloc[0]["test_RMSE_log"] - weighted.iloc[0]["test_RMSE_log"]) / baseline.iloc[0]["test_RMSE_log"]
            cols[2].metric("Weighted vs tree baseline", f"{delta_pct:.2f}% lower test RMSE", "point estimate; CIs overlap")
        cols[3].metric("Lowest test point estimate", str(best_test["strategy"]), f"RMSE {best_test['test_RMSE_log']:.4f}")

    st.info("The notebook saved no deployable fitted model or raw development data. This dashboard presents the recorded experiments and an interactive weight-mapping demonstration; it does not generate property price predictions.")
    st.subheader("Project conclusion")
    st.write("Influence-aware modelling showed conditional promise, not a uniform advantage. The chosen weighting rule, corruption type, base learner, price segment, and metric all affected the result. No paired CV comparison survived the reported Holm correction.")
    st.subheader("Key findings")
    findings = [
        ("CV ranking", "Hard-removal HistGBDT had the lowest mean development-CV RMSE (0.13948 ± 0.01501), narrowly ahead of no-handling HistGBDT (0.14009 ± 0.01455).", "table_07_01_paired_cv_ranked_comparison.csv"),
        ("Hold-out result", "The preselected weighted Gradient Boosting strategy had test RMSE 0.12376 versus 0.13043 for no-handling HistGBDT; the bootstrap intervals overlapped.", "table_07_05_one_time_holdout_results.csv"),
        ("Weighting ablation", "Linear weighting improved average Ridge RMSE/MAE/P90 in Stage 9, but its fold RMSE SD was higher than the unweighted baseline.", "table_09_02_signal_ablation_summary.csv"),
        ("Synthetic detection", "Detection was strongest for leverage and mixed injected corruption and weaker for target-only noise.", "table_08_04_injected_row_detection_summary.csv"),
    ]
    for heading, detail, ref in findings:
        st.markdown(f"**{heading}.** {detail}  \n*Source: `{ref}`*")

    st.subheader("Selected report figure")
    show_figure("fig_07_01_ranked_cv_rmse.png", "Paired development-CV strategy comparison")

elif page == "Price predictions":
    st.title("House price predictions")
    st.markdown("Train an estimator from an Ames-style labeled CSV, then estimate a single house or upload a batch. The saved experiment tables are not used as training data, and the notebook’s locked test set is not used here.")
    source = st.radio("Training data source", ["Upload a labeled CSV", "Load Ames Housing from OpenML"], horizontal=True)
    training_bytes = None
    if source == "Upload a labeled CSV":
        train_file = st.file_uploader("Upload training data (CSV with SalePrice)", type=["csv"], key="prediction_train_csv")
        if train_file is not None:
            training_bytes = train_file.getvalue()
    else:
        st.caption("This option retrieves the same house_prices dataset used by the notebook. If OpenML is unavailable, use the upload option with a local Ames training CSV.")
        if st.button("Load Ames from OpenML", type="primary"):
            try:
                training_bytes = fetch_ames_csv_bytes()
                st.session_state["ames_training_bytes"] = training_bytes
            except Exception as exc:
                st.error(f"Could not retrieve Ames Housing from OpenML: {exc}")
        training_bytes = training_bytes or st.session_state.get("ames_training_bytes")

    model_kind = st.selectbox("Prediction model", [
        "Influence-weighted Gradient Boosting", "Standard HistGradientBoosting"],
        help="The weighted model uses training-only influence signals and the notebook's exponential reliability mapping (k=3, w_min=0.05).")
    if training_bytes is not None and st.button("Train / load model", type="primary"):
        try:
            fitted_model, feature_names, model_info = train_house_model(training_bytes, model_kind)
            st.session_state["house_model"] = fitted_model
            st.session_state["house_feature_names"] = feature_names
            st.session_state["house_model_info"] = model_info
            st.success("Model is ready for predictions.")
        except Exception as exc:
            st.error(f"Model training failed: {exc}")

    if "house_model" not in st.session_state:
        st.info("Choose a training source, provide labeled data, and train a model to enable predictions. Kaggle-style Ames train.csv files with a SalePrice column are supported.")
    else:
        fitted_model = st.session_state["house_model"]
        feature_names = st.session_state["house_feature_names"]
        info = st.session_state["house_model_info"]
        if info["model_kind"] != model_kind:
            st.warning("The selected model differs from the model currently in memory. Train again to switch models.")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Training rows", f"{info['rows']:,}")
        metric_cols[1].metric("Input features", str(info["features"]))
        metric_cols[2].metric("Training target median", money(info["target_median"]))
        if info["weights"]:
            metric_cols[3].metric("Mean reliability weight", f"{info['weights']['mean_weight']:.3f}")
            st.caption(f"Training-only scorer: 7 ranked influence signals; mean score {info['weights']['mean_score']:.3f}; 5% floor reached by {info['weights']['floor_fraction']:.1%} of training rows.")

        mode = st.radio("Prediction input", ["Single-house form", "Batch CSV"], horizontal=True)
        if mode == "Single-house form":
            st.warning("The short form supplies common predictors and leaves other training features missing for pipeline imputation. For a more complete estimate, use the batch CSV option with the full property features.")
            with st.form("single_house_form"):
                form_cols = st.columns(4)
                values = {}
                numeric_defaults = {
                    "OverallQual": (6, 1, 10, 1), "GrLivArea": (1500, 300, 7000, 25),
                    "GarageCars": (2, 0, 5, 1), "TotalBsmtSF": (900, 0, 5000, 25),
                    "YearBuilt": (2000, 1870, 2026, 1), "LotArea": (9000, 1000, 100000, 250),
                    "FullBath": (2, 0, 5, 1), "BedroomAbvGr": (3, 0, 10, 1),
                }
                available_numeric = [k for k in numeric_defaults if k in feature_names]
                for i, name in enumerate(available_numeric):
                    default, low, high, step = numeric_defaults[name]
                    values[name] = form_cols[i % 4].number_input(name, min_value=low, max_value=high,
                        value=min(max(default, low), high), step=step, key=f"single_{name}")
                if "Neighborhood" in feature_names:
                    choices = info.get("neighborhoods") or ["MISSING"]
                    values["Neighborhood"] = st.selectbox("Neighborhood", choices, index=0)
                submitted = st.form_submit_button("Estimate sale price", type="primary")
            if submitted:
                row_values = {name: values.get(name, np.nan) for name in feature_names}
                row = pd.DataFrame([row_values], columns=feature_names)
                log_prediction = float(fitted_model.predict(row)[0])
                price_prediction = max(0.0, float(np.expm1(log_prediction)))
                st.success(f"Estimated sale price: {money(price_prediction)}")
                st.caption(f"Model output on log1p scale: {log_prediction:.4f}. No prediction interval is available. This form estimate uses {len(values)} entered fields; other features were imputed.")
        else:
            batch_file = st.file_uploader("Upload property rows as CSV", type=["csv"], key="prediction_batch_csv")
            if batch_file is not None:
                try:
                    batch_bytes = batch_file.getvalue()
                    batch = pd.read_csv(BytesIO(batch_bytes), low_memory=False)
                    source_columns = pd.read_csv(BytesIO(batch_bytes), nrows=0).columns
                    source_ids = pd.read_csv(BytesIO(batch_bytes), usecols=["Id"])["Id"] if "Id" in source_columns else None
                    batch = batch.drop(columns=["SalePrice", "LogSalePrice", "Id"], errors="ignore")
                    missing = [name for name in feature_names if name not in batch.columns]
                    batch = batch.reindex(columns=feature_names)
                    predictions = np.maximum(0, np.expm1(fitted_model.predict(batch)))
                    output = pd.DataFrame({"predicted_SalePrice": predictions})
                    if source_ids is not None:
                        output.insert(0, "Id", source_ids.to_numpy())
                    st.dataframe(output, use_container_width=True, hide_index=True)
                    st.download_button("Download predictions CSV", output.to_csv(index=False).encode("utf-8"),
                        file_name="ames_price_predictions.csv", mime="text/csv")
                    if missing:
                        st.caption(f"Missing {len(missing)} model features; those inputs were imputed. Provide a complete feature file for best results.")
                except Exception as exc:
                    st.error(f"Could not score this CSV: {exc}")

elif page == "Model comparison":
    st.title("Model comparison")
    st.caption("Development results come from repeated 5×3 CV. Hold-out metrics are from one final evaluation per frozen strategy and are descriptive, not a new selection step.")
    cv = load_table("table_07_01_paired_cv_ranked_comparison.csv")
    if cv.empty:
        st.warning("Stage 7 CV summary is not available.")
    else:
        metric_label = st.selectbox("Development-CV metric", ["RMSE_log", "MAE_log", "MedAE_log", "P90_log", "R2"])
        spread_col = f"{metric_label}_std"
        plot = cv.sort_values(metric_label, ascending=(metric_label != "R2"))
        fig = px.bar(plot, x=metric_label, y="strategy", orientation="h", color="strategy",
                     error_x=spread_col if spread_col in plot.columns else None,
                     hover_data=[c for c in ["n_folds", "wilcoxon_p_holm", "mean_RMSE_difference"] if c in plot.columns],
                     title=f"Repeated-CV strategy comparison: {metric_label}",
                     labels={metric_label: "Mean outer-fold score", "strategy": "Strategy"})
        fig.update_layout(showlegend=False, height=max(480, 31 * len(plot)), margin=dict(l=20, r=25, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Ranked CV results and paired tests")
        show_table("table_07_01_paired_cv_ranked_comparison.csv", height=440)

    st.subheader("Frozen hold-out results")
    test = load_table("table_07_05_one_time_holdout_results.csv")
    if not test.empty:
        metric = st.selectbox("Hold-out metric", ["test_RMSE_log", "test_MAE_log", "test_MedAE_log", "test_P90_log"], key="test_metric")
        fig = px.bar(test.sort_values(metric), x=metric, y="strategy", orientation="h", color="strategy",
                     hover_data=[c for c in [f"{metric}_CI_low", f"{metric}_CI_high", "cv_RMSE_log", "large_CV_test_gap"] if c in test.columns],
                     title=f"One-time hold-out comparison: {metric}", labels={metric: "Test metric", "strategy": "Frozen strategy"})
        fig.update_layout(showlegend=False, height=max(500, 29 * len(test)), margin=dict(l=20, r=20, t=60, b=20))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Full hold-out table with bootstrap intervals"):
            st.dataframe(test, use_container_width=True, hide_index=True)
    st.warning("HuberRegressor had a reported test RMSE of 1.97891 versus CV RMSE 0.15528. It is retained in the table as an observed generalization failure.")

elif page == "Contamination lab":
    st.title("Synthetic contamination lab")
    st.caption("Corruption was applied to outer training folds only; evaluation folds remained clean. Two contamination repeats were used, so repeat-SD bars are coarse.")
    perf = load_table("table_08_02_contamination_performance_summary.csv")
    detection = load_table("table_08_04_injected_row_detection_summary.csv")
    threshold = load_table("table_08_05_sharp_degradation_thresholds.csv")
    if perf.empty:
        st.warning("Stage 8 results are not available.")
    else:
        types = sorted(perf["contamination_type"].dropna().unique().tolist())
        chosen_types = st.multiselect("Corruption type", types, default=types)
        methods = sorted(perf["strategy"].dropna().unique().tolist())
        chosen_methods = st.multiselect("Strategies", methods, default=methods)
        metric = st.selectbox("Performance metric", ["RMSE_log", "P90_log", "MAE_log", "MedAE_log"])
        filtered = perf[perf["contamination_type"].isin(chosen_types) & perf["strategy"].isin(chosen_methods)].copy()
        err_col = f"{metric}_repeat_sd"
        if not filtered.empty:
            fig = px.line(filtered, x="contamination_pct", y=f"{metric}_mean", color="strategy",
                          facet_col="contamination_type", facet_col_wrap=1,
                          markers=True, error_y=err_col if err_col in filtered.columns else None,
                          title=f"{metric} under training contamination",
                          labels={"contamination_pct": "Contaminated training rows (%)", f"{metric}_mean": metric})
            fig.update_layout(height=max(450, 260 * max(1, len(chosen_types))), legend_title="Strategy")
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Performance summary table"):
            st.dataframe(filtered, use_container_width=True, hide_index=True)
    st.subheader("Injected-row detection")
    if not detection.empty:
        dtypes = sorted(detection["contamination_type"].dropna().unique())
        dtype = st.selectbox("Detection scenario", dtypes)
        view = detection[(detection["contamination_type"] == dtype) & (detection["contamination_pct"] > 0)]
        long = view.melt(id_vars="contamination_pct", value_vars=["AUROC_mean", "precision_at_k_mean"],
                         var_name="metric", value_name="score")
        fig = px.line(long, x="contamination_pct", y="score", color="metric", markers=True,
                      title=f"Detection scores: {dtype}", labels={"contamination_pct": "Injected rows (%)", "score": "Score"})
        fig.add_hline(y=.5, line_dash="dot", line_color="gray", annotation_text="AUROC chance")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(view, use_container_width=True, hide_index=True)
    st.subheader("First 10% RMSE-degradation threshold")
    if not threshold.empty:
        st.dataframe(threshold, use_container_width=True, hide_index=True)

elif page == "Influence & weights":
    st.title("Influence scoring and reliability weights")
    st.markdown("Explore how a normalized anomaly score is translated into a sample reliability weight. Higher scores indicate more anomalous/influential observations; the floor prevents zero-weighting.")
    left, right = st.columns([1, 2])
    with left:
        mapping = st.selectbox("Weight mapping", ["Linear", "Exponential", "Sigmoid"])
        score = st.slider("Example anomaly score", 0.0, 1.0, 0.75, 0.01)
        floor = st.slider("Minimum weight (w_min)", 0.01, 0.50, 0.05, 0.01)
        decay = st.slider("Exponential decay (k)", 0.0, 8.0, 3.0, 0.1)
        slope = st.slider("Sigmoid slope", 1.0, 20.0, 10.0, 0.5)

    def weight_map(x, kind):
        if kind == "Linear":
            return 1 - x * (1 - floor)
        if kind == "Exponential":
            return max(floor, math.exp(-decay * x))
        z = max(-60, min(60, slope * (x - .5)))
        return floor + (1 - floor) / (1 + math.exp(z))

    mapped = weight_map(score, mapping)
    with right:
        st.metric("Reliability weight", f"{mapped:.3f}", f"Influence score {score:.2f}")
        xs = [i / 200 for i in range(201)]
        ys = [weight_map(x, mapping) for x in xs]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=mapping, line=dict(width=3)))
        fig.add_trace(go.Scatter(x=[score], y=[mapped], mode="markers", name="Selected score", marker=dict(size=11)))
        fig.update_layout(title=f"{mapping} score-to-weight mapping", xaxis_title="Normalized anomaly score", yaxis_title="Reliability weight", yaxis=dict(range=[0, 1.03]), height=390)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Observed Stage 4/5 diagnostics")
    diagnostic_cols = st.columns(2)
    with diagnostic_cols[0]: show_figure("fig_04_02_signal_correlation.png", "Spearman correlation between influence signals")
    with diagnostic_cols[1]: show_figure("fig_05_02_weight_distributions.png", "Training-fold reliability-weight distributions")
    st.subheader("Signal and IRLS tables")
    t1, t2 = st.tabs(["Signal summary", "IRLS convergence"])
    with t1: show_table("table_04_01_score_summary.csv")
    with t2: show_table("table_05_02_irls_convergence.csv")

elif page == "Ablations & error analysis":
    st.title("Ablations and error analysis")
    st.caption("All score ablations are summarized across repeated development CV. The price-segment cuts are fold-specific and include out-of-range tail houses.")
    ablation = load_table("table_09_02_signal_ablation_summary.csv")
    if not ablation.empty:
        metric = st.selectbox("Ablation metric", ["RMSE_log_mean", "MAE_log_mean", "P90_log_mean"])
        error = metric.replace("_mean", "_sd")
        sort = ablation.sort_values(metric)
        fig = px.bar(sort, x=metric, y="method", orientation="h", color="method",
                     error_x=error if error in sort.columns else None,
                     title=f"Signal ablation: {metric.replace('_mean','')}", labels={metric: "Repeated-CV mean"})
        fig.update_layout(showlegend=False, height=570)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Ablation results"): st.dataframe(ablation, use_container_width=True, hide_index=True)
    sens = load_table("table_09_05_k_wmin_sensitivity.csv")
    if not sens.empty:
        st.subheader("Weight-parameter sensitivity")
        pivot = sens.pivot(index="w_min", columns="k", values="RMSE_log_mean")
        fig = px.imshow(pivot, text_auto=".4f", aspect="auto", color_continuous_scale="YlGnBu_r",
                        labels={"x": "Decay k", "y": "Minimum weight", "color": "CV RMSE"},
                        title="Mean repeated-CV RMSE across (k, w_min)")
        st.plotly_chart(fig, use_container_width=True)
    seg = load_table("table_09_06_error_by_price_segment.csv")
    if not seg.empty:
        st.subheader("Error by sale-price segment")
        metric = st.selectbox("Segment error metric", ["MAE_log", "RMSE_log", "P90_log"], key="segmetric")
        fig = px.bar(seg, x="price_segment", y=metric, color="model", barmode="group",
                     title=f"{metric} by price quartile", labels={"price_segment": "Fold-specific price band"})
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(seg, use_container_width=True, hide_index=True)
    st.subheader("Feature-importance changes")
    imp = load_table("table_09_09_permutation_importance_comparison.csv")
    if not imp.empty:
        topn = st.slider("Features to display", 5, min(20, len(imp)), min(12, len(imp)))
        show_imp = imp.head(topn).melt(id_vars="feature", value_vars=["Baseline Ridge", "Weighted Ridge"], var_name="model", value_name="importance")
        fig = px.bar(show_imp, x="importance", y="feature", color="model", barmode="group", orientation="h",
                     title="Permutation importance: baseline vs weighted Ridge")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Examples with low reliability")
    show_table("table_09_10_low_reliability_house_examples.csv", height=420)
    st.caption("Low reliability identifies unusual influence patterns; it does not establish that a house record is invalid.")

elif page == "Figures & tables":
    st.title("Report figures and tables")
    fig_files = sorted(p.name for p in FIG_DIR.glob("*.png")) if FIG_DIR.exists() else []
    table_files = sorted(p.name for p in TABLE_DIR.glob("*.csv")) if TABLE_DIR.exists() else []
    tab_fig, tab_table = st.tabs([f"Figures ({len(fig_files)})", f"Tables ({len(table_files)})"])
    with tab_fig:
        if fig_files:
            selected = st.selectbox("Choose a saved figure", fig_files)
            show_figure(selected, selected.replace("_", " ").removesuffix(".png"))
        else: st.info("No figure assets found.")
    with tab_table:
        if table_files:
            selected_table = st.selectbox("Choose a saved result table", table_files)
            data = load_table(selected_table)
            st.dataframe(data, use_container_width=True, height=500, hide_index=True)
            md_path = TABLE_DIR / selected_table.replace(".csv", ".md")
            if md_path.exists():
                with st.expander("Markdown companion"):
                    st.code(md_path.read_text(encoding="utf-8"), language="markdown")
        else: st.info("No table assets found.")
    index_csv = TABLE_DIR / "table_10_01_best_method_summary.csv"
    if index_csv.exists():
        with st.expander("Report asset index count"):
            st.write(f"{len(fig_files)} figures and {len(table_files)} CSV result tables are available.")

elif page == "Protocol & caveats":
    st.title("Validation protocol and threats to validity")
    st.markdown("""
    - **Split:** One 80/20 split stratified by quantile bins of `log1p(SalePrice)`, seed 42. Development set: 1,168 rows; locked hold-out test: 292 rows.
    - **Model comparison:** Repeated 5-fold CV with three repeats (15 outer folds) on development data.
    - **Inner selection:** Hyperparameter selection and boosting early stopping use a 15% validation split drawn only from each outer training fold.
    - **Leakage controls:** Preprocessing, outlier scores, filtering, clipping thresholds, and weights are learned within training data. CV validation remains clean.
    - **Test handling:** The test set is not used for EDA, score calculation, reweighting, or tuning. Each frozen strategy is evaluated once; 95% bootstrap intervals use 1,000 resamples.
    - **Contamination experiment:** Synthetic corruption is applied only to the outer training rows. Outer validation and test remain clean; the inner early-stopping subset may be contaminated because it is drawn after injection.

    **Threats to validity.** The test set is modest (~290 rows), limiting precision for rare and expensive homes. Bootstrap intervals reflect resampling within a single split, not uncertainty across new splits, populations, or time periods. Repeated-CV folds reuse observations and are not fully independent. Stage 8 used two contamination repeats and a limited collection of synthetic corruption rules. High influence is not equivalent to a data error, and some legitimate rare homes may be down-weighted. All results come from one dataset.
    """)
    status = load_table("table_10_02_final_hypothesis_status.csv")
    if status.empty: status = load_table("table_07_06_hypothesis_status.csv")
    if not status.empty:
        st.subheader("Hypothesis summary")
        st.dataframe(status, use_container_width=True, hide_index=True)

st.divider()
st.caption("Outlier-Aware Regression Framework · Ames Housing · seed 42")
