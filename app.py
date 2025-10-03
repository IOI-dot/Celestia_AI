# Celestia AI — Exoplanet Discovery Suite
# Researcher-friendly Streamlit app with data ingestion, live training, hyperparameter tuning,
# evaluation dashboard, thresholding, and export. Uses 10 Kepler features.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import json
import time
import warnings
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_fscore_support,
    brier_score_loss,
    roc_curve
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Celestia AI — Exoplanet Discovery Suite",
    layout="wide",
    page_icon="✨",
    initial_sidebar_state="expanded",
)

# =========================
# Fixed CSS — dark canvas, WHITE text for titles + labels
# =========================
def inject_css():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&display=swap');
      :root {
        --celestia-white: #ffffff;
      }
      .stApp { background: #0c0f1a; }
      .main .block-container { padding-top: 1.0rem; }
      h1, h2, h3, h4, h5, h6 { color: var(--celestia-white) !important; font-family: "Space Grotesk", sans-serif; }
      label, .stMarkdown p, .stText, .stCaption, .stSelectbox, .stNumberInput, .stFileUploader, .stCheckbox, .stRadio, .stMetric-value, .stMetric-label {
        color: var(--celestia-white) !important;
        font-family: "Space Grotesk", sans-serif;
      }
      .stAlert { background: rgba(255,255,255,0.06) !important; border: 1px solid rgba(255,255,255,0.15); }
      .stButton > button {
          background: linear-gradient(90deg, #6b7bff 0%, #7b5fe0 100%);
          border: none; color: #fff; padding: 10px 22px; border-radius: 12px; font-weight: 700;
          box-shadow: 0 4px 16px rgba(107,123,255,.25);
      }
      .stDownloadButton > button {
          background: rgba(255,255,255,0.06); color:#fff; border:1px solid rgba(255,255,255,0.2);
          border-radius: 12px; padding: 8px 16px;
      }
      [data-testid="stFileUploadDropzone"] {
          background: rgba(255,255,255,0.05);
          border: 2px dashed rgba(255,255,255,0.18);
          border-radius: 12px;
      }
      [data-testid="stMetric"] { background: rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.1); border-radius:12px; padding: 8px; }
      .e1f1d6gn3 a, .stMarkdown a { color: #8cc7ff; text-decoration: none; }
      .stTabs [data-baseweb="tab-list"] { gap:8px; }
      .stTabs [data-baseweb="tab"] {
          background: rgba(255,255,255,0.05); color:#fff; border-radius: 12px; padding: 6px 10px; border:1px solid rgba(255,255,255,.08);
      }
      .stTabs [aria-selected="true"] { background: rgba(255,255,255,.12); }
      .stDataFrame, .stTable { background: rgba(255,255,255,0.03); }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# =========================
# Constants
# =========================
APP_TITLE = "Celestia AI — Exoplanet Discovery Suite"
SELECTED_FEATURES = [
    "koi_score",
    "koi_fpflag_nt",
    "koi_model_snr",
    "koi_fpflag_co",
    "koi_fpflag_ss",
    "koi_fpflag_ec",
    "koi_impact",
    "koi_duration",
    "koi_prad",
    "koi_period",
]
TARGET_COLUMNS_CANDIDATES = ["koi_disposition", "disposition", "label"]

# =========================
# Session State
# =========================
if "data_store" not in st.session_state:
    st.session_state["data_store"] = pd.DataFrame(columns=SELECTED_FEATURES + ["koi_disposition"])
if "pipeline" not in st.session_state:
    st.session_state["pipeline"] = None
if "label_encoder" not in st.session_state:
    st.session_state["label_encoder"] = None
if "metrics" not in st.session_state:
    st.session_state["metrics"] = {}
if "last_train_time" not in st.session_state:
    st.session_state["last_train_time"] = None

# =========================
# Header
# =========================
st.markdown(f"# {APP_TITLE}")
st.caption("A unified interface for scientists and learners to ingest exoplanet data, train robust models, tune hyperparameters, and classify candidates with confidence.")

# =========================
# Utilities
# =========================
def coerce_and_align(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with exactly our 10 features (created if missing, numeric coerced)."""
    X = df.copy()
    for col in SELECTED_FEATURES:
        if col not in X.columns:
            X[col] = 0
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    return X[SELECTED_FEATURES]

def find_target_column(df: pd.DataFrame) -> str | None:
    for c in TARGET_COLUMNS_CANDIDATES:
        if c in df.columns:
            return c
    return None

def build_pipeline(params: dict) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, SELECTED_FEATURES)])
    # Handle class_weight support gracefully
    hgb_kwargs = dict(
        learning_rate=params["learning_rate"],
        max_iter=params["max_iter"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"],
        early_stopping=params["early_stopping"],
        validation_fraction=params["validation_fraction"],
        n_iter_no_change=params["n_iter_no_change"],
        random_state=42,
    )
    if params.get("class_weight"):
        # sklearn >=1.4 supports class_weight in HGBClassifier; ignore if not available
        try:
            clf = HistGradientBoostingClassifier(**hgb_kwargs, class_weight=params["class_weight"])
        except TypeError:
            clf = HistGradientBoostingClassifier(**hgb_kwargs)  # fallback
    else:
        clf = HistGradientBoostingClassifier(**hgb_kwargs)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

def ensure_label_encoder(le: LabelEncoder | None, y_values: pd.Series | list) -> LabelEncoder:
    if le is None:
        le = LabelEncoder()
        le.fit(sorted(pd.Series(y_values).astype(str).unique()))
    return le

def compute_metrics(y_true, y_pred, proba, classes_):
    report = classification_report(y_true, y_pred, target_names=classes_, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes_)))
    # ROC-AUC (OvR)
    y_true_bin = label_binarize(y_true, classes=range(len(classes_)))
    try:
        auc_ovr = roc_auc_score(y_true_bin, proba, average="macro", multi_class="ovr")
    except Exception:
        auc_ovr = np.nan
    # Calibration (Brier) – macro over classes by averaging per-class Brier
    briers = []
    for i in range(len(classes_)):
        briers.append(brier_score_loss(y_true_bin[:, i], proba[:, i]))
    brier_macro = float(np.mean(briers))
    return report, cm, auc_ovr, brier_macro

def plot_confusion_matrix(cm, classes_):
    df_cm = pd.DataFrame(cm, index=classes_, columns=classes_)
    fig = px.imshow(df_cm, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                    title="Confusion Matrix")
    fig.update_layout(template="plotly_dark", height=420)
    return fig

def plot_calibration(proba, y_true, classes_):
    # show for each class a reliability curve (OvR)
    fig = go.Figure()
    y_true_bin = label_binarize(y_true, classes=range(len(classes_)))
    for i, name in enumerate(classes_):
        prob_pos = proba[:, i]
        frac_pos, mean_pred = calibration_curve(y_true_bin[:, i], prob_pos, n_bins=10, strategy="quantile")
        fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name=f"{name}"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect", line=dict(dash="dash")))
    fig.update_layout(title="Calibration (Reliability) Curves", xaxis_title="Mean Predicted Probability", yaxis_title="Fraction of Positives",
                      template="plotly_dark", height=420)
    return fig

def plot_feature_importance(pipeline: Pipeline):
    try:
        clf = pipeline.named_steps["classifier"]
        # HistGradientBoosting has .feature_importances_ in newer versions
        importances = getattr(clf, "feature_importances_", None)
        if importances is None:
            return None
        s = pd.Series(importances, index=SELECTED_FEATURES).sort_values(ascending=True)
        fig = px.bar(x=s.values, y=s.index, orientation="h", title="Feature Importances")
        fig.update_layout(template="plotly_dark", height=420, xaxis_title="Importance", yaxis_title="")
        return fig
    except Exception:
        return None

def save_artifacts(pipeline, label_encoder):
    joblib.dump(pipeline, "celestia_pipeline.joblib")
    joblib.dump(label_encoder, "celestia_label_encoder.joblib")

def load_artifacts():
    try:
        pipe = joblib.load("kepler_gb_pipeline_weighted.joblib")
        le = joblib.load("kepler_label_encoder.joblib")
        return pipe, le
    except Exception:
        # Try Celestia filenames (if previously saved here)
        try:
            pipe = joblib.load("celestia_pipeline.joblib")
            le = joblib.load("celestia_label_encoder.joblib")
            return pipe, le
        except Exception:
            return None, None

# =========================
# Sidebar — Global Controls
# =========================
with st.sidebar:
    st.subheader("⚙️ Global Controls")
    auto_train = st.checkbox("Auto-train on ingest", value=True, help="Retrain model whenever new labeled data is ingested.")
    show_guidance = st.checkbox("Guided Mode (tooltips)", value=True)
    st.markdown("---")
    st.caption("Artifacts")
    if st.button("🔄 Load Existing Artifacts (joblib)"):
        pipe, le = load_artifacts()
        if pipe is not None and le is not None:
            st.session_state["pipeline"] = pipe
            st.session_state["label_encoder"] = le
            st.success("Loaded existing artifacts.")
        else:
            st.warning("No compatible artifacts found in working directory.")

# =========================
# Tabs
# =========================
tab_ingest, tab_explore, tab_train, tab_classify, tab_metrics, tab_export, tab_about = st.tabs([
    "📥 Ingest Data",
    "🛰️ Explore",
    "🧬 Train & Tune",
    "🧪 Classify",
    "📊 Metrics & Explain",
    "📦 Export",
    "ℹ️ About"
])

# -------------------------
# Ingest Data
# -------------------------
with tab_ingest:
    st.subheader("📥 Ingest Kepler/TESS-like CSV")
    st.caption("Upload one or more CSV files that include the 10 features. If a label column is present (e.g., `koi_disposition`), it will be used for supervised training.")
    up = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    col_t = st.columns(3)
    with col_t[0]:
        if st.button("🧾 Download CSV Schema Template"):
            # Create an in-memory CSV template
            template = pd.DataFrame({c: [] for c in SELECTED_FEATURES + ["koi_disposition"]})
            buf = io.StringIO()
            template.to_csv(buf, index=False)
            st.download_button("📥 Get Template", data=buf.getvalue(), file_name="celestia_template.csv", mime="text/csv", key="dl_temp")

    if up:
        combined_added = 0
        for f in up:
            try:
                df = pd.read_csv(f, comment="#")
                df.columns = df.columns.str.strip()
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}")
                continue

            # Identify target if exists
            target_col = find_target_column(df)
            if target_col is None:
                df["koi_disposition"] = np.nan  # unlabeled; still ingest for later classification

            # keep only necessary columns if present
            keep_cols = [c for c in SELECTED_FEATURES if c in df.columns]
            missing = [c for c in SELECTED_FEATURES if c not in df.columns]
            if missing:
                st.warning(f"{f.name}: Missing features {missing} — they will be created as 0 during preprocessing.")
            # Align features and maintain (possible) label
            X = coerce_and_align(df)
            y = df["koi_disposition"] if "koi_disposition" in df.columns else np.nan
            block = X.copy()
            block["koi_disposition"] = y
            st.session_state["data_store"] = pd.concat([st.session_state["data_store"], block], axis=0, ignore_index=True)
            combined_added += len(block)

        st.success(f"Ingested {combined_added} rows.")
        if auto_train and st.session_state["data_store"]["koi_disposition"].notna().any():
            st.info("Auto-training model on labeled rows…")
            st.experimental_rerun()

# -------------------------
# Explore
# -------------------------
with tab_explore:
    st.subheader("🛰️ Explore Dataset")
    data = st.session_state["data_store"]
    if data.empty:
        st.info("No data ingested yet.")
    else:
        st.write(f"Rows: **{len(data)}** | Features: **{len(SELECTED_FEATURES)}**")
        with st.expander("Preview (first 200 rows)", expanded=False):
            st.dataframe(data.head(200), use_container_width=True)

        # Simple summaries
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Labeled", int(data["koi_disposition"].notna().sum()))
        with c2: st.metric("Unlabeled", int(data["koi_disposition"].isna().sum()))
        with c3: st.metric("Features", len(SELECTED_FEATURES))
        with c4: st.metric("Artifacts Loaded", "Yes" if st.session_state["pipeline"] is not None else "No")

        # Disposition distribution (if labeled)
        if data["koi_disposition"].notna().any():
            vc = data["koi_disposition"].dropna().value_counts()
            fig = px.bar(vc, title="Label Distribution (Ingested)", labels={"value":"Count","index":"Label"})
            fig.update_layout(template="plotly_dark", height=420)
            st.plotly_chart(fig, use_container_width=True)

        # Quick univariate explorer
        feat = st.selectbox("Feature to inspect", SELECTED_FEATURES)
        fig = px.histogram(data, x=feat, nbins=40, title=f"Distribution: {feat}")
        fig.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Train & Tune
# -------------------------
with tab_train:
    st.subheader("🧬 Train & Hyperparameter Tuning")

    data = st.session_state["data_store"]
    labeled = data.dropna(subset=["koi_disposition"])
    if labeled.empty:
        st.info("No labeled data available. Ingest a CSV with a label column (e.g., koi_disposition).")
    else:
        # Target encoding
        le = ensure_label_encoder(st.session_state["label_encoder"], labeled["koi_disposition"])
        y = le.transform(labeled["koi_disposition"].astype(str))
        X = coerce_and_align(labeled)

        # Tuning controls
        st.markdown("#### Training Hyperparameters")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            learning_rate = st.slider("learning_rate", 0.01, 0.5, 0.05, 0.01)
            max_iter = st.slider("max_iter", 50, 1000, 500, 50)
            max_depth = st.slider("max_depth", 2, 16, 6, 1)
        with cc2:
            min_samples_leaf = st.slider("min_samples_leaf", 5, 200, 20, 5)
            l2_regularization = st.slider("l2_regularization", 0.0, 5.0, 1.0, 0.1)
            early_stopping = st.checkbox("early_stopping", value=True)
        with cc3:
            validation_fraction = st.slider("validation_fraction", 0.05, 0.3, 0.1, 0.01)
            n_iter_no_change = st.slider("n_iter_no_change", 5, 50, 20, 1)
            use_class_weight = st.checkbox("Use class_weight (weight CONFIRMED & CANDIDATE higher)", value=True)

        class_weight = None
        if use_class_weight:
            # Map label order to weights; assume classes_ sorted the same way as le.classes_
            # Default: weight FP=1.0; others=5.0
            classes_list = list(le.classes_)
            w_conf = st.number_input("Weight: CONFIRMED", min_value=1.0, value=5.0, step=0.5)
            w_cand = st.number_input("Weight: CANDIDATE", min_value=1.0, value=5.0, step=0.5)
            w_fp   = st.number_input("Weight: FALSE POSITIVE", min_value=0.5, value=1.0, step=0.5)
            cw_map = {
                int(np.where(classes_list=="CONFIRMED")[0][0]) if "CONFIRMED" in classes_list else 1: w_conf,
                int(np.where(classes_list=="CANDIDATE")[0][0]) if "CANDIDATE" in classes_list else 0: w_cand,
                int(np.where(classes_list=="FALSE POSITIVE")[0][0]) if "FALSE POSITIVE" in classes_list else 2: w_fp,
            }
            class_weight = cw_map

        params = dict(
            learning_rate=learning_rate,
            max_iter=int(max_iter),
            max_depth=int(max_depth),
            min_samples_leaf=int(min_samples_leaf),
            l2_regularization=float(l2_regularization),
            early_stopping=early_stopping,
            validation_fraction=float(validation_fraction),
            n_iter_no_change=int(n_iter_no_change),
            class_weight=class_weight
        )

        cta1, cta2, cta3 = st.columns([1,1,2])
        with cta1:
            train_btn = st.button("🚀 Train / Retrain")
        with cta2:
            cv_btn = st.button("🧪 5-Fold CV (macro F1)")

        if train_btn:
            with st.spinner("Training model…"):
                pipe = build_pipeline(params)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                                    random_state=42, stratify=y)
                pipe.fit(X_train, y_train)
                proba = pipe.predict_proba(X_test)
                y_pred = pipe.predict(X_test)
                report, cm, auc_ovr, brier_macro = compute_metrics(y_test, y_pred, proba, le.classes_)
                st.session_state["pipeline"] = pipe
                st.session_state["label_encoder"] = le
                st.session_state["metrics"] = dict(
                    report=report, cm=cm.tolist(), auc_ovr=auc_ovr, brier=brier_macro,
                    params=params, timestamp=datetime.now().isoformat()
                )
                st.session_state["last_train_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("Model trained and stored in session.")

        if cv_btn:
            with st.spinner("Running Stratified 5-Fold cross-validation…"):
                pipe = build_pipeline(params)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                # macro F1 via scoring
                try:
                    from sklearn.metrics import make_scorer, f1_score
                    scores = cross_val_score(pipe, X, y, scoring=make_scorer(f1_score, average="macro"), cv=skf, n_jobs=None)
                except Exception:
                    scores = cross_val_score(pipe, X, y, scoring="accuracy", cv=skf, n_jobs=None)
                st.info(f"CV macro F1 (mean ± std): **{np.mean(scores):.3f} ± {np.std(scores):.3f}**")

        if st.session_state["pipeline"] is not None and st.session_state["metrics"]:
            st.markdown("### Current Model Summary")
            m = st.session_state["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("ROC-AUC (OvR, macro)", f"{m['auc_ovr']:.3f}" if m["auc_ovr"]==m["auc_ovr"] else "n/a")
            with c2: st.metric("Brier (macro)", f"{m['brier']:.3f}")
            with c3: st.metric("Trained", st.session_state["last_train_time"] or "—")
            with c4: st.metric("Samples (labeled)", int(len(X)))

# -------------------------
# Classify
# -------------------------
with tab_classify:
    st.subheader("🧪 Classify Candidates")

    # Threshold tuning (per-class)
    st.markdown("#### Decision Thresholds")
    st.caption("Adjust probability thresholds that trigger each predicted label. The highest class whose probability exceeds its threshold is selected; otherwise the argmax is used.")
    t1, t2, t3 = st.columns(3)
    th_fp = t1.slider("Threshold — FALSE POSITIVE", 0.0, 0.99, 0.50, 0.01)
    th_cand = t2.slider("Threshold — CANDIDATE", 0.0, 0.99, 0.50, 0.01)
    th_conf = t3.slider("Threshold — CONFIRMED", 0.0, 0.99, 0.50, 0.01)
    thresholds = {"FALSE POSITIVE": th_fp, "CANDIDATE": th_cand, "CONFIRMED": th_conf}

    # Quick single row form
    st.markdown("#### Quick Form")
    q1, q2, q3 = st.columns(3)
    with q1:
        koi_score = st.number_input("koi_score", 0.0, 1.0, 0.5, 0.01)
        koi_model_snr = st.number_input("koi_model_snr", 0.0, 1e6, 20.0, 0.1)
        koi_impact = st.number_input("koi_impact", 0.0, 3.0, 0.5, 0.01)
    with q2:
        koi_duration = st.number_input("koi_duration (hours)", 0.0, 1e6, 10.0, 0.1)
        koi_prad = st.number_input("koi_prad (Earth radii)", 0.0, 1e6, 1.5, 0.1)
        koi_period = st.number_input("koi_period (days)", 0.0, 1e6, 30.0, 0.1)
    with q3:
        koi_fpflag_nt = st.selectbox("koi_fpflag_nt", [0,1], index=0)
        koi_fpflag_co = st.selectbox("koi_fpflag_co", [0,1], index=0)
        koi_fpflag_ss = st.selectbox("koi_fpflag_ss", [0,1], index=0)
        koi_fpflag_ec = st.selectbox("koi_fpflag_ec", [0,1], index=0)

    do_single = st.button("🔮 Classify Single")
    if do_single:
        df_row = pd.DataFrame([{
            "koi_score": koi_score,
            "koi_fpflag_nt": koi_fpflag_nt,
            "koi_model_snr": koi_model_snr,
            "koi_fpflag_co": koi_fpflag_co,
            "koi_fpflag_ss": koi_fpflag_ss,
            "koi_fpflag_ec": koi_fpflag_ec,
            "koi_impact": koi_impact,
            "koi_duration": koi_duration,
            "koi_prad": koi_prad,
            "koi_period": koi_period,
        }])
        X = coerce_and_align(df_row)
        pipe = st.session_state["pipeline"]
        le = st.session_state["label_encoder"]
        if pipe is None or le is None:
            st.error("No trained/loaded model. Train a model in **Train & Tune** first or load artifacts from the sidebar.")
        else:
            proba = pipe.predict_proba(X)[0]
            classes_ = list(le.classes_)
            # decision with thresholds
            class_idx = np.argmax(proba)
            pick = classes_[class_idx]
            for idx, name in enumerate(classes_):
                if proba[idx] >= thresholds.get(name, 0.5):
                    pick = name
            conf = float(np.max(proba) * 100)
            st.markdown(f"### Prediction: **{pick}**  •  confidence: **{conf:.1f}%**")
            fig = go.Figure(data=[go.Bar(x=classes_, y=(proba*100).round(2), text=[f"{p*100:.1f}%" for p in proba], textposition="auto")])
            fig.update_layout(template="plotly_dark", height=360, title="Probability Distribution", xaxis_title="Class", yaxis_title="Probability (%)")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Batch CSV Classification")
    up_batch = st.file_uploader("Upload CSV to classify (no label needed)", type=["csv"], key="cls_csv")
    if up_batch:
        df = pd.read_csv(up_batch, comment="#")
        X = coerce_and_align(df)
        pipe = st.session_state["pipeline"]
        le = st.session_state["label_encoder"]
        if pipe is None or le is None:
            st.error("No trained/loaded model.")
        else:
            P = pipe.predict_proba(X)
            classes_ = list(le.classes_)
            # apply thresholds
            argmax = np.argmax(P, axis=1)
            labels = [classes_[i] for i in argmax]
            # threshold override
            for i in range(len(X)):
                for j, name in enumerate(classes_):
                    if P[i, j] >= thresholds.get(name, 0.5):
                        labels[i] = name
            out = df.copy()
            out["prediction"] = labels
            for j, name in enumerate(classes_):
                out[f"proba_{name}"] = np.round(P[:, j], 6)
            st.success(f"Classified {len(out)} rows.")
            st.dataframe(out.head(200), use_container_width=True)
            csv_buf = io.StringIO()
            out.to_csv(csv_buf, index=False)
            st.download_button("📥 Download Predictions CSV", data=csv_buf.getvalue(),
                               file_name=f"celestia_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv")

# -------------------------
# Metrics & Explain
# -------------------------
with tab_metrics:
    st.subheader("📊 Metrics & Explainability")

    if st.session_state["pipeline"] is None or st.session_state["label_encoder"] is None or not st.session_state["metrics"]:
        st.info("Train a model in **Train & Tune** to populate this dashboard.")
    else:
        m = st.session_state["metrics"]
        le = st.session_state["label_encoder"]
        classes_ = list(le.classes_)

        # Summaries
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("ROC-AUC (OvR)", f"{m['auc_ovr']:.3f}" if m['auc_ovr']==m['auc_ovr'] else "n/a")
        with c2: st.metric("Brier", f"{m['brier']:.3f}")
        with c3: st.metric("Params", value=sum(1 for _ in m["params"].items()))
        with c4: st.metric("Last Trained", m.get("timestamp","—").split("T")[0])

        # Confusion Matrix
        fig_cm = plot_confusion_matrix(np.array(m["cm"]), classes_)
        st.plotly_chart(fig_cm, use_container_width=True)

        # Per-class table (precision/recall/f1/support)
        rep = pd.DataFrame(m["report"]).T
        rep = rep.loc[[c for c in classes_] + ["macro avg","weighted avg"], ["precision","recall","f1-score","support"]]
        st.markdown("#### Classification Report")
        st.dataframe(np.round(rep, 3), use_container_width=True)

        # Recreate holdout split to plot calibration and feature importance (safe approach: re-run quick split)
        data = st.session_state["data_store"].dropna(subset=["koi_disposition"])
        X = coerce_and_align(data)
        y = le.transform(data["koi_disposition"].astype(str))
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        pipe = st.session_state["pipeline"]
        proba = pipe.predict_proba(Xte)

        # Calibration
        st.markdown("#### Calibration")
        cal_fig = plot_calibration(proba, yte, classes_)
        st.plotly_chart(cal_fig, use_container_width=True)

        # Feature Importances
        st.markdown("#### Feature Importances")
        fi_fig = plot_feature_importance(pipe)
        if fi_fig is not None:
            st.plotly_chart(fi_fig, use_container_width=True)
        else:
            st.info("Estimator does not provide feature importances on this version.")

# -------------------------
# Export
# -------------------------
with tab_export:
    st.subheader("📦 Export Artifacts & Reports")
    pipe = st.session_state["pipeline"]
    le = st.session_state["label_encoder"]
    if pipe is None or le is None:
        st.info("Nothing to export yet — train or load a model first.")
    else:
        save_artifacts(pipe, le)
        st.success("Artifacts saved as `celestia_pipeline.joblib` and `celestia_label_encoder.joblib` in the working directory.")

        with open("celestia_pipeline.joblib", "rb") as f:
            st.download_button("📥 Download Pipeline", f.read(), "celestia_pipeline.joblib")
        with open("celestia_label_encoder.joblib", "rb") as f:
            st.download_button("📥 Download Label Encoder", f.read(), "celestia_label_encoder.joblib")

        # Simple model card JSON
        if st.session_state["metrics"]:
            m = st.session_state["metrics"].copy()
            m["classes_"] = list(le.classes_)
            m["feature_set"] = SELECTED_FEATURES
            card = json.dumps(m, indent=2)
            st.download_button("📄 Download Evaluation Report (JSON)", data=card, file_name="celestia_model_report.json", mime="application/json")

# -------------------------
# About
# -------------------------
with tab_about:
    st.subheader("ℹ️ About Celestia AI")
    st.markdown("""
**Celestia AI — Exoplanet Discovery Suite** focuses on two personas:

- **Researchers** who need reproducible training, explicit thresholds, metrics (ROC-AUC, calibration),
  exportable artifacts, and traceable hyperparameters.
- **Novices** who benefit from a guided, opinionated interface, a CSV schema template, and intuitive visual feedback.

**Innovations & UX extras**:
- Auto-train on ingest for rapid iteration with new labeled data  
- Per-class **decision thresholds** to operationalize candidate triage  
- **Calibration** visualization to detect over/under-confidence  
- Light **feature importance** to guide future curation  
- One-click **model card** (JSON) and artifact export  
""")

    with st.expander("Data Dictionary (10 features)", expanded=True):
        dd = pd.DataFrame({
            "feature": SELECTED_FEATURES,
            "description": [
                "Pipeline score of KOI (higher is better)",
                "Not transit-like flag (1=true flag)",
                "Model signal-to-noise ratio",
                "Centroid offset flag",
                "Significant secondary event flag",
                "Eclipsing binary flag",
                "Transit impact parameter (0-1 typical)",
                "Transit duration (hours)",
                "Planet radius (Earth radii)",
                "Orbital period (days)",
            ]
        })
        st.dataframe(dd, use_container_width=True)

# End
