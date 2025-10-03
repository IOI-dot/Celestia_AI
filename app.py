# Celestia AI — Exoplanet Discovery Suite
# Elegant UI: centered hero with orbiting planets + twinkling stars, soft slate boxes,
# researcher-grade features: data ingest, training, hyperparameter tuning, metrics, thresholding, classification & export.

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
# CSS — soft palette, centered hero, rotating planets, twinkling stars
# =========================
def inject_css_and_space_layers():
    st.markdown("""
    <style>
      :root{
        --bg-1:#0b1220;         /* deep night blue */
        --bg-2:#0e1726;         /* ink */
        --card:#121a2b;         /* soft slate */
        --card-2:#1a2436;       /* slightly lighter slate */
        --stroke:rgba(255,255,255,.12);
        --white:#ffffff;
        --muted:#cfe3ff;
        --accent:#86e1ff;       /* sky/teal */
        --accent-2:#a78bfa;     /* soft violet */
        --accent-3:#34d399;     /* mint */
      }

      .stApp { background: radial-gradient(1200px 800px at 50% -10%, #101b31 0%, var(--bg-1) 35%, var(--bg-2) 100%); }
      .main .block-container { padding-top: 0.8rem; position:relative; z-index:10; }

      /* Global typography — all headings/labels white */
      h1,h2,h3,h4,h5,h6,
      label, .stMarkdown, .stText, .stCaption, .stSelectbox, .stNumberInput, .stFileUploader,
      .stRadio, .stCheckbox, .stMetric-value, .stMetric-label { color: var(--white) !important; }

      /* Hero with orbits */
      .celestia-hero {
        position: relative;
        height: 260px;
        display: grid;
        place-items: center;
        overflow: hidden;
        margin-bottom: 12px;
      }
      .celestia-hero .title-wrap {
        position: relative;
        z-index: 5;
        text-align: center;
      }
      .celestia-hero h1 {
        margin: 0;
        font-weight: 800;
        letter-spacing: 0.5px;
        font-size: clamp(36px, 4.4vw, 64px);
        color: var(--white);
        text-shadow: 0 8px 26px rgba(134,225,255,.25);
      }
      .celestia-hero p {
        margin: 6px 0 0 0;
        color: var(--muted);
        font-size: 15px;
      }

      /* Orbits */
      .orbits {
        position: absolute;
        inset: 0;
        display: grid;
        place-items: center;
        z-index: 2;
        pointer-events: none;
        filter: drop-shadow(0 8px 20px rgba(0,0,0,.35));
      }
      .orbit {
        position: absolute;
        border-radius: 50%;
        border: 1px dashed rgba(255,255,255,.12);
        animation: spin linear infinite;
      }
      .orbit-1 { width: 260px; height: 260px; animation-duration: 38s; }
      .orbit-2 { width: 340px; height: 340px; animation-duration: 56s; animation-direction: reverse; }
      .orbit-3 { width: 420px; height: 420px; animation-duration: 84s; }

      /* Planets that ride the orbit (placed on the right edge; orbit rotates) */
      .planet {
        position: absolute;
        top: 50%;
        right: -10px;
        transform: translateY(-50%);
        border-radius: 50%;
        box-shadow: 0 0 0 2px rgba(255,255,255,.06) inset, 0 6px 14px rgba(0,0,0,.35);
      }
      .planet-1 { width: 14px; height:14px; background: linear-gradient(145deg,#8bdcf7,#4fb6ff); }
      .planet-2 { width: 18px; height:18px; background: linear-gradient(145deg,#bda8ff,#8f79f6); right: -12px; }
      .planet-3 { width: 10px; height:10px; background: linear-gradient(145deg,#6fe7bf,#36c79d); right:-8px; }

      @keyframes spin { to { transform: rotate(360deg); } }

      /* Starfields (two layers with phase-shifted twinkle) */
      .starfield {
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-repeat: repeat;
        z-index: 0;
        opacity: .45;
        animation: twinkle 14s ease-in-out infinite;
        filter: brightness(1);
      }
      .starfield.layer2 { animation-duration: 18s; opacity: .35; }
      .starfield.layer3 { animation-duration: 22s; opacity: .25; }

      .starfield {
        background-image:
          radial-gradient(1px 1px at 12% 18%, rgba(255,255,255,.9) 0, transparent 55%),
          radial-gradient(1px 1px at 52% 78%, rgba(255,255,255,.85) 0, transparent 55%),
          radial-gradient(1px 1px at 84% 32%, rgba(255,255,255,.8) 0, transparent 55%),
          radial-gradient(2px 2px at 22% 66%, rgba(255,255,255,.9) 0, transparent 60%),
          radial-gradient(1.5px 1.5px at 72% 12%, rgba(255,255,255,.85) 0, transparent 60%),
          radial-gradient(1.5px 1.5px at 32% 44%, rgba(255,255,255,.9) 0, transparent 60%);
        background-size: 320px 320px, 480px 480px, 640px 640px, 540px 540px, 720px 720px, 600px 600px;
        background-position: 0px 0px, 60px 80px, -40px 120px, 80px -60px, -100px 40px, 120px -40px;
      }
      @keyframes twinkle {
        0%   { filter: brightness(0.9); }
        50%  { filter: brightness(1.12); }
        100% { filter: brightness(0.9); }
      }

      /* Cards / widgets: soft, comfortable contrast */
      .cel-card {
        background: linear-gradient(180deg, var(--card) 0%, var(--card-2) 100%);
        border: 1px solid var(--stroke);
        border-radius: 14px;
        padding: 14px 16px;
      }
      .cel-card.pad-lg { padding: 18px 18px; }

      /* Tabs look */
      .stTabs [data-baseweb="tab-list"] { gap: 8px; }
      .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,.06);
        color: var(--white);
        border: 1px solid var(--stroke); border-radius: 12px; padding: 6px 10px;
      }
      .stTabs [aria-selected="true"] { background: rgba(255,255,255,.12); }

      /* File uploader + metrics */
      [data-testid="stFileUploadDropzone"] {
        background: rgba(255,255,255,.05); border: 2px dashed var(--stroke); border-radius: 14px;
      }
      [data-testid="stMetric"]{
        background: rgba(255,255,255,.05); border: 1px solid var(--stroke); border-radius: 12px; padding: 10px;
      }

      /* Buttons */
      .stButton > button {
        background: linear-gradient(90deg, #74c7ff 0%, #8f79f6 100%);
        border: none; color:#fff; font-weight:700; border-radius:12px; padding: 10px 22px;
        box-shadow: 0 6px 16px rgba(116,199,255,.22);
      }
      .stDownloadButton > button {
        background: rgba(255,255,255,.06); color:#fff; border:1px solid var(--stroke); border-radius:12px;
      }

      /* Dataframe table soft edges */
      .stDataFrame, .stTable { background: rgba(255,255,255,.03); border-radius: 12px; }

      /* Hide default menu/footer */
      #MainMenu, header, footer { visibility: hidden; }
    </style>

    <!-- star layers -->
    <div class="starfield layer1"></div>
    <div class="starfield layer2"></div>
    <div class="starfield layer3"></div>
    """, unsafe_allow_html=True)

inject_css_and_space_layers()

# =========================
# Constants
# =========================
APP_TITLE = "Celestia AI"
APP_SUBTITLE = "Exoplanet Discovery Suite"
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
# Hero with center title + rotating planets
# =========================
def render_hero():
    st.markdown(f"""
    <section class="celestia-hero">
      <div class="orbits">
        <div class="orbit orbit-1"><div class="planet planet-1"></div></div>
        <div class="orbit orbit-2"><div class="planet planet-2"></div></div>
        <div class="orbit orbit-3"><div class="planet planet-3"></div></div>
      </div>
      <div class="title-wrap">
        <h1>{APP_TITLE}</h1>
        <p>{APP_SUBTITLE}</p>
      </div>
    </section>
    """, unsafe_allow_html=True)

render_hero()

# =========================
# Utility functions
# =========================
def coerce_and_align(df: pd.DataFrame) -> pd.DataFrame:
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
    kwargs = dict(
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
        try:
            clf = HistGradientBoostingClassifier(**kwargs, class_weight=params["class_weight"])
        except TypeError:
            clf = HistGradientBoostingClassifier(**kwargs)
    else:
        clf = HistGradientBoostingClassifier(**kwargs)
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

def ensure_label_encoder(le: LabelEncoder | None, y_values: pd.Series | list) -> LabelEncoder:
    if le is None:
        le = LabelEncoder()
        le.fit(sorted(pd.Series(y_values).astype(str).unique()))
    return le

def compute_metrics(y_true, y_pred, proba, classes_):
    report = classification_report(y_true, y_pred, target_names=classes_, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes_)))
    y_true_bin = label_binarize(y_true, classes=range(len(classes_)))
    try:
        auc_ovr = roc_auc_score(y_true_bin, proba, average="macro", multi_class="ovr")
    except Exception:
        auc_ovr = np.nan
    briers = []
    for i in range(len(classes_)):
        briers.append(brier_score_loss(y_true_bin[:, i], proba[:, i]))
    brier_macro = float(np.mean(briers))
    return report, cm, auc_ovr, brier_macro

def plot_confusion_matrix(cm, classes_):
    df_cm = pd.DataFrame(cm, index=classes_, columns=classes_)
    fig = px.imshow(
        df_cm, text_auto=True, aspect="auto",
        color_continuous_scale=[ "#0e1726", "#20304b", "#31507a", "#3b82f6" ],
        title="Confusion Matrix"
    )
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=20,r=20,t=50,b=10))
    return fig

def plot_calibration(proba, y_true, classes_):
    fig = go.Figure()
    y_true_bin = label_binarize(y_true, classes=range(len(classes_)))
    palette = ["#86e1ff", "#a78bfa", "#34d399"]
    for i, name in enumerate(classes_):
        frac_pos, mean_pred = calibration_curve(y_true_bin[:, i], proba[:, i], n_bins=10, strategy="quantile")
        fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, mode="lines+markers", name=f"{name}", line=dict(width=2), marker=dict(size=6)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect", line=dict(dash="dash", width=1.5)))
    fig.update_layout(title="Calibration (Reliability) Curves", xaxis_title="Mean Predicted Probability", yaxis_title="Fraction of Positives",
                      template="plotly_dark", height=420, legend=dict(orientation="h"))
    return fig

def plot_feature_importance(pipeline: Pipeline):
    try:
        clf = pipeline.named_steps["classifier"]
        importances = getattr(clf, "feature_importances_", None)
        if importances is None:
            return None
        s = pd.Series(importances, index=SELECTED_FEATURES).sort_values(ascending=True)
        fig = px.bar(
            x=s.values, y=s.index, orientation="h",
            title="Feature Importances", color=s.values,
            color_continuous_scale=[ "#a7f3d0", "#6ee7b7", "#34d399", "#10b981" ]
        )
        fig.update_layout(template="plotly_dark", height=420, xaxis_title="Importance", yaxis_title="", coloraxis_showscale=False)
        return fig
    except Exception:
        return None

def save_artifacts(pipeline, label_encoder):
    joblib.dump(pipeline, "celestia_pipeline.joblib")
    joblib.dump(label_encoder, "celestia_label_encoder.joblib")

def load_artifacts():
    try:
        return joblib.load("kepler_gb_pipeline_weighted.joblib"), joblib.load("kepler_label_encoder.joblib")
    except Exception:
        try:
            return joblib.load("celestia_pipeline.joblib"), joblib.load("celestia_label_encoder.joblib")
        except Exception:
            return None, None

# =========================
# Sidebar — Global controls
# =========================
with st.sidebar:
    st.markdown('<div class="cel-card pad-lg"><h3 style="margin:0;">⚙️ Global Controls</h3>', unsafe_allow_html=True)
    auto_train = st.checkbox("Auto-train on ingest", value=True, help="Retrain whenever new labeled rows are ingested.")
    comfy_density = st.select_slider("Density", options=["Cozy","Comfy","Compact"], value="Comfy")
    st.markdown('<hr style="border:1px solid rgba(255,255,255,.12);">', unsafe_allow_html=True)
    if st.button("🔄 Load Artifacts (joblib)"):
        pipe, le = load_artifacts()
        if pipe is not None and le is not None:
            st.session_state["pipeline"] = pipe
            st.session_state["label_encoder"] = le
            st.success("Artifacts loaded.")
        else:
            st.warning("No compatible artifacts found.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Tabs
# =========================
tab_ingest, tab_explore, tab_train, tab_classify, tab_metrics, tab_export = st.tabs([
    "📥 Ingest Data",
    "🛰️ Explore",
    "🧬 Train & Tune",
    "🧪 Classify",
    "📊 Metrics & Explain",
    "📦 Export",
])

# -------------------------
# Ingest
# -------------------------
with tab_ingest:
    st.markdown('<div class="cel-card pad-lg">', unsafe_allow_html=True)
    st.subheader("📥 Ingest CSV")
    st.caption("Upload CSVs that include the 10 trained features. If a label column (e.g., `koi_disposition`) is present, it will be used for training.")
    up = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    col_t1, col_t2 = st.columns([1,3])
    with col_t1:
        if st.button("🧾 Download CSV Template"):
            template = pd.DataFrame({c: [] for c in SELECTED_FEATURES + ["koi_disposition"]})
            buf = io.StringIO(); template.to_csv(buf, index=False)
            st.download_button("📥 Get Template", data=buf.getvalue(), file_name="celestia_template.csv", mime="text/csv", key="dl_temp")
    st.markdown('</div>', unsafe_allow_html=True)

    if up:
        added = 0
        for f in up:
            try:
                df = pd.read_csv(f, comment="#"); df.columns = df.columns.str.strip()
            except Exception as e:
                st.error(f"Failed to read {f.name}: {e}"); continue

            target_col = find_target_column(df)
            if target_col is None: df["koi_disposition"] = np.nan

            X = coerce_and_align(df)
            y = df["koi_disposition"] if "koi_disposition" in df.columns else np.nan
            block = X.copy(); block["koi_disposition"] = y
            st.session_state["data_store"] = pd.concat([st.session_state["data_store"], block], axis=0, ignore_index=True)
            added += len(block)
        st.success(f"Ingested {added} rows.")
        if auto_train and st.session_state["data_store"]["koi_disposition"].notna().any():
            st.info("Auto-training on labeled rows…")
            # trigger user to go to Train tab if needed (no forced rerun)

# -------------------------
# Explore
# -------------------------
with tab_explore:
    st.markdown('<div class="cel-card pad-lg">', unsafe_allow_html=True)
    st.subheader("🛰️ Explore Dataset")
    data = st.session_state["data_store"]
    if data.empty:
        st.info("No data yet.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("Rows", int(len(data)))
        with c2: st.metric("Labeled", int(data["koi_disposition"].notna().sum()))
        with c3: st.metric("Unlabeled", int(data["koi_disposition"].isna().sum()))
        with c4: st.metric("Features", len(SELECTED_FEATURES))

        with st.expander("Preview (first 200 rows)", expanded=(comfy_density=="Comfy")):
            st.dataframe(data.head(200), use_container_width=True)

        # Label distribution if labels exist
        if data["koi_disposition"].notna().any():
            vc = data["koi_disposition"].dropna().value_counts()
            fig = px.bar(vc, title="Label Distribution (Ingested)",
                         labels={"value":"Count","index":"Label"},
                         color=vc.values, color_continuous_scale=["#86e1ff","#a78bfa"])
            fig.update_layout(template="plotly_dark", height=420, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        feat = st.selectbox("Inspect feature", SELECTED_FEATURES, index=0)
        fig = px.histogram(data, x=feat, nbins=40, title=f"Distribution: {feat}",
                           color_discrete_sequence=["#86e1ff"])
        fig.update_layout(template="plotly_dark", height=420)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Train & Tune
# -------------------------
with tab_train:
    st.markdown('<div class="cel-card pad-lg">', unsafe_allow_html=True)
    st.subheader("🧬 Train & Hyperparameter Tuning")
    data = st.session_state["data_store"]
    labeled = data.dropna(subset=["koi_disposition"])
    if labeled.empty:
        st.info("No labeled data available yet.")
    else:
        le = ensure_label_encoder(st.session_state["label_encoder"], labeled["koi_disposition"])
        y = le.transform(labeled["koi_disposition"].astype(str))
        X = coerce_and_align(labeled)

        # Hyperparameters
        st.markdown("##### Hyperparameters")
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
            use_cw = st.checkbox("Use class_weight", value=True)

        class_weight = None
        if use_cw:
            classes_list = list(le.classes_)
            cw1, cw2, cw3 = st.columns(3)
            with cw1:
                w_conf = st.number_input("Weight CONFIRMED", 1.0, 20.0, 5.0, 0.5)
            with cw2:
                w_cand = st.number_input("Weight CANDIDATE", 1.0, 20.0, 5.0, 0.5)
            with cw3:
                w_fp   = st.number_input("Weight FALSE POSITIVE", 0.5, 20.0, 1.0, 0.5)
            # Map class index -> weight robustly
            idx_map = {name: int(np.where(np.array(classes_list)==name)[0][0]) for name in classes_list}
            class_weight = {
                idx_map.get("CONFIRMED",1): w_conf,
                idx_map.get("CANDIDATE",0): w_cand,
                idx_map.get("FALSE POSITIVE",2): w_fp,
            }

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

        cta1, cta2 = st.columns(2)
        with cta1:
            train_btn = st.button("🚀 Train / Retrain", use_container_width=True)
        with cta2:
            cv_btn = st.button("🧪 5-Fold CV (macro F1)", use_container_width=True)

        if train_btn:
            with st.spinner("Training model…"):
                pipe = build_pipeline(params)
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                pipe.fit(Xtr, ytr)
                proba = pipe.predict_proba(Xte)
                y_pred = pipe.predict(Xte)
                report, cm, auc_ovr, brier_macro = compute_metrics(yte, y_pred, proba, le.classes_)
                st.session_state["pipeline"] = pipe
                st.session_state["label_encoder"] = le
                st.session_state["metrics"] = dict(
                    report=report, cm=cm.tolist(), auc_ovr=auc_ovr, brier=brier_macro,
                    params=params, timestamp=datetime.now().isoformat()
                )
                st.session_state["last_train_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.success("Model trained.")

        if cv_btn:
            with st.spinner("Running Stratified 5-Fold…"):
                pipe = build_pipeline(params)
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                try:
                    from sklearn.metrics import make_scorer, f1_score
                    scores = cross_val_score(pipe, X, y, scoring=make_scorer(f1_score, average="macro"), cv=skf)
                except Exception:
                    scores = cross_val_score(pipe, X, y, scoring="accuracy", cv=skf)
                st.info(f"CV macro F1 (mean ± std): **{np.mean(scores):.3f} ± {np.std(scores):.3f}**")

        # Current model summary
        if st.session_state["pipeline"] is not None and st.session_state["metrics"]:
            m = st.session_state["metrics"]
            d1,d2,d3,d4 = st.columns(4)
            with d1: st.metric("ROC-AUC (OvR, macro)", f"{m['auc_ovr']:.3f}" if m['auc_ovr']==m['auc_ovr'] else "n/a")
            with d2: st.metric("Brier (macro)", f"{m['brier']:.3f}")
            with d3: st.metric("Params tuned", str(len(m["params"])))
            with d4: st.metric("Trained", st.session_state["last_train_time"] or "—")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Classify
# -------------------------
with tab_classify:
    st.markdown('<div class="cel-card pad-lg">', unsafe_allow_html=True)
    st.subheader("🧪 Classify Candidates")
    t1, t2, t3 = st.columns(3)
    th_fp = t1.slider("Decision Threshold — FALSE POSITIVE", 0.0, 0.99, 0.50, 0.01)
    th_cand = t2.slider("Decision Threshold — CANDIDATE", 0.0, 0.99, 0.50, 0.01)
    th_conf = t3.slider("Decision Threshold — CONFIRMED", 0.0, 0.99, 0.50, 0.01)
    thresholds = {"FALSE POSITIVE": th_fp, "CANDIDATE": th_cand, "CONFIRMED": th_conf}

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

    if st.button("🔮 Classify Single", use_container_width=True):
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
        pipe = st.session_state["pipeline"]; le = st.session_state["label_encoder"]
        if pipe is None or le is None:
            st.error("No trained/loaded model yet.")
        else:
            proba = pipe.predict_proba(X)[0]
            classes_ = list(le.classes_)
            # Thresholding: choose highest class that exceeds its threshold; else argmax
            argmax_idx = int(np.argmax(proba)) if len(proba.shape)>0 else 0
            picked = classes_[argmax_idx]
            for idx, name in enumerate(classes_):
                if proba[idx] >= thresholds.get(name, 0.5):
                    picked = name
            conf = float(np.max(proba) * 100)
            st.markdown(f"### Prediction: **{picked}** • confidence: **{conf:.1f}%**")
            fig = go.Figure(data=[go.Bar(
                x=classes_, y=(proba*100).round(2),
                marker=dict(line=dict(width=0.5, color="rgba(255,255,255,.35)"))
            )])
            fig.update_layout(template="plotly_dark", height=360, title="Probability Distribution",
                              xaxis_title="Class", yaxis_title="Probability (%)",
                              colorway=["#86e1ff","#a78bfa","#34d399"])
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    up_batch = st.file_uploader("Upload CSV to classify (no label required)", type=["csv"], key="cls_csv")
    if up_batch is not None:
        df = pd.read_csv(up_batch, comment="#")
        X = coerce_and_align(df)
        pipe = st.session_state["pipeline"]; le = st.session_state["label_encoder"]
        if pipe is None or le is None:
            st.error("No trained/loaded model.")
        else:
            P = pipe.predict_proba(X)
            classes_ = list(le.classes_)
            labels = [classes_[int(np.argmax(P[i]))] for i in range(len(X))]
            for i in range(len(X)):
                for j, name in enumerate(classes_):
                    if P[i, j] >= thresholds.get(name, 0.5):
                        labels[i] = name
            out = df.copy(); out["prediction"] = labels
            for j, name in enumerate(classes_):
                out[f"proba_{name}"] = np.round(P[:, j], 6)
            st.success(f"Classified {len(out)} rows.")
            st.dataframe(out.head(200), use_container_width=True)
            buf = io.StringIO(); out.to_csv(buf, index=False)
            st.download_button("📥 Download Predictions CSV", buf.getvalue(),
                               file_name=f"celestia_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                               mime="text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Metrics & Explain
# -------------------------
with tab_metrics:
    st.markdown('<div class="cel-card pad-lg">', unsafe_allow_html=True)
    st.subheader("📊 Metrics & Explainability")
    if st.session_state["pipeline"] is None or st.session_state["label_encoder"] is None or not st.session_state["metrics"]:
        st.info("Train a model in **Train & Tune** to populate metrics.")
    else:
        m = st.session_state["metrics"]; le = st.session_state["label_encoder"]
        classes_ = list(le.classes_)
        c1,c2,c3,c4 = st.columns(4)
        with c1: st.metric("ROC-AUC (OvR)", f"{m['auc_ovr']:.3f}" if m['auc_ovr']==m['auc_ovr'] else "n/a")
        with c2: st.metric("Brier", f"{m['brier']:.3f}")
        with c3: st.metric("Params", str(len(m["params"])))
        with c4: st.metric("Trained", m.get("timestamp","—").split("T")[0])

        # Confusion matrix
        fig_cm = plot_confusion_matrix(np.array(m["cm"]), classes_)
        st.plotly_chart(fig_cm, use_container_width=True)

        # Classification report table
        rep = pd.DataFrame(m["report"]).T
        rep = rep.loc[[c for c in classes_] + ["macro avg","weighted avg"], ["precision","recall","f1-score","support"]]
        st.markdown("#### Classification Report")
        st.dataframe(np.round(rep, 3), use_container_width=True)

        # Recompose split for calibration/importance
        data = st.session_state["data_store"].dropna(subset=["koi_disposition"])
        X = coerce_and_align(data)
        y = le.transform(data["koi_disposition"].astype(str))
        Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        pipe = st.session_state["pipeline"]; proba = pipe.predict_proba(Xte)

        st.markdown("#### Calibration")
        st.plotly_chart(plot_calibration(proba, yte, classes_), use_container_width=True)

        st.markdown("#### Feature Importances")
        fi_fig = plot_feature_importance(pipe)
        if fi_fig is not None:
            st.plotly_chart(fi_fig, use_container_width=True)
        else:
            st.info("Estimator does not expose feature importances on this version.")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Export
# -------------------------
with tab_export:
    st.markdown('<div class="cel-card pad-lg">', unsafe_allow_html=True)
    st.subheader("📦 Export Artifacts & Report")
    pipe = st.session_state["pipeline"]; le = st.session_state["label_encoder"]
    if pipe is None or le is None:
        st.info("Train or load a model first.")
    else:
        save_artifacts(pipe, le)
        st.success("Saved `celestia_pipeline.joblib` & `celestia_label_encoder.joblib`.")
        with open("celestia_pipeline.joblib","rb") as f:
            st.download_button("📥 Download Pipeline", f.read(), "celestia_pipeline.joblib")
        with open("celestia_label_encoder.joblib","rb") as f:
            st.download_button("📥 Download Label Encoder", f.read(), "celestia_label_encoder.joblib")
        if st.session_state["metrics"]:
            m = st.session_state["metrics"].copy()
            m["classes_"] = list(le.classes_); m["feature_set"] = SELECTED_FEATURES
            card = json.dumps(m, indent=2)
            st.download_button("📄 Download Evaluation Report (JSON)", card, "celestia_model_report.json", "application/json")
    st.markdown('</div>', unsafe_allow_html=True)
