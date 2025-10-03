# app.py
# Celestial Ai (Kepler 10-feature edition) — Streamlit app
# Beautiful UI (Discord-style), no synthetic random generation for analysis,
# Three planets orbiting around the title, batch + single classify, all visuals displayed together,
# and a "Pretrained & Retrain" tab to optionally retrain from a user CSV.
# Default sample CSV is bundled and used for the "Analyze 10 from sample" button.

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import warnings
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Celestial Ai - NASA Exoplanet Detection",
    layout="wide",
    page_icon="🌌",
    initial_sidebar_state="expanded",
)

# =========================
# Constants / Features
# =========================
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

# Path to the bundled sample CSV (used for the 10-row sample analysis button)
SAMPLE_CSV_PATH = "/mnt/data/Celestial_AI___Synthetic_Kepler_10-feature_Data_with_Labels__preview_.csv"

# =========================
# CSS / Theme (keep visuals; force black/white text to blue/purple hues)
# =========================
def inject_custom_css():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Grotesk:wght@300;400;700&display=swap');

            /* Main animated gradient background */
            .stApp {
                background: linear-gradient(-45deg, #0a0118, #1a0033, #0f0c29, #24243e);
                background-size: 400% 400%;
                animation: gradientShift 15s ease infinite;
                overflow-x: hidden;
            }
            @keyframes gradientShift {
                0%{background-position:0% 50%}
                50%{background-position:100% 50%}
                100%{background-position:0% 50%}
            }

            /* Star fields (parallax layers) */
            .stars, .stars2, .stars3 {
                position: fixed; top:0; left:0; width:100%; height:100%; pointer-events:none; z-index:0;
            }
            .stars {
                background-image:
                    radial-gradient(2px 2px at 20px 30px, white, transparent),
                    radial-gradient(2px 2px at 40px 70px, white, transparent),
                    radial-gradient(1px 1px at 50px 50px, white, transparent),
                    radial-gradient(1px 1px at 80px 10px, white, transparent),
                    radial-gradient(2px 2px at 130px 80px, white, transparent);
                background-size:200px 200px; opacity:0.15;
                animation: stars 60s linear infinite;
            }
            @keyframes stars { 0% {transform: translateY(0);} 100% {transform: translateY(-120px);} }
            .stars2 {
                background-image:
                    radial-gradient(1px 1px at 60px 120px, #bbddff, transparent),
                    radial-gradient(2px 2px at 150px 90px, #ffffff, transparent),
                    radial-gradient(1px 1px at 200px 40px, #aaccff, transparent),
                    radial-gradient(2px 2px at 300px 160px, #ffffff, transparent);
                background-size:300px 300px; opacity:0.12;
                animation: stars2 120s linear infinite;
            }
            @keyframes stars2 { 0% {transform: translateY(0);} 100% {transform: translateY(-180px);} }
            .stars3 {
                background-image:
                    radial-gradient(1px 1px at 120px 60px, #88c0ff, transparent),
                    radial-gradient(2px 2px at 240px 180px, #ffffff, transparent),
                    radial-gradient(1px 1px at 400px 120px, #cfe8ff, transparent);
                background-size:500px 500px; opacity:0.08;
                animation: stars3 240s linear infinite;
            }
            @keyframes stars3 { 0% {transform: translateY(0);} 100% {transform: translateY(-220px);} }

            /* Nebula glow */
            .nebula {
                position: fixed; width: 600px; height: 600px; pointer-events: none; z-index: 0;
                opacity: 0.15; filter: blur(100px);
            }
            .nebula-1 {
                top: -200px; left: -200px;
                background: radial-gradient(circle, rgba(138, 43, 226, 0.4) 0%, transparent 70%);
                animation: nebulaPulse 20s infinite ease-in-out;
            }
            .nebula-2 {
                bottom: -200px; right: -200px;
                background: radial-gradient(circle, rgba(0, 191, 255, 0.4) 0%, transparent 70%);
                animation: nebulaPulse 25s infinite ease-in-out 5s;
            }
            @keyframes nebulaPulse { 0%,100% {transform: scale(1) rotate(0deg);} 50% {transform: scale(1.2) rotate(180deg);} }

            /* Title + orbit container */
            .orbit-title-container {
                position: relative;
                width: 480px;
                height: 220px;
                margin: 0 auto 10px auto;
                display: flex;
                align-items: center;
                justify-content: center;
                transform: translateZ(0);
            }

            /* Title text */
            .celestial-title {
                font-family:'Orbitron', monospace !important; font-weight:900 !important;
                background: linear-gradient(120deg, #00e5ff, #c77dff, #00e5ff);
                background-size:200% auto; background-clip:text; -webkit-background-clip:text;
                -webkit-text-fill-color:transparent; animation: shine 3s linear infinite;
                font-size: 3.2rem; text-align:center; margin: 0; line-height: 1.1;
                text-shadow: 0 0 40px rgba(0,255,255,0.5);
            }
            @keyframes shine { to { background-position: 200% center; } }

            /* Circular orbits around the title */
            .orbit-ring {
                position: absolute; border: 1px dashed rgba(120, 180, 255, 0.25); border-radius: 50%;
                animation: orbitRotate 20s linear infinite;
            }
            .ring-1 { width: 360px; height: 360px; }
            .ring-2 { width: 420px; height: 420px; animation-duration: 26s; animation-direction: reverse; }
            .ring-3 { width: 300px; height: 300px; animation-duration: 32s; }

            @keyframes orbitRotate { 0%{transform: rotate(0deg);} 100%{transform: rotate(360deg);} }

            /* Planets that orbit along the rings */
            .planet {
                position: absolute; top: -6px; left: 50%; transform: translateX(-50%);
                width: 18px; height: 18px; border-radius: 50%;
                box-shadow: 0 0 12px rgba(0, 255, 255, 0.6);
            }
            .planet-1 { background: radial-gradient(circle at 30% 30%, #7ad0ff, #2b6cb0 60%, #1a365d 100%); }
            .planet-2 { background: radial-gradient(circle at 30% 30%, #ffd27a, #c05621 60%, #7b341e 100%); width: 22px; height: 22px; top: -8px; }
            .planet-3 { background: radial-gradient(circle at 30% 30%, #e0b3ff, #8a2be2 60%, #4b0082 100%); width: 14px; height: 14px; top: -5px; }

            /* Subheading pill under the title */
            .title-pill {
                text-align:center; margin: 6px auto 16px auto;
                display:inline-block; padding: 6px 16px; border-radius: 24px;
                color:#fff; font-size: 0.95rem; font-family:'Space Grotesk', sans-serif;
                background: linear-gradient(90deg,#667eea,#764ba2);
                box-shadow:0 6px 20px rgba(102,126,234,0.4);
            }

            /* General headings */
            h2, h3 {
                font-family:'Space Grotesk', sans-serif !important;
                color:#d7d7ff !important; text-shadow:0 0 12px rgba(100,100,255,0.35);
            }

            /* Tabs container */
            .stTabs [data-baseweb="tab-list"] {
                background: rgba(255,255,255,0.06); backdrop-filter: blur(10px);
                border-radius:16px; padding:10px; border:1px solid rgba(255,255,255,0.12);
                box-shadow: 0 8px 32px rgba(0,0,0,0.25);
            }
            .stTabs [data-baseweb="tab"] {
                color:#a0a0ff !important; font-family:'Space Grotesk', sans-serif !important;
                font-weight:700; transition: all .25s ease; position: relative; overflow: hidden;
            }
            .stTabs [data-baseweb="tab"]:hover { transform: translateY(-2px); color: #71b7ff !important; }
            .stTabs [aria-selected="true"] {
                background: linear-gradient(90deg, rgba(0,255,255,0.18), rgba(255,0,255,0.18));
                border-radius:12px; box-shadow: 0 4px 18px rgba(0,255,255,0.28);
            }

            /* Inputs: blue theme */
            .stTextInput input, .stNumberInput input, .stSelectbox select, .stSlider > div > div {
                background: rgba(255,255,255,0.06) !important; border: 2px solid rgba(0,200,255,0.35) !important;
                color:#e8f1ff !important; border-radius:12px !important; backdrop-filter: blur(5px);
                transition: all .25s ease; box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
            }
            .stTextInput input:focus, .stNumberInput input:focus {
                border-color:#66e0ff !important; box-shadow:0 0 28px rgba(0,225,255,0.35), inset 0 2px 4px rgba(0,0,0,0.2) !important;
                transform: scale(1.01);
            }

            /* Force all labels/paragraphs to blue/purple hues */
            label, .stMarkdown p, .stSelectbox label, .stNumberInput label, .stTextInput label, .stSlider label, .stDownloadButton button span, .stButton button span {
                color: #97b3ff !important;
            }

            /* Buttons (neon-blue vibe) */
            .stButton > button, .stDownloadButton > button {
                background: linear-gradient(135deg, #5ba8ff 0%, #7a5cff 100%);
                color:white !important; border:none; padding:12px 34px; border-radius:32px;
                font-weight:800; font-family:'Space Grotesk', sans-serif;
                transition: all .25s cubic-bezier(0.2, 0.8, 0.2, 1);
                box-shadow:0 10px 30px rgba(91,168,255,0.35);
                position: relative; overflow: hidden;
            }
            .stButton > button:hover, .stDownloadButton > button:hover {
                transform: translateY(-3px) scale(1.03);
                box-shadow:0 12px 38px rgba(122,92,255,0.45);
            }

            /* Alerts, metrics, upload dropzone */
            .stAlert {
                background: rgba(255,255,255,0.06) !important; backdrop-filter: blur(10px);
                border:2px solid rgba(255,255,255,0.22); border-radius:16px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            }
            [data-testid="metric-container"] {
                background: rgba(255,255,255,0.06); backdrop-filter: blur(10px);
                border:2px solid rgba(255,255,255,0.12); padding:16px; border-radius:16px;
                box-shadow:0 8px 26px rgba(0,0,0,0.3);
                transition: all .25s ease;
            }
            [data-testid="metric-container"]:hover {
                transform: translateY(-3px) scale(1.01); border-color: rgba(0,200,255,0.35);
                box-shadow:0 10px 34px rgba(0,200,255,0.25);
            }
            [data-testid="stFileUploadDropzone"] {
                background: rgba(255,255,255,0.05);
                border:3px dashed rgba(0,200,255,0.35); border-radius:18px;
                transition: all .25s ease;
            }
            [data-testid="stFileUploadDropzone"]:hover {
                background: rgba(0,200,255,0.10); border-color: rgba(0,255,255,0.65);
                box-shadow: 0 0 22px rgba(0,255,255,0.25);
            }

            /* Progress bar */
            .stProgress > div > div > div {
                background: linear-gradient(90deg, #00ffff, #ff00ff);
                border-radius:10px; animation: pulse 2s infinite;
                box-shadow: 0 0 20px rgba(0,255,255,0.4);
            }
            @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.85} }

            /* Scrollbar */
            ::-webkit-scrollbar { width:12px; background: rgba(255,255,255,0.05); }
            ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #667eea, #764ba2); border-radius:10px; border: 2px solid rgba(255,255,255,0.1); }

            /* Hide default menu */
            #MainMenu, header, footer { visibility: hidden; }

            .main .block-container { position:relative; z-index:5; padding-top: 1.5rem; }
        </style>
        <div class="stars"></div>
        <div class="stars2"></div>
        <div class="stars3"></div>
        <div class="nebula nebula-1"></div>
        <div class="nebula nebula-2"></div>
        """,
        unsafe_allow_html=True,
    )

inject_custom_css()

# =========================
# Utilities
# =========================
@st.cache_resource
def load_model_safe(file_or_path):
    try:
        return joblib.load(file_or_path)
    except Exception:
        return None

def align_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the dataframe has exactly the required 10 features, numeric, in order.
    NOTE: The loaded sklearn Pipeline handles imputation + scaling internally.
    """
    X = df.copy()
    for col in SELECTED_FEATURES:
        if col not in X.columns:
            X[col] = 0 if col.startswith("koi_fpflag_") else 0.0
    for col in SELECTED_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    return X[SELECTED_FEATURES]

def decode_labels(y_pred_int: np.ndarray, label_encoder) -> np.ndarray:
    try:
        if hasattr(label_encoder, "inverse_transform"):
            return label_encoder.inverse_transform(y_pred_int)
        classes_ = getattr(label_encoder, "classes_", np.array(["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]))
        return np.array([classes_[i] for i in y_pred_int])
    except Exception:
        classes_ = np.array(["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"])
        return np.array([classes_[i] for i in y_pred_int])

# =========================
# Header with orbiting planets
# =========================
def show_header():
    st.markdown(
        """
        <div class="orbit-title-container">
            <!-- Rings -->
            <div class="orbit-ring ring-1">
                <div class="planet planet-1"></div>
            </div>
            <div class="orbit-ring ring-2">
                <div class="planet planet-2"></div>
            </div>
            <div class="orbit-ring ring-3">
                <div class="planet planet-3"></div>
            </div>
            <!-- Title -->
            <h1 class="celestial-title">🌌 Celestial Ai</h1>
        </div>
        <div style='text-align:center;'>
            <span class="title-pill">Advanced Exoplanet Detection • Kepler 10-Feature Model</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

show_header()

# =========================
# Sidebar: Model Controls
# =========================
with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center;padding:18px;background:rgba(255,255,255,0.05);
                    border-radius:16px;margin-bottom:16px;border:1px solid rgba(0,255,255,0.25);'>
            <h2 style='margin:0;font-size:1.45rem;'>🚀 Control Panel</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔧 Model Artifacts", expanded=True):
        model_upload = st.file_uploader("Upload Pipeline (.joblib)", type=["joblib"], key="model")
        encoder_upload = st.file_uploader("Upload Encoder (.joblib)", type=["joblib"], key="encoder")

    pipeline = None
    label_encoder = None

    if model_upload and encoder_upload:
        pipeline = load_model_safe(model_upload)
        label_encoder = load_model_safe(encoder_upload)
        if pipeline is not None and label_encoder is not None:
            st.success("✅ Custom artifacts loaded (uploaded).")
        else:
            st.error("❌ Failed to load uploaded artifacts. Please re-upload valid files.")
    else:
        pipeline = load_model_safe("gb_pipeline.joblib")
        label_encoder = load_model_safe("label_encoder.joblib")
        if pipeline is not None and label_encoder is not None:
            st.info("📁 Using local artifacts: gb_pipeline.joblib + label_encoder.joblib")
        else:
            st.warning("⚠️ Model artifacts not found. Upload valid .joblib files above to enable predictions/retraining.")

    st.markdown("### 📊 System Status")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Model", "✅ Ready" if (pipeline is not None and label_encoder is not None) else "❌ Missing")
    with c2:
        st.metric("Features", f"{len(SELECTED_FEATURES)} used", delta="Kepler 10-feature")

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Batch Analysis",
    "✨ Quick Classify",
    "🧠 Pretrained & Retrain",
    "📈 Visualizations",
    "ℹ️ About",
])

# -------------------------
# Tab 1: Batch Analysis
# -------------------------
with tab1:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.05);
                        border-radius:16px;margin-bottom:24px;border:1px solid rgba(0,255,255,0.22);'>
                <h3 style='margin:0;'>🚀 Batch Exoplanet Analysis</h3>
                <p style='color:#b2c5ff;margin-top:10px;'>
                    Upload a CSV with the <b>10 Kepler features</b>. Or analyze 10 random rows from the bundled sample CSV.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    uploaded_file = st.file_uploader(
        "📁 Upload CSV Dataset (must include the 10 feature columns)",
        type=["csv"],
        key="batch_csv",
    )

    # internal df holder
    df = None
    used_sample = False

    # Load from upload
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, comment="#")
            df.columns = df.columns.str.strip()
            st.success(f"✅ Loaded {len(df)} rows from uploaded file")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    # If no upload, show helper to analyze 10 rows from bundled CSV
    if df is None:
        st.markdown(
            "<div style='margin-top:8px;margin-bottom:12px;'>Or:</div>",
            unsafe_allow_html=True,
        )
        cbtn = st.columns([1, 1, 1])
        with cbtn[1]:
            if st.button("🔵 Analyze 10 Rows From Sample CSV", use_container_width=True, type="primary"):
                try:
                    df_sample = pd.read_csv(SAMPLE_CSV_PATH)
                    df_sample.columns = df_sample.columns.str.strip()
                    if len(df_sample) >= 10:
                        df = df_sample.sample(10, random_state=np.random.randint(0, 1_000_000)).reset_index(drop=True)
                        used_sample = True
                        st.success("✅ Took 10 random rows from the bundled sample CSV.")
                    else:
                        st.error("Sample CSV has fewer than 10 rows.")
                except Exception as e:
                    st.error(f"Could not load the bundled sample CSV: {e}")

    # If we have df (either uploaded or sampled), run analysis
    if df is not None and pipeline is not None and label_encoder is not None:
        with st.expander("📋 Dataset Preview (up to 10 rows)", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)

        c1r, c2r, c3r = st.columns([1, 1, 1])
        with c2r:
            if st.button("🔬 Run Analysis", use_container_width=True, type="primary"):
                with st.spinner("Analyzing candidates..."):
                    progress_bar = st.progress(0)
                    X = align_features_df(df)
                    progress_bar.progress(30)
                    preds = pipeline.predict(X)
                    probas = pipeline.predict_proba(X)
                    progress_bar.progress(70)
                    labels = decode_labels(preds, label_encoder)
                    results = df.copy()
                    results["prediction"] = labels
                    results["confidence"] = (probas.max(axis=1) * 100).round(2)
                    try:
                        classes = list(getattr(label_encoder, "classes_", ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]))
                        confirmed_idx = classes.index("CONFIRMED")
                    except ValueError:
                        confirmed_idx = -1
                    results["confirmed_prob"] = (
                        (probas[:, confirmed_idx] * 100).round(2) if confirmed_idx >= 0 else (probas.max(axis=1) * 100).round(2)
                    )
                    progress_bar.progress(100)
                    st.session_state["results"] = results
                    msg_source = "sample CSV" if used_sample else "uploaded CSV"
                    st.success(f"✅ Analysis complete! Processed {len(results)} rows from {msg_source}.")
                    time.sleep(0.2)

    # Show results / controls
    if "results" in st.session_state:
        results = st.session_state["results"]
        st.markdown("### 📊 Analysis Summary")
        c1m, c2m, c3m, c4m = st.columns(4)
        with c1m:
            confirmed = (results["prediction"] == "CONFIRMED").sum()
            st.metric("🌟 Confirmed", int(confirmed), delta=f"{confirmed/len(results)*100:.1f}%")
        with c2m:
            cand = (results["prediction"] == "CANDIDATE").sum()
            st.metric("🔍 Candidates", int(cand), delta=f"{cand/len(results)*100:.1f}%")
        with c3m:
            fp = (results["prediction"] == "FALSE POSITIVE").sum()
            st.metric("❌ False Positives", int(fp), delta=f"{fp/len(results)*100:.1f}%")
        with c4m:
            avg_conf = float(results["confidence"].mean())
            st.metric("💪 Avg Confidence", f"{avg_conf:.1f}%", delta="High" if avg_conf > 70 else "Moderate")

        st.markdown("### 🔎 Detailed Results")
        f1, f2, f3 = st.columns(3)
        with f1:
            filt = st.selectbox("Filter by prediction:", ["All", "CONFIRMED", "CANDIDATE", "FALSE POSITIVE"])
        with f2:
            min_conf = st.slider("Min confidence:", 0, 100, 0)
        with f3:
            sort_by = st.selectbox("Sort by:", ["confidence", "confirmed_prob", "koi_score", "koi_model_snr", "koi_period"])
        view = results.copy()
        if filt != "All":
            view = view[view["prediction"] == filt]
        view = view[view["confidence"] >= min_conf]
        if sort_by in view.columns:
            view = view.sort_values(sort_by, ascending=False)
        st.dataframe(view, use_container_width=True)

        csv = view.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name=f"exoplanet_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    elif pipeline is None or label_encoder is None:
        st.info("Upload model artifacts from the sidebar to enable analysis.")

# -------------------------
# Tab 2: Quick Classify (human-readable names, no defaults)
# -------------------------
with tab2:
    st.markdown(
        """
        <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.05);
                    border-radius:16px;margin-bottom:24px;border:1px solid rgba(0,255,255,0.22);'>
            <h3 style='margin:0;'>✨ Quick Candidate Classification</h3>
            <p style='color:#b2c5ff;margin-top:10px;'>Enter values for the trained 10 features. (No defaults.)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # We'll use text inputs (placeholders) and then validate, so there are no default values.
    colL, colR = st.columns(2)
    with colL:
        koi_score_txt = st.text_input("KOI Score (0–1)")
        koi_model_snr_txt = st.text_input("Transit Model Signal-to-Noise Ratio")
        koi_impact_txt = st.text_input("Impact Parameter (0–1.5)")
        koi_duration_txt = st.text_input("Transit Duration (hours)")
        koi_prad_txt = st.text_input("Planet Radius (Earth radii)")
    with colR:
        koi_period_txt = st.text_input("Orbital Period (days)")
        koi_fpflag_nt_txt = st.text_input("Not Transit-like Flag (0 or 1)")
        koi_fpflag_co_txt = st.text_input("Centroid Offset Flag (0 or 1)")
        koi_fpflag_ss_txt = st.text_input("Significant Secondary Flag (0 or 1)")
        koi_fpflag_ec_txt = st.text_input("Eclipsing Binary Flag (0 or 1)")

    def _parse_float(x, name):
        try:
            return float(x)
        except Exception:
            raise ValueError(f"'{name}' must be a number.")

    def _parse_int01(x, name):
        v = int(float(x))
        if v not in (0, 1):
            raise ValueError(f"'{name}' must be 0 or 1.")
        return v

    c1b, c2b, c3b = st.columns([1, 2, 1])
    with c2b:
        if st.button("🔮 Classify Candidate", use_container_width=True, type="primary"):
            if pipeline is None or label_encoder is None:
                st.error("Model not loaded. Upload valid artifacts in the sidebar.")
            else:
                try:
                    row = {
                        "koi_score": _parse_float(koi_score_txt, "KOI Score"),
                        "koi_fpflag_nt": _parse_int01(koi_fpflag_nt_txt, "Not Transit-like Flag"),
                        "koi_model_snr": _parse_float(koi_model_snr_txt, "Transit Model SNR"),
                        "koi_fpflag_co": _parse_int01(koi_fpflag_co_txt, "Centroid Offset Flag"),
                        "koi_fpflag_ss": _parse_int01(koi_fpflag_ss_txt, "Significant Secondary Flag"),
                        "koi_fpflag_ec": _parse_int01(koi_fpflag_ec_txt, "Eclipsing Binary Flag"),
                        "koi_impact": _parse_float(koi_impact_txt, "Impact Parameter"),
                        "koi_duration": _parse_float(koi_duration_txt, "Transit Duration"),
                        "koi_prad": _parse_float(koi_prad_txt, "Planet Radius"),
                        "koi_period": _parse_float(koi_period_txt, "Orbital Period"),
                    }
                except ValueError as ve:
                    st.error(str(ve))
                    st.stop()

                with st.spinner("Analyzing candidate..."):
                    progress_bar = st.progress(0)
                    X = align_features_df(pd.DataFrame([row]))
                    progress_bar.progress(40)
                    pred = pipeline.predict(X)[0]
                    proba = pipeline.predict_proba(X)[0]
                    label = decode_labels(np.array([pred]), label_encoder)[0]
                    confidence = float(proba.max() * 100)
                    progress_bar.progress(100)
                    time.sleep(0.1)

                if label == "CONFIRMED":
                    st.balloons()
                    color_block = ("rgba(0,255,0,0.1)", "rgba(0,255,255,0.55)")
                    title = "🌟 EXOPLANET CONFIRMED!"
                    sub = "This candidate shows strong signs of being a real exoplanet."
                elif label == "CANDIDATE":
                    color_block = ("rgba(255,255,0,0.12)", "rgba(255,255,0,0.55)")
                    title = "🔍 CANDIDATE"
                    sub = "Further observation needed to confirm."
                else:
                    color_block = ("rgba(255,0,0,0.12)", "rgba(255,0,0,0.55)")
                    title = "🚫 FALSE POSITIVE"
                    sub = "This signal is likely not from a real exoplanet."

                st.markdown(
                    f"""
                    <div style='text-align:center;padding:28px;background:linear-gradient(135deg,{color_block[0]},{color_block[0]});
                                border-radius:18px;border:2px solid {color_block[1]};'>
                        <h1 style='margin:0;'>{title}</h1>
                        <h2>Confidence: {confidence:.1f}%</h2>
                        <p style='color:#cfe3ff;'>{sub}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown("### 📊 Probability Distribution")
                classes = list(getattr(label_encoder, "classes_", ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]))
                fig = go.Figure(
                    data=[go.Bar(
                        x=classes,
                        y=(proba * 100).round(2),
                        text=[f"{p*100:.1f}%" for p in proba],
                        textposition="auto",
                    )]
                )
                fig.update_layout(
                    title="Classification Probabilities",
                    xaxis_title="Class",
                    yaxis_title="Probability (%)",
                    template="plotly_dark",
                    height=400,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Tab 3: Pretrained & Retrain (user can upload CSV to retrain)
# -------------------------
with tab3:
    st.markdown(
        """
        <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.05);
                    border-radius:16px;margin-bottom:24px;border:1px solid rgba(0,255,255,0.22);'>
            <h3 style='margin:0;'>🧠 Pretrained & Retrain</h3>
            <p style='color:#b2c5ff;margin-top:10px;'>Optionally upload a labeled CSV (with <code>koi_disposition</code>) to retrain the model.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    train_file = st.file_uploader(
        "Upload Training Dataset (CSV with 10 features + 'koi_disposition')",
        type=["csv"],
        key="training_file",
    )

    if train_file is not None:
        try:
            train_df = pd.read_csv(train_file, comment="#")
            train_df.columns = train_df.columns.str.strip()
            st.success(f"✅ Loaded {len(train_df)} training samples")
            with st.expander("📋 Training Data Preview"):
                st.dataframe(train_df.head(), use_container_width=True)

            colA, colB = st.columns(2)
            with colA:
                test_size = st.slider("Test size", 0.1, 0.4, 0.2, step=0.05)
            with colB:
                max_iter = st.number_input("Max Iterations", 50, 1000, 500, step=50)

            if st.button("🚀 Retrain Model", type="primary"):
                if "koi_disposition" not in train_df.columns:
                    st.error("Training CSV must include 'koi_disposition'.")
                else:
                    with st.spinner("Retraining model..."):
                        from sklearn.model_selection import train_test_split
                        from sklearn.preprocessing import StandardScaler
                        from sklearn.ensemble import HistGradientBoostingClassifier
                        from sklearn.compose import ColumnTransformer
                        from sklearn.pipeline import Pipeline
                        from sklearn.impute import SimpleImputer

                        train_le = LabelEncoder()
                        y = train_le.fit_transform(train_df["koi_disposition"].astype(str))

                        def align_features_df_train(df: pd.DataFrame) -> pd.DataFrame:
                            X = df.copy()
                            for col in SELECTED_FEATURES:
                                if col not in X.columns:
                                    X[col] = 0 if col.startswith("koi_fpflag_") else 0.0
                                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
                            return X[SELECTED_FEATURES]

                        X = align_features_df_train(train_df)
                        numeric_transformer = Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                                ("scaler", StandardScaler()),
                            ]
                        )
                        preprocessor = ColumnTransformer(
                            transformers=[("num", numeric_transformer, SELECTED_FEATURES)]
                        )
                        clf = HistGradientBoostingClassifier(
                            learning_rate=0.05,
                            max_iter=int(max_iter),
                            max_depth=6,
                            min_samples_leaf=20,
                            l2_regularization=1.0,
                            early_stopping=True,
                            validation_fraction=0.1,
                            n_iter_no_change=20,
                            random_state=42,
                        )
                        new_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

                        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
                        new_pipeline.fit(Xtr, ytr)
                        acc = new_pipeline.score(Xte, yte)

                        joblib.dump(new_pipeline, "gb_pipeline.joblib")
                        joblib.dump(train_le, "label_encoder.joblib")
                        st.success(f"✅ Retraining complete. Test accuracy: {acc:.2%}")

                        c1d, c2d = st.columns(2)
                        with c1d:
                            with open("gb_pipeline.joblib", "rb") as f:
                                st.download_button("📥 Download Pipeline", f.read(), "gb_pipeline.joblib", use_container_width=True)
                        with c2d:
                            with open("label_encoder.joblib", "rb") as f:
                                st.download_button("📥 Download Encoder", f.read(), "label_encoder.joblib", use_container_width=True)
        except Exception as e:
            st.error(f"Error during retraining: {e}")
    else:
        st.info("Upload a labeled CSV to retrain the model. (Your current artifacts are used for predictions.)")

# -------------------------
# Tab 4: Visualizations (show everything together, no selection)
# -------------------------
with tab4:
    st.markdown(
        """
        <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.05);
                    border-radius:16px;margin-bottom:24px;border:1px solid rgba(0,255,255,0.22);'>
            <h3 style='margin:0;'>📈 Data Visualizations</h3>
            <p style='color:#b2c5ff;margin-top:10px;'>All plots displayed together (Distribution, Confidence, Correlations, Radius vs Period, 3D Explorer).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "results" in st.session_state:
        results = st.session_state["results"].copy()

        # 1) Distribution (pie) and Confidence (hist)
        c_top1, c_top2 = st.columns(2)
        with c_top1:
            vc = results["prediction"].value_counts()
            if len(vc) > 0:
                fig = px.pie(
                    values=vc.values, names=vc.index, title="Classification Distribution",
                    color_discrete_sequence=['#6ea8fe', '#e599f7', '#63e6be']
                )
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(template="plotly_dark", height=420)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No predictions to visualize yet.")
        with c_top2:
            if "confidence" in results.columns:
                fig = px.histogram(results, x="confidence", nbins=20, title="Confidence Distribution")
                fig.update_layout(template="plotly_dark", height=420)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Confidence column not found.")

        # 2) Correlation heatmap
        st.markdown("### 🔬 Feature Correlations")
        numeric_cols = [c for c in SELECTED_FEATURES if c in results.columns]
        if len(numeric_cols) >= 3:
            corr = results[numeric_cols].corr()
            fig_heat = px.imshow(corr, title="Feature Correlation Heatmap", color_continuous_scale="Viridis")
            fig_heat.update_layout(template="plotly_dark", height=520)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("Not enough numeric features available for correlation heatmap.")

        # 3) Radius vs Period scatter
        st.markdown("### 🪐 Radius vs Period (confidence-scaled)")
        if all(col in results.columns for col in ["koi_prad", "koi_period"]):
            fig_scatter = px.scatter(
                results,
                x="koi_period",
                y="koi_prad",
                color="prediction",
                size="confidence" if "confidence" in results.columns else None,
                title="Radius vs Period",
            )
            fig_scatter.update_layout(template="plotly_dark", height=520)
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Required features missing: koi_period & koi_prad")

        # 4) 3D Explorer: SNR vs Radius vs Period
        st.markdown("### 🌌 3D Explorer: SNR vs Radius vs Period")
        need3d = ["koi_model_snr", "koi_prad", "koi_period"]
        if all(c in results.columns for c in need3d):
            fig3d = px.scatter_3d(
                results,
                x="koi_model_snr",
                y="koi_prad",
                z="koi_period",
                color="prediction",
                size="confidence" if "confidence" in results.columns else None,
                title="3D Explorer",
            )
            fig3d.update_layout(template="plotly_dark", height=720)
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            st.info("Need koi_model_snr, koi_prad, koi_period for 3D view.")
    else:
        st.info("📊 Run an analysis in the Batch Analysis tab to populate visualizations.")

# -------------------------
# Tab 5: About
# -------------------------
with tab5:
    st.markdown(
        """
        <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.05);
                    border-radius:16px;margin-bottom:24px;border:1px solid rgba(0,255,255,0.22);'>
            <h3 style='margin:0;'>ℹ️ About Celestial Ai</h3>
            <p style='color:#b2c5ff;margin-top:10px;'>NASA Space Apps Challenge 2025 — Kepler 10-feature model edition</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    colA, colB = st.columns([2, 1])
    with colA:
        st.markdown(
            """
            ### 🌌 Project Overview
            Celestial Ai integrates a trained Kepler pipeline that uses exactly 10 features:
            `koi_score, koi_fpflag_nt, koi_model_snr, koi_fpflag_co, koi_fpflag_ss, koi_fpflag_ec, koi_impact, koi_duration, koi_prad, koi_period`.
            It supports batch analysis, single-candidate classification, optional retraining, and rich interactive visuals.
            """
        )
    with colB:
        st.markdown(
            """
            ### 🏆 Highlights
            - Exact feature alignment to your model  
            - No synthetic demo generation for analysis  
            - Neon-blue/purple themed UI with animations  
            - Visuals include 3D explorer & correlations
            """
        )

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align:center;padding:16px;background:rgba(255,255,255,0.05);border-radius:14px;'>
            <h4 style='color:#77e1ff;margin:0;'>👨‍🚀 Built for NASA Space Apps 2025</h4>
            <p style='color:#b2c5ff;'>Exploring new worlds through AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# Footer
# =========================
st.markdown(
    """
    <div style='text-align:center;margin-top:36px;padding:16px;border-top:1px solid rgba(255,255,255,0.12);'>
        <p style='color:#9ab3ff;font-size:0.95rem;'>
            Celestial Ai • NASA Space Apps 2025 •
            <span style='color:#71e3ff;'>Kepler 10-Feature Pipeline</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

