# Celestial Ai (Kepler 10-feature edition) — Streamlit app
# Integrates trained pipeline: gb_pipeline.joblib + label_encoder.joblib
# Beautiful UI with Discord-style effects, NO RANDOM DATA, batch + single classify, retrain, and all-in-one visuals

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
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
import streamlit.components.v1 as components
from pathlib import Path
import base64

warnings.filterwarnings('ignore')

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Celestial Ai - NASA Exoplanet Detection",
    layout="wide",
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

# =========================
# Constants / Data Pathing (bundled default dataset)
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

APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

BUNDLED_CSV_NAME = "Celestial_AI___Synthetic_Kepler_10-feature_Data_with_Labels__preview_.csv"
BUNDLED_CSV_PATH = DATA_DIR / BUNDLED_CSV_NAME

# Embedded 10-row fallback (guarantees feature works on new device if bundled CSV is missing)
EMBEDDED_CSV_B64 = base64.b64encode(
    b"koi_score,koi_fpflag_nt,koi_model_snr,koi_fpflag_co,koi_fpflag_ss,koi_fpflag_ec,koi_impact,koi_duration,koi_prad,koi_period,koi_disposition\n"
    b"0.73,0,19.2,0,0,0,0.41,9.7,1.6,28.5,CONFIRMED\n"
    b"0.62,0,14.8,0,0,0,0.32,11.1,1.2,41.0,CANDIDATE\n"
    b"0.15,1,6.3,1,0,0,1.05,4.1,3.7,3.2,FALSE POSITIVE\n"
    b"0.58,0,22.9,0,0,0,0.27,12.2,1.9,74.4,CANDIDATE\n"
    b"0.81,0,33.5,0,0,0,0.55,8.9,1.3,15.8,CONFIRMED\n"
    b"0.09,1,5.1,0,1,1,1.12,2.7,0.9,1.9,FALSE POSITIVE\n"
    b"0.67,0,18.0,0,0,0,0.44,10.0,2.1,90.3,CANDIDATE\n"
    b"0.84,0,29.6,0,0,0,0.38,7.4,1.1,36.7,CONFIRMED\n"
    b"0.22,1,9.4,0,1,0,0.98,6.1,4.0,2.8,FALSE POSITIVE\n"
    b"0.76,0,24.3,0,0,0,0.29,11.4,1.4,52.2,CONFIRMED\n"
).decode("utf-8")


@st.cache_data(show_spinner=False)
def get_default_dataset() -> pd.DataFrame:
    """
    Load the default CSV from a repo-bundled relative path.
    If it's missing, create it from the embedded 10-row CSV so every new device still works.
    """
    if not BUNDLED_CSV_PATH.exists():
        try:
            raw = base64.b64decode(EMBEDDED_CSV_B64.encode("utf-8"))
            with open(BUNDLED_CSV_PATH, "wb") as f:
                f.write(raw)
        except Exception as e:
            raise RuntimeError(f"Failed to materialize embedded CSV fallback: {e}")
    df = pd.read_csv(BUNDLED_CSV_PATH)
    df.columns = df.columns.str.strip()
    return df

# =========================
# Enhanced CSS with Discord-style animations + orbiting planets + blue/purple labels
# =========================
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Grotesk:wght@300;400;700&display=swap');

        .stApp {
            background: linear-gradient(-45deg, #0a0118, #1a0033, #0f0c29, #24243e);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            overflow-x: hidden;
        }
        @keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

        /* Parallax star layers */
        .stars, .stars2, .stars3 { position:fixed; top:0; left:0; width:100%; height:100%; pointer-events:none; z-index:0; }
        .stars{
            background-image:
                radial-gradient(2px 2px at 20px 30px, white, transparent),
                radial-gradient(2px 2px at 40px 70px, white, transparent),
                radial-gradient(1px 1px at 50px 50px, white, transparent),
                radial-gradient(1px 1px at 80px 10px, white, transparent),
                radial-gradient(2px 2px at 130px 80px, white, transparent);
            background-repeat: repeat; background-size:200px 200px;
            animation: stars 60s linear infinite; opacity:0.15;
        }
        .stars2{
            background-image:
                radial-gradient(1px 1px at 60px 120px, #bbddff, transparent),
                radial-gradient(2px 2px at 150px 90px, #ffffff, transparent),
                radial-gradient(1px 1px at 200px 40px, #aaccff, transparent),
                radial-gradient(2px 2px at 300px 160px, #ffffff, transparent);
            background-repeat: repeat; background-size:300px 300px;
            animation: stars2 120s linear infinite; opacity:0.12;
        }
        .stars3{
            background-image:
                radial-gradient(1px 1px at 120px 60px, #88c0ff, transparent),
                radial-gradient(2px 2px at 240px 180px, #ffffff, transparent),
                radial-gradient(1px 1px at 400px 120px, #cfe8ff, transparent);
            background-repeat: repeat; background-size:500px 500px;
            animation: stars3 240s linear infinite; opacity:0.08;
        }
        @keyframes stars { 0% {transform: translateY(0);} 100% {transform: translateY(-120px);} }
        @keyframes stars2 { 0% {transform: translateY(0);} 100% {transform: translateY(-180px);} }
        @keyframes stars3 { 0% {transform: translateY(0);} 100% {transform: translateY(-220px);} }

        .main .block-container { position:relative; z-index:10; padding-top:2rem; }

        /* Animated title + orbiting planets wrapper */
        .title-wrap { position: relative; display: inline-block; padding: 30px 60px; }
        .orbit {
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            border-radius: 50%;
            border: 1px dashed rgba(100, 100, 255, 0.25);
            pointer-events: none;
        }
        .orbit.o1 { width: 380px; height: 380px; animation: orbitRotate 24s linear infinite; }
        .orbit.o2 { width: 540px; height: 540px; animation: orbitRotate 36s linear infinite reverse; }
        .orbit.o3 { width: 700px; height: 700px; animation: orbitRotate 48s linear infinite; }
        @keyframes orbitRotate { 0% { transform: translate(-50%, -50%) rotate(0deg);} 100% { transform: translate(-50%, -50%) rotate(360deg);} }

        .planet {
            position: absolute; border-radius: 50%; box-shadow: 0 0 20px rgba(0,200,255,0.4);
        }
        .p1 { width: 26px; height: 26px; background: radial-gradient(circle at 35% 35%, #7ad0ff, #2b6cb0 60%, #1a365d 100%); top: -13px; left: 50%; transform: translateX(-50%); }
        .p2 { width: 34px; height: 34px; background: radial-gradient(circle at 40% 30%, #ffd27a, #c05621 60%, #7b341e 100%); top: -17px; left: 50%; transform: translateX(-50%); }
        .p3 { width: 18px; height: 18px; background: radial-gradient(circle at 40% 30%, #d7a4ff, #8b5cf6 60%, #5538a3 100%); top: -9px; left: 50%; transform: translateX(-50%); }

        h1 {
            font-family:'Orbitron', monospace !important; font-weight:900 !important;
            background: linear-gradient(120deg, #00ffff, #ff00ff, #8b5cf6, #00ffff);
            background-size:200% auto; background-clip:text; -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            animation: shine 3s linear infinite;
            text-align:center; font-size:3.6rem !important; margin:0 !important;
            text-shadow:0 0 40px rgba(0,255,255,0.6);
        }
        @keyframes shine { to { background-position: 200% center; } }

        h2,h3 { font-family:'Space Grotesk', sans-serif !important; color:#b9c4ff !important; text-shadow:0 0 15px rgba(100,100,255,0.35); }

        .stTabs [data-baseweb="tab-list"] {
            background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);
            border-radius:15px; padding:10px; border:1px solid rgba(255,255,255,0.1);
            box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        }
        .stTabs [data-baseweb="tab"] {
            color:#8ab4ff !important; font-family:'Space Grotesk', sans-serif !important; font-weight:700;
            transition: all .3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover { transform: translateY(-2px); color:#7c3aed !important; }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, rgba(0,255,255,0.18), rgba(139,92,246,0.25));
            border-radius:10px; box-shadow: 0 4px 20px rgba(0,255,255,0.28);
        }

        .stTextInput input, .stNumberInput input, .stSelectbox select, .stSlider label {
            background: rgba(255,255,255,0.05) !important; border: 2px solid rgba(0,255,255,0.3) !important;
            color:#e6e6ff !important; border-radius:10px !important; backdrop-filter: blur(5px);
            transition: all .3s ease; box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
        }
        .stTextInput input:focus, .stNumberInput input:focus {
            border-color:#7c3aed !important; box-shadow:0 0 30px rgba(124,58,237,0.45), inset 0 2px 4px rgba(0,0,0,0.2) !important;
            transform: scale(1.02);
        }

        .stButton > button {
            background: linear-gradient(135deg, #667eea 0%, #7c3aed 100%); color:#f1f1ff; border:none;
            padding:12px 35px; border-radius:30px; font-weight:800; font-family:'Space Grotesk', sans-serif;
            transition: all .25s cubic-bezier(0.175, 0.885, 0.32, 1.275); box-shadow:0 6px 20px rgba(102,126,234,0.4);
            position: relative; overflow: hidden;
        }
        .stButton > button:hover { transform: translateY(-2px) scale(1.02); box-shadow:0 10px 35px rgba(124,58,237,0.5); }

        .stAlert {
            background: rgba(255,255,255,0.05) !important; backdrop-filter: blur(10px);
            border:2px solid rgba(255,255,255,0.2); border-radius:15px; box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }

        [data-testid="metric-container"] {
            background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);
            border:2px solid rgba(255,255,255,0.1); padding:20px; border-radius:15px;
            box-shadow:0 6px 25px rgba(0,0,0,0.3); transition: all 0.25s ease; position: relative; overflow: hidden;
        }
        [data-testid="metric-container"]:hover {
            transform: translateY(-3px) scale(1.01);
            border-color: rgba(124,58,237,0.35);
            box-shadow:0 10px 40px rgba(124,58,237,0.35);
        }

        [data-testid="stFileUploadDropzone"] {
            background: rgba(255,255,255,0.03); border:3px dashed rgba(124,58,237,0.35);
            border-radius:20px; transition: all 0.25s ease;
        }
        [data-testid="stFileUploadDropzone"]:hover { background: rgba(124,58,237,0.1); border-color: rgba(124,58,237,0.6); transform: scale(1.01); }

        .stProgress > div > div > div {
            background: linear-gradient(90deg, #00ffff, #7c3aed);
            border-radius:10px; animation: progressPulse 2s infinite; box-shadow: 0 0 20px rgba(0,255,255,0.5);
        }
        @keyframes progressPulse { 0%,100%{opacity:1; box-shadow: 0 0 20px rgba(0,255,255,0.5);} 50%{opacity:.85; box-shadow: 0 0 40px rgba(124,58,237,0.7);} }

        ::-webkit-scrollbar { width:12px; background: rgba(255,255,255,0.05); }
        ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #667eea, #7c3aed); border-radius:10px; border: 2px solid rgba(255,255,255,0.1); }

        .streamlit-expanderHeader { background: rgba(255,255,255,0.07) !important; border-radius:10px !important; color:#a8b4ff !important; }
        .streamlit-expanderHeader:hover { background: rgba(124,58,237,0.1) !important; }

        /* Force all labels that default to black/white into blue/purple */
        label, .stMarkdown p label, .stSelectbox label, .stNumberInput label, .stTextInput label, .stSlider label, .stRadio label, .stCheckbox label {
            color: #8ab4ff !important;
        }
        .stMarkdown, p, span, li {
            color:#c7d2ff !important;
        }
        #MainMenu, header, footer { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

    # Parallax star layers
    st.markdown("""
    <div class="stars"></div>
    <div class="stars2"></div>
    <div class="stars3"></div>
    """, unsafe_allow_html=True)

inject_custom_css()

# =========================
# Header with orbiting planets
# =========================
def show_header():
    st.markdown("""
        <div style='width:100%;display:flex;justify-content:center;align-items:center;margin-top:10px;margin-bottom:4px;'>
            <div class="title-wrap">
                <div class="orbit o3"><div class="planet p3"></div></div>
                <div class="orbit o2"><div class="planet p2"></div></div>
                <div class="orbit o1"><div class="planet p1"></div></div>
                <h1>🌌 Celestial Ai</h1>
            </div>
        </div>
        <p style='text-align:center;color:#9bb3ff;font-family:Space Grotesk;font-size:1.2rem;margin-top:8px;'>
            Advanced Exoplanet Detection System • NASA Space Apps 2025
        </p>
        <div style='text-align:center;margin:16px 0;'>
            <span style='background:linear-gradient(90deg,#667eea,#7c3aed);padding:8px 18px;border-radius:22px;color:#fff;font-size:.95rem;font-family:Space Grotesk;box-shadow:0 6px 20px rgba(102,126,234,0.35);display:inline-block;'>
                ✨ Powered by a Kepler 10-Feature Model
            </span>
        </div>
    """, unsafe_allow_html=True)

show_header()

# =========================
# Sidebar: Model Controls (no demo toggles)
# =========================
with st.sidebar:
    st.markdown("""
        <div style='text-align:center;padding:18px;background:rgba(255,255,255,0.05);
                    border-radius:15px;margin-bottom:16px;border:1px solid rgba(0,255,255,0.28);'>
            <h2 style='margin:0;font-size:1.45rem;'>🚀 Control Panel</h2>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("🔧 Model Configuration", expanded=True):
        model_upload = st.file_uploader("Upload Pipeline (.joblib)", type=["joblib"], key="model")
        encoder_upload = st.file_uploader("Upload Encoder (.joblib)", type=["joblib"], key="encoder")

    @st.cache_resource
    def load_model_safe(file_or_path):
        try:
            return joblib.load(file_or_path)
        except Exception:
            return None

    pipeline = None
    label_encoder = None

    if model_upload and encoder_upload:
        pipeline = load_model_safe(model_upload)
        label_encoder = load_model_safe(encoder_upload)
        if pipeline and label_encoder:
            st.success("✅ Custom artifacts loaded (uploaded).")
        else:
            st.error("❌ Failed to load uploaded artifacts.")
    else:
        pipeline = load_model_safe("gb_pipeline.joblib")
        label_encoder = load_model_safe("label_encoder.joblib")
        if pipeline and label_encoder:
            st.info("📁 Using local artifacts: gb_pipeline.joblib + label_encoder.joblib")
        else:
            st.error("❌ Could not load model artifacts. Upload valid .joblib files above.")

    st.markdown("### 📊 System Status")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Model", "✅ Ready" if (pipeline is not None and label_encoder is not None) else "❌ Missing")
    with c2:
        st.metric("Features", f"{len(SELECTED_FEATURES)} used", delta="Kepler 10-feature")

# =========================
# Utilities
# =========================
def align_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure exactly the required 10 features, numeric, ordered. Pipeline handles impute + scale."""
    X = df.copy()
    for col in SELECTED_FEATURES:
        if col not in X.columns:
            X[col] = 0 if col.startswith("koi_fpflag_") else 0.0
    for col in SELECTED_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    return X[SELECTED_FEATURES]

def decode_labels(y_pred_int: np.ndarray) -> np.ndarray:
    try:
        if hasattr(label_encoder, "inverse_transform"):
            return label_encoder.inverse_transform(y_pred_int)
        classes_ = getattr(label_encoder, "classes_", np.array(["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]))
        return np.array([classes_[i] for i in y_pred_int])
    except Exception:
        classes_ = np.array(["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"])
        return np.array([classes_[i] for i in y_pred_int])

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Batch Analysis",
    "✨ Quick Classify",
    "🧠 Pretrained Model",
    "📈 Visualizations",
    "ℹ️ About",
])

# -------------------------
# Tab 1: Batch Analysis (Upload OR analyze 10 rows from bundled dataset)
# -------------------------
with tab1:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.05);
                        border-radius:16px;margin-bottom:24px;border:1px solid rgba(0,255,255,0.22);'>
                <h3 style='margin:0;'>🚀 Batch Exoplanet Analysis</h3>
                <p style='color:#b2c5ff;margin-top:10px;'>
                    Upload a CSV with the <b>10 Kepler features</b>, or click the blue button to analyze
                    <b>10 random rows</b> from the <u>default dataset</u> bundled with the app.
                </p>
            </div>
        """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📁 Upload CSV Dataset (must include the 10 feature columns)", type=["csv"], key="batch_csv",
    )

    df = None
    used_source = None  # "uploaded" | "default"

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, comment="#")
            df.columns = df.columns.str.strip()
            used_source = "uploaded"
            st.success(f"✅ Loaded {len(df)} rows from uploaded file")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    # Blue action button: analyze 10 rows from default dataset immediately
    cbtn = st.columns([1, 1, 1])
    with cbtn[1]:
        if st.button("🔵 Analyze 10 Rows From Default Dataset", use_container_width=True, type="primary"):
            if pipeline is None or label_encoder is None:
                st.error("Model artifacts not loaded. Upload valid .joblib files in the sidebar.")
            else:
                try:
                    df_default = get_default_dataset()
                    if len(df_default) < 10:
                        st.error("Default dataset has fewer than 10 rows. Please bundle a larger CSV in /data.")
                    else:
                        df = df_default.sample(10, random_state=np.random.randint(0, 1_000_000)).reset_index(drop=True)
                        used_source = "default"
                        with st.spinner("Analyzing 10 rows from the default dataset..."):
                            X = align_features_df(df)
                            preds = pipeline.predict(X)
                            probas = pipeline.predict_proba(X)
                            labels = decode_labels(preds)
                            results = df.copy()
                            results["prediction"] = labels
                            results["confidence"] = (probas.max(axis=1) * 100).round(2)
                            try:
                                classes = list(getattr(label_encoder, "classes_", ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]))
                                confirmed_idx = classes.index("CONFIRMED")
                            except ValueError:
                                confirmed_idx = -1
                            results["confirmed_prob"] = (
                                (probas[:, confirmed_idx] * 100).round(2)
                                if confirmed_idx >= 0
                                else (probas.max(axis=1) * 100).round(2)
                            )
                            st.session_state["results"] = results
                        st.success("✅ Analysis complete! Processed 10 rows from the default dataset.")
                except Exception as e:
                    st.error(f"Could not load the bundled dataset: {e}")

    # Regular "Run Analysis" if the user uploaded a CSV
    if df is not None and used_source == "uploaded":
        with st.expander("📋 Dataset Preview (first 10 rows)", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)

        c1r, c2r, c3r = st.columns([1, 1, 1])
        with c2r:
            if st.button("🔬 Run Analysis", use_container_width=True, type="primary"):
                if pipeline is None or label_encoder is None:
                    st.error("Model artifacts not loaded. Upload valid .joblib files in the sidebar.")
                else:
                    with st.spinner("Analyzing candidates..."):
                        X = align_features_df(df)
                        preds = pipeline.predict(X)
                        probas = pipeline.predict_proba(X)
                        labels = decode_labels(preds)
                        results = df.copy()
                        results["prediction"] = labels
                        results["confidence"] = (probas.max(axis=1) * 100).round(2)
                        try:
                            classes = list(getattr(label_encoder, "classes_", ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]))
                            confirmed_idx = classes.index("CONFIRMED")
                        except ValueError:
                            confirmed_idx = -1
                        results["confirmed_prob"] = (
                            (probas[:, confirmed_idx] * 100).round(2)
                            if confirmed_idx >= 0
                            else (probas.max(axis=1) * 100).round(2)
                        )
                        st.session_state["results"] = results
                        st.success(f"✅ Analysis complete! Processed {len(results)} rows from uploaded CSV.")

    # Show results (from either path)
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
# Tab 2: Quick Classify (human labels, no defaults; pipeline standardizes/encodes)
# -------------------------
with tab2:
    st.markdown("""
        <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.05);
                    border-radius:16px;margin-bottom:24px;border:1px solid rgba(0,255,255,0.22);'>
            <h3 style='margin:0;'>✨ Quick Candidate Classification</h3>
            <p style='color:#b2c5ff;margin-top:10px;'>Enter the 10 trained features (the model scales/encodes internally).</p>
        </div>
    """, unsafe_allow_html=True)

    # Two columns: human-friendly names, no default values
    colA, colB = st.columns(2)

    with colA:
        koi_score = st.number_input("KOI Score (0–1)", min_value=0.0, max_value=1.0, step=0.01)
        koi_model_snr = st.number_input("Signal-to-Noise Ratio (SNR)", min_value=0.0, step=0.1)
        koi_impact = st.number_input("Transit Impact Parameter (0–1.5)", min_value=0.0, max_value=1.5, step=0.01)
        koi_duration = st.number_input("Transit Duration (hours)", min_value=0.0, step=0.1)
        koi_prad = st.number_input("Planet Radius (Earth radii)", min_value=0.0, step=0.1)

    with colB:
        koi_period = st.number_input("Orbital Period (days)", min_value=0.0, step=0.1)
        koi_fpflag_nt = st.selectbox("Not Transit-like (0/1)", [0, 1], index=0, help="0 = transit-like OK; 1 = not transit-like")
        koi_fpflag_co = st.selectbox("Centroid Offset Flag (0/1)", [0, 1], index=0)
        koi_fpflag_ss = st.selectbox("Significant Secondary Flag (0/1)", [0, 1], index=0)
        koi_fpflag_ec = st.selectbox("Eclipsing Binary Flag (0/1)", [0, 1], index=0)

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("🔮 Classify Candidate", use_container_width=True, type="primary"):
            if pipeline is None or label_encoder is None:
                st.error("Model not loaded. Upload valid artifacts in the sidebar.")
            else:
                with st.spinner("Analyzing candidate..."):
                    row = pd.DataFrame([{
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
                    X = align_features_df(row)
                    pred = pipeline.predict(X)[0]
                    proba = pipeline.predict_proba(X)[0]
                    label = decode_labels(np.array([pred]))[0]
                    confidence = float(proba.max() * 100)

                    if label == "CONFIRMED":
                        st.balloons()
                        color_block = ("rgba(0,255,0,0.1)", "rgba(0,255,255,0.5)")
                        title = "🌟 EXOPLANET CONFIRMED!"
                        sub = "This candidate shows strong signs of being a real exoplanet."
                    elif label == "CANDIDATE":
                        color_block = ("rgba(255,255,0,0.1)", "rgba(255,255,0,0.5)")
                        title = "🔍 CANDIDATE"
                        sub = "Further observation needed to confirm."
                    else:
                        color_block = ("rgba(255,0,0,0.1)", "rgba(255,0,0,0.5)")
                        title = "🚫 FALSE POSITIVE"
                        sub = "This signal is likely not from a real exoplanet."

                    st.markdown(f"""
                        <div style='text-align:center;padding:30px;background:linear-gradient(135deg,{color_block[0]},{color_block[0]});
                                    border-radius:20px;border:2px solid {color_block[1]};'>
                            <h2 style='margin:0;'>{title}</h2>
                            <h3>Confidence: {confidence:.1f}%</h3>
                            <p style='color:#cfe3ff;'>{sub}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    st.markdown("### 📊 Probability Distribution")
                    classes = list(getattr(label_encoder, "classes_", ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]))
                    fig = go.Figure(data=[go.Bar(
                        x=classes,
                        y=(proba * 100).round(2),
                        text=[f"{p*100:.1f}%" for p in proba],
                        textposition="auto"
                    )])
                    fig.update_layout(
                        title="Classification Probabilities",
                        xaxis_title="Class",
                        yaxis_title="Probability (%)",
                        template="plotly_dark",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Tab 3: Pretrained Model (allow retrain from CSV)
# -------------------------
with tab3:
    st.markdown("""
        <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.05);
                    border-radius:16px;margin-bottom:24px;border:1px solid rgba(0,255,255,0.22);'>
            <h3 style='margin:0;'>🧠 Pretrained Model</h3>
            <p style='color:#b2c5ff;margin-top:10px;'>
                Upload a CSV with the 10 features <b>+ 'koi_disposition'</b> to retrain and export a new pipeline & encoder.
            </p>
        </div>
    """, unsafe_allow_html=True)

    train_file = st.file_uploader("📥 Upload Training Dataset (CSV with 10 features + koi_disposition)", type=["csv"], key="training_file")
    if train_file is not None:
        try:
            train_df = pd.read_csv(train_file, comment="#")
            train_df.columns = train_df.columns.str.strip()
            st.success(f"✅ Loaded {len(train_df)} training samples")
            with st.expander("📋 Training Data Preview", expanded=False):
                st.dataframe(train_df.head(), use_container_width=True)

            colA, colB = st.columns(2)
            with colA:
                test_size = st.slider("Test size", 0.1, 0.4, 0.2, step=0.05)
            with colB:
                max_iter = st.number_input("Max Iterations", 50, 1000, 500, step=50)

            if st.button("🚀 Retrain Model", type="primary"):
                with st.spinner("Training model..."):
                    if "koi_disposition" not in train_df.columns:
                        st.error("Training CSV must include 'koi_disposition'.")
                    else:
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
                        Xtr, Xte, ytr, yte = train_test_split(
                            X, y, test_size=test_size, random_state=42, stratify=y
                        )
                        new_pipeline.fit(Xtr, ytr)
                        acc = new_pipeline.score(Xte, yte)
                        joblib.dump(new_pipeline, "gb_pipeline.joblib")
                        joblib.dump(train_le, "label_encoder.joblib")
                        st.success(f"✅ Retraining complete. Test accuracy: {acc:.2%}")
                        c1, c2 = st.columns(2)
                        with c1:
                            with open("gb_pipeline.joblib", "rb") as f:
                                st.download_button("📥 Download Pipeline", f.read(), "gb_pipeline.joblib")
                        with c2:
                            with open("label_encoder.joblib", "rb") as f:
                                st.download_button("📥 Download Encoder", f.read(), "label_encoder.joblib")
        except Exception as e:
            st.error(f"Error during training: {e}")
    else:
        st.info("Upload training data to (re)train a compatible 10-feature model.")

# -------------------------
# Tab 4: Visualizations (no selection; show all plots together)
# -------------------------
with tab4:
    st.markdown("""
        <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.05);
                    border-radius:16px;margin-bottom:24px;border:1px solid rgba(0,255,255,0.22);'>
            <h3 style='margin:0;'>📈 Data Visualizations</h3>
            <p style='color:#b2c5ff;margin-top:10px;'>All visualizations render automatically from current results; if none, they use the bundled dataset.</p>
        </div>
    """, unsafe_allow_html=True)

    # Determine the dataset to visualize: results -> bundled dataset predictions
    if "results" in st.session_state:
        viz_df = st.session_state["results"].copy()
    else:
        try:
            # Compute predictions for the whole bundled dataset (no random)
            base_df = get_default_dataset().copy()
            Xv = align_features_df(base_df)
            if pipeline is not None and label_encoder is not None:
                yhat = pipeline.predict(Xv)
                yproba = pipeline.predict_proba(Xv)
                viz_df = base_df.copy()
                viz_df["prediction"] = decode_labels(yhat)
                viz_df["confidence"] = (yproba.max(axis=1) * 100).round(2)
            else:
                viz_df = base_df.copy()
                viz_df["prediction"] = viz_df.get("koi_disposition", "CANDIDATE")
                viz_df["confidence"] = 50.0
        except Exception as e:
            st.error(f"Could not prepare data for visualizations: {e}")
            viz_df = None

    if viz_df is not None and len(viz_df) > 0:
        # Row 1: distribution + confidence histogram
        cA, cB = st.columns(2)
        with cA:
            vc = viz_df["prediction"].value_counts()
            fig = px.pie(values=vc.values, names=vc.index, title="Classification Distribution")
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(template="plotly_dark", height=420)
            st.plotly_chart(fig, use_container_width=True)
        with cB:
            fig = px.histogram(viz_df, x="confidence", nbins=24, title="Confidence Distribution")
            fig.update_layout(template="plotly_dark", height=420)
            st.plotly_chart(fig, use_container_width=True)

        # Row 2: 2D scatter (Radius vs Period)
        cC, cD = st.columns(2)
        with cC:
            if all(c in viz_df.columns for c in ["koi_period", "koi_prad"]):
                fig = px.scatter(
                    viz_df, x="koi_period", y="koi_prad", color="prediction", size="confidence",
                    title="Radius vs Period (confidence-scaled)"
                )
                fig.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Missing columns for Radius vs Period plot.")

        with cD:
            # Correlation heatmap for numeric features
            num_cols = [c for c in SELECTED_FEATURES if c in viz_df.columns]
            if len(num_cols) >= 3:
                corr = viz_df[num_cols].corr()
                fig = px.imshow(corr, title="Feature Correlation Heatmap", color_continuous_scale='Viridis')
                fig.update_layout(template="plotly_dark", height=500)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough numeric features for correlation heatmap.")

        # Row 3: 3D explorer if columns exist
        if all(c in viz_df.columns for c in ["koi_model_snr", "koi_prad", "koi_period"]):
            fig3d = px.scatter_3d(
                viz_df, x="koi_model_snr", y="koi_prad", z="koi_period",
                color="prediction", size="confidence", title="3D Explorer: SNR vs Radius vs Period"
            )
            fig3d.update_layout(template="plotly_dark", height=700)
            st.plotly_chart(fig3d, use_container_width=True)
        else:
            st.info("3D Explorer requires koi_model_snr, koi_prad, koi_period.")
    else:
        st.info("No data available to visualize yet.")

# -------------------------
# Tab 5: About
# -------------------------
with tab5:
    st.markdown("""
        <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.05);border-radius:15px;margin-bottom:24px;'>
            <h3 style='margin:0;'>ℹ️ About Celestial Ai</h3>
            <p style='color:#b2c5ff;margin-top:10px;'>NASA Space Apps Challenge 2025 — Kepler 10-feature model edition</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
            ### 🌌 Project Overview
            This build integrates your trained Kepler pipeline that uses exactly 10 features:
            `koi_score, koi_fpflag_nt, koi_model_snr, koi_fpflag_co, koi_fpflag_ss, koi_fpflag_ec, koi_impact, koi_duration, koi_prad, koi_period`.
            It supports batch analysis, single-candidate classification, optional retraining, and interactive visuals—no random data.
        """)
    with col2:
        st.markdown("""
            ### 🏆 Highlights
            - Exact feature alignment to your model  
            - Bundled default dataset for first-time users  
            - Clean UI with metrics & downloads  
            - Plotly visuals & 3D explorer
        """)

    st.markdown("---")
    st.markdown("""
        <div style='text-align:center;padding:18px;background:rgba(255,255,255,0.05);border-radius:15px;'>
            <h4 style='color:#00ffff;margin:0;'>👨‍🚀 Built for NASA Space Apps 2025</h4>
            <p style='color:#9bb3ff;'>Exploring new worlds through AI</p>
        </div>
    """, unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.markdown("""
    <div style='text-align:center;margin-top:38px;padding:18px;border-top:1px solid rgba(255,255,255,0.1);'>
        <p style='color:#8fa1ff;font-size:0.95rem;'>
            Celestial Ai • NASA Space Apps 2025 • <span style='color:#7c3aed;'>Kepler 10-Feature Pipeline</span>
        </p>
    </div>
""", unsafe_allow_html=True)


