# Celestial Ai (Kepler 10-feature edition) — Streamlit app (NO RANDOM DATA)
# Visuals & animations inspired by the ExoHunter build (Discord-style + parallax + planets)
# Integrates trained pipeline: gb_pipeline.joblib + label_encoder.joblib
# Batch: analyzes 10 random rows from uploaded CSV
# Quick Classify: human-friendly names, no default values; pipeline handles standardization/encoding

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
import streamlit.components.v1 as components

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
# Enhanced CSS with Discord-style animations + orbiting planets
# =========================
def inject_custom_css():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Grotesk:wght@300;400;700&display=swap');

            /* Main app background */
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

            /* Parallax star fields */
            .stars{
                position:fixed; top:0; left:0; width:100%; height:100%; pointer-events:none;
                background-image:
                    radial-gradient(2px 2px at 20px 30px, white, transparent),
                    radial-gradient(2px 2px at 40px 70px, white, transparent),
                    radial-gradient(1px 1px at 50px 50px, white, transparent),
                    radial-gradient(1px 1px at 80px 10px, white, transparent),
                    radial-gradient(2px 2px at 130px 80px, white, transparent);
                background-repeat: repeat; background-size:200px 200px;
                animation: stars 60s linear infinite; opacity:0.15; z-index:0;
            }
            @keyframes stars { 0% {transform: translateY(0);} 100% {transform: translateY(-120px);} }

            .stars2{
                position:fixed; top:0; left:0; width:100%; height:100%; pointer-events:none;
                background-image:
                    radial-gradient(1px 1px at 60px 120px, #bbddff, transparent),
                    radial-gradient(2px 2px at 150px 90px, #ffffff, transparent),
                    radial-gradient(1px 1px at 200px 40px, #aaccff, transparent),
                    radial-gradient(2px 2px at 300px 160px, #ffffff, transparent);
                background-repeat: repeat; background-size:300px 300px;
                animation: stars2 120s linear infinite; opacity:0.12; z-index:0;
            }
            @keyframes stars2 { 0% {transform: translateY(0);} 100% {transform: translateY(-180px);} }

            .stars3{
                position:fixed; top:0; left:0; width:100%; height:100%; pointer-events:none;
                background-image:
                    radial-gradient(1px 1px at 120px 60px, #88c0ff, transparent),
                    radial-gradient(2px 2px at 240px 180px, #ffffff, transparent),
                    radial-gradient(1px 1px at 400px 120px, #cfe8ff, transparent);
                background-repeat: repeat; background-size:500px 500px;
                animation: stars3 240s linear infinite; opacity:0.08; z-index:0;
            }
            @keyframes stars3 { 0% {transform: translateY(0);} 100% {transform: translateY(-220px);} }

            /* Nebula glow blobs */
            .nebula {
                position: fixed; width: 600px; height: 600px; pointer-events: none; z-index: 0; opacity: 0.15; filter: blur(100px);
            }
            .nebula-1 { top: -200px; left: -200px; background: radial-gradient(circle, rgba(138, 43, 226, 0.4) 0%, transparent 70%); animation: nebulaPulse 20s infinite ease-in-out; }
            .nebula-2 { bottom: -200px; right: -200px; background: radial-gradient(circle, rgba(0, 191, 255, 0.4) 0%, transparent 70%); animation: nebulaPulse 25s infinite ease-in-out; animation-delay: 5s; }
            @keyframes nebulaPulse { 0%, 100% { transform: scale(1) rotate(0deg); } 50% { transform: scale(1.2) rotate(180deg); } }

            /* Planets (ambient) */
            .planet {
                position:fixed; border-radius:50%; pointer-events:none; z-index:0; filter: drop-shadow(0 0 20px rgba(0,200,255,0.3)); opacity:0.3;
            }
            .planet-1 {
                width:160px; height:160px; bottom:5%; left:6%;
                background: radial-gradient(circle at 35% 35%, #7ad0ff, #2b6cb0 60%, #1a365d 100%);
                animation: floatPlanet1 28s ease-in-out infinite;
            }
            @keyframes floatPlanet1 { 0%,100% { transform: translateY(0) translateX(0); } 25% { transform: translateY(-10px) translateX(5px);} 50% { transform: translateY(-6px) translateX(-5px);} 75% { transform: translateY(-8px) translateX(3px);} }

            .planet-2 {
                width:110px; height:110px; top:8%; right:10%;
                background: radial-gradient(circle at 40% 30%, #ffd27a, #c05621 60%, #7b341e 100%);
                animation: floatPlanet2 32s ease-in-out infinite;
            }
            @keyframes floatPlanet2 { 0%,100% { transform: translateY(0) scale(1);} 50% { transform: translateY(10px) scale(1.05);} }

            /* ===== Orbiting planets around title ===== */
            .title-orbit {
                position: relative;
                width: 100%;
                display: flex;
                justify-content: center;
                margin-bottom: 1rem;
            }
            .orbit-wrapper {
                position: absolute;
                width: 320px; height: 320px;
                margin-top: -40px; /* raise orbit around title */
                pointer-events: none;
            }
            .orbit-ring {
                position: absolute;
                inset: 0;
                border-radius: 50%;
                border: 1px dashed rgba(0, 255, 255, 0.25);
                animation: orbitRotate 28s linear infinite;
            }
            .orbit-planet {
                position: absolute;
                width: 18px; height: 18px; border-radius: 50%;
                top: -9px; left: 50%; transform: translateX(-50%);
                box-shadow: 0 0 10px rgba(0,255,255,0.8);
            }
            .p1 { background: radial-gradient(circle, #00ffff, #0088ff); }
            .p2 { background: radial-gradient(circle, #ff99ff, #aa44cc); }
            .p3 { background: radial-gradient(circle, #a6ff00, #33cc66); }
            .ring-2 { transform: rotate(60deg); animation-duration: 34s; }
            .ring-3 { transform: rotate(120deg); animation-duration: 40s; }
            @keyframes orbitRotate {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            /* Reveal animation */
            .reveal-element { opacity: 0; transform: translateY(30px); transition: opacity 0.8s ease, transform 0.8s ease; }
            .reveal-element.revealed { opacity: 1; transform: translateY(0); }

            /* Title */
            h1 {
                font-family:'Orbitron', monospace !important; font-weight:900 !important;
                background: linear-gradient(120deg, #00ffff, #ff00ff, #00ffff);
                background-size:200% auto; -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
                animation: shine 3s linear infinite; text-align:center; font-size:3.6rem !important; margin-bottom:0.8rem !important;
                text-shadow:0 0 40px rgba(0,255,255,0.6);
            }
            @keyframes shine { to { background-position: 200% center; } }

            h2,h3, p, li, span, div, label {
                color:#b8b8ff !important; /* ensure no white/black text; make it bluish-purple */
            }

            /* Tabs */
            .stTabs [data-baseweb="tab-list"] {
                background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border-radius:15px; padding:10px; border:1px solid rgba(255,255,255,0.1);
                box-shadow: 0 8px 32px rgba(0,0,0,0.2);
            }
            .stTabs [data-baseweb="tab"] {
                color:#a0a0ff !important; font-family:'Space Grotesk', sans-serif !important; font-weight:700; transition: all .3s ease;
            }
            .stTabs [data-baseweb="tab"]:hover { transform: translateY(-2px); color: #9ed0ff !important; }
            .stTabs [aria-selected="true"] {
                background: linear-gradient(90deg, rgba(0,255,255,0.2), rgba(255,0,255,0.2));
                border-radius:10px; box-shadow: 0 4px 20px rgba(0,255,255,0.3);
            }

            /* Inputs */
            .stTextInput input, .stNumberInput input, .stSelectbox select, .stTextArea textarea {
                background: rgba(255,255,255,0.05) !important; border: 2px solid rgba(0,255,255,0.3) !important;
                color:#e0e0ff !important; border-radius:10px !important; backdrop-filter: blur(5px);
                transition: all .3s ease; box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
            }
            .stTextInput input:focus, .stNumberInput input:focus, .stTextArea textarea:focus {
                border-color:#00ffff !important; box-shadow:0 0 30px rgba(0,255,255,0.4), inset 0 2px 4px rgba(0,0,0,0.2) !important; transform: scale(1.01);
            }

            /* Buttons */
            .stButton > button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color:white; border:none; padding:12px 35px; border-radius:30px; font-weight:800; font-family:'Space Grotesk', sans-serif;
                transition: all .25s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                box-shadow:0 6px 20px rgba(102,126,234,0.4);
                position: relative; overflow: hidden;
            }
            .stButton > button:hover { transform: translateY(-3px) scale(1.05); box-shadow:0 10px 35px rgba(102,126,234,0.6); }

            /* Alerts */
            .stAlert {
                background: rgba(255,255,255,0.05) !important; backdrop-filter: blur(10px);
                border:2px solid rgba(255,255,255,0.2); border-radius:15px; animation: slideInBounce .8s cubic-bezier(0.68, -0.55, 0.265, 1.55);
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            }
            @keyframes slideInBounce {
                0% { opacity:0; transform: translateY(-30px) scale(0.9); }
                60% { transform: translateY(5px) scale(1.02); }
                100% { opacity:1; transform: translateY(0) scale(1); }
            }

            /* Metrics */
            [data-testid="metric-container"] {
                background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);
                border:2px solid rgba(255,255,255,0.1); padding:20px; border-radius:15px;
                box-shadow:0 6px 25px rgba(0,0,0,0.3); transition: all 0.3s ease; position: relative; overflow: hidden;
            }
            [data-testid="metric-container"]:hover { transform: translateY(-5px) scale(1.02); border-color: rgba(0,255,255,0.3); box-shadow:0 10px 40px rgba(0,255,255,0.3); }

            /* File Upload */
            [data-testid="stFileUploadDropzone"] {
                background: rgba(255,255,255,0.03); border:3px dashed rgba(0,255,255,0.3);
                border-radius:20px; transition: all 0.4s ease; position: relative;
            }
            [data-testid="stFileUploadDropzone"]:hover {
                background: rgba(0,255,255,0.08); border-color: rgba(0,255,255,0.6);
                transform: scale(1.02); box-shadow: 0 0 30px rgba(0,255,255,0.3);
            }

            /* Progress bar */
            .stProgress > div > div > div {
                background: linear-gradient(90deg, #00ffff, #ff00ff); border-radius:10px;
                animation: progressPulse 2s infinite; box-shadow: 0 0 20px rgba(0,255,255,0.5);
            }
            @keyframes progressPulse {
                0%,100%{ opacity:1; box-shadow: 0 0 20px rgba(0,255,255,0.5); }
                50%{ opacity:.8; box-shadow: 0 0 40px rgba(255,0,255,0.7); }
            }

            /* Scrollbar */
            ::-webkit-scrollbar { width:12px; background: rgba(255,255,255,0.05); }
            ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #667eea, #764ba2); border-radius:10px; border: 2px solid rgba(255,255,255,0.1); }

            /* Expanders */
            .streamlit-expanderHeader { background: rgba(255,255,255,0.05) !important; border-radius:10px !important; transition: all 0.3s ease; }
            .streamlit-expanderHeader:hover { background: rgba(255,255,255,0.08) !important; transform: translateX(5px); }

            /* Hide default Streamlit elements */
            #MainMenu, header, footer { visibility: hidden; }

            /* Force all form labels that default to black/white to bright blue or purple */
            label, .stMarkdown p label, .stSelectbox label, .stNumberInput label, .stTextInput label, .stSlider label {
                color: #8bb6ff !important;
            }
        </style>

        <!-- Parallax layers -->
        <div class="stars"></div>
        <div class="stars2"></div>
        <div class="stars3"></div>
        <div class="nebula nebula-1"></div>
        <div class="nebula nebula-2"></div>
        <div class="planet planet-1"></div>
        <div class="planet planet-2"></div>
        """,
        unsafe_allow_html=True,
    )

    # Scroll + mouse animation helpers
    components.html(
        """
        <script>
        // Reveal on intersection
        const observerOptions = { threshold: 0.1, rootMargin: '0px 0px -50px 0px' };
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => { if (entry.isIntersecting) { entry.target.classList.add('revealed'); }});
        }, observerOptions);

        document.addEventListener('DOMContentLoaded', () => {
            const elements = document.querySelectorAll('.reveal-element');
            elements.forEach(el => observer.observe(el));
        });

        // Gentle parallax for planets/nebula
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const nebulas = document.querySelectorAll('.nebula');
            nebulas.forEach((nebula, index) => {
                const speed = (index + 1) * 0.02;
                nebula.style.transform = `translateY(${scrolled * speed}px)`;
            });
        });
        </script>
        """,
        height=0,
    )

inject_custom_css()

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

# =========================
# Load trained artifacts ONLY (no demo fallback)
# =========================
@st.cache_resource
def load_model_safe(file_or_path):
    try:
        return joblib.load(file_or_path)
    except Exception:
        return None

# =========================
# Header with orbiting planets
# =========================
def show_header():
    st.markdown(
        """
        <div class="reveal-element title-orbit">
            <div class="orbit-wrapper">
                <div class="orbit-ring ring-1">
                    <div class="orbit-planet p1"></div>
                </div>
                <div class="orbit-ring ring-2">
                    <div class="orbit-planet p2"></div>
                </div>
                <div class="orbit-ring ring-3">
                    <div class="orbit-planet p3"></div>
                </div>
            </div>
            <h1>🌌 Celestial Ai</h1>
        </div>
        <p style='text-align:center;color:#a0a0ff;font-family:Space Grotesk;font-size:1.2rem;margin-top:0;'>
            Advanced Exoplanet Detection System • NASA Space Apps 2025
        </p>
        <div style='text-align:center;margin:20px 0;'>
            <span style='background:linear-gradient(90deg,#667eea,#764ba2);padding:8px 20px;border-radius:25px;color:#fff;font-size:1rem;font-family:Space Grotesk;box-shadow:0 6px 20px rgba(102,126,234,0.4);display:inline-block;'>
                ✨ Powered by Your Kepler 10-Feature Model
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

show_header()

# =========================
# Sidebar: Model Controls (no demo toggles)
# =========================
with st.sidebar:
    st.markdown(
        """
        <div class="reveal-element" style='text-align:center;padding:20px;background:rgba(255,255,255,0.05);
                    border-radius:15px;margin-bottom:20px;border:2px solid rgba(0,255,255,0.3);box-shadow:0 8px 30px rgba(0,255,255,0.2);'>
            <h2 style='margin:0;font-size:1.6rem;'>🚀 Control Panel</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔧 Model Configuration", expanded=True):
        model_upload = st.file_uploader("Upload Pipeline (.joblib)", type=["joblib"], key="model")
        encoder_upload = st.file_uploader("Upload Encoder (.joblib)", type=["joblib"], key="encoder")

    # Load priority: uploaded -> local
    pipeline = None
    label_encoder = None

    if model_upload and encoder_upload:
        pipeline = load_model_safe(model_upload)
        label_encoder = load_model_safe(encoder_upload)
        if pipeline and label_encoder:
            st.success("✅ Custom artifacts loaded (uploaded).")
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
    """Ensure the dataframe has exactly the required 10 features in order.
    NOTE: The loaded sklearn Pipeline handles imputation + scaling (standardization)."""
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
# Tabs (renamed & colored)
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Batch Analysis (10-row sample)",
    "✨ Quick Classify (single)",
    "🧠 Pretrained Model",
    "📈 Visualizations",
    "ℹ️ About"
])

# -------------------------
# Tab 1: Batch Analysis — analyze 10 random rows from uploaded CSV
# -------------------------
with tab1:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(
            """
            <div class="reveal-element" style='text-align:center;padding:25px;background:rgba(255,255,255,0.03);
                        border-radius:20px;margin-bottom:35px;border:2px solid rgba(0,255,255,0.2);box-shadow:0 10px 40px rgba(0,0,0,0.3);'>
                <h3 style='margin:0;font-size:1.8rem;'>🚀 Batch Exoplanet Analysis</h3>
                <p style='margin-top:15px;font-size:1.05rem;'>
                    Upload a CSV containing the <b>10 Kepler features</b>. We’ll randomly sample <b>10 rows</b> for analysis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    uploaded_file = st.file_uploader(
        "📁 Upload CSV Dataset (must include the 10 feature columns)", type=["csv"]
    )

    df = None
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, comment="#")
            df.columns = df.columns.str.strip()
            st.success(f"✅ Loaded {len(df)} rows from file")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    if df is not None and pipeline is not None and label_encoder is not None:
        with st.expander("📋 Dataset Preview (first 12 rows)", expanded=False):
            st.dataframe(df.head(12), use_container_width=True)

        c1, c2, c3 = st.columns([1, 1, 1])
        with c2:
            if st.button("🔬 Analyze 10 Random Rows", use_container_width=True, type="primary"):
                with st.spinner("Sampling & analyzing 10 rows..."):
                    progress_bar = st.progress(0)

                    # Sample exactly 10 rows (or fewer if file has <10)
                    sample_n = min(10, len(df))
                    sample_df = df.sample(sample_n, random_state=None).reset_index(drop=True)

                    progress_bar.progress(25)
                    X = align_features_df(sample_df)
                    progress_bar.progress(50)
                    preds = pipeline.predict(X)
                    probas = pipeline.predict_proba(X)
                    progress_bar.progress(75)
                    labels = decode_labels(preds)
                    results = sample_df.copy()
                    results["prediction"] = labels
                    results["confidence"] = (probas.max(axis=1) * 100).round(2)

                    try:
                        classes = list(getattr(label_encoder, "classes_", ["FALSE POSITIVE", "CANDIDATE", "CONFIRMED"]))
                        confirmed_idx = classes.index("CONFIRMED")
                    except ValueError:
                        confirmed_idx = -1
                    results["confirmed_prob"] = (
                        (probas[:, confirmed_idx] * 100).round(2)
                        if confirmed_idx >= 0 else (probas.max(axis=1) * 100).round(2)
                    )

                    progress_bar.progress(100)
                    st.session_state["results"] = results
                    st.success(f"✅ Analysis complete! Processed {len(results)} sampled rows")

    # Show results
    if "results" in st.session_state:
        results = st.session_state["results"]

        st.markdown("""<div class="reveal-element"><h3 style='margin-top:30px;'>📊 Analysis Summary</h3></div>""", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            confirmed = (results["prediction"] == "CONFIRMED").sum()
            st.metric("🌟 Confirmed", int(confirmed), delta=f"{confirmed/len(results)*100:.1f}%")
        with c2:
            cand = (results["prediction"] == "CANDIDATE").sum()
            st.metric("🔍 Candidates", int(cand), delta=f"{cand/len(results)*100:.1f}%")
        with c3:
            fp = (results["prediction"] == "FALSE POSITIVE").sum()
            st.metric("❌ False Positives", int(fp), delta=f"{fp/len(results)*100:.1f}%")
        with c4:
            avg_conf = float(results["confidence"].mean())
            st.metric("💪 Avg Confidence", f"{avg_conf:.1f}%", delta="High" if avg_conf > 70 else "Moderate")

        st.markdown("""<div class="reveal-element"><h3>📋 Detailed Predictions</h3></div>""", unsafe_allow_html=True)
        st.dataframe(results, use_container_width=True)

        csv = results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name=f"exoplanet_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# -------------------------
# Tab 2: Quick Classify — human-friendly names, no default values
# -------------------------
with tab2:
    st.markdown(
        """
        <div class="reveal-element" style='text-align:center;padding:25px;background:rgba(255,255,255,0.03);
                    border-radius:20px;margin-bottom:35px;border:2px solid rgba(0,255,255,0.2);box-shadow:0 10px 40px rgba(0,0,0,0.3);'>
            <h3 style='margin:0;font-size:1.8rem;'>✨ Quick Candidate Classification</h3>
            <p style='margin-top:15px;font-size:1.05rem;'>
                Enter the <b>10 Kepler features</b> (no pre-filled defaults). Your pretrained pipeline will standardize/encode internally.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # We use text inputs (instead of number_input) to avoid forced defaults.
    # We'll validate and parse floats/ints after submission.
    st.markdown("#### 📡 Signal & Score")
    q1, q2, q3 = st.columns(3)
    with q1:
        v_koi_score = st.text_input("KOI Score (0-1)", placeholder="e.g., 0.58")
    with q2:
        v_koi_model_snr = st.text_input("Model Signal-to-Noise Ratio", placeholder="e.g., 22.4")
    with q3:
        v_koi_impact = st.text_input("Impact Parameter (0-1.5)", placeholder="e.g., 0.47")

    st.markdown("#### 🪐 Geometry & Timing")
    r1, r2, r3 = st.columns(3)
    with r1:
        v_koi_duration = st.text_input("Transit Duration (hours)", placeholder="e.g., 9.8")
    with r2:
        v_koi_prad = st.text_input("Planet Radius (Earth radii)", placeholder="e.g., 1.6")
    with r3:
        v_koi_period = st.text_input("Orbital Period (days)", placeholder="e.g., 37.2")

    st.markdown("#### 🚩 False-Positive Flags")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        v_koi_fpflag_nt = st.selectbox("Not Transit-like (0/1)", ['-- select --', 0, 1], index=0,
                                       help="1 if the signal is not transit-like; otherwise 0")
    with f2:
        v_koi_fpflag_co = st.selectbox("Centroid Offset (0/1)", ['-- select --', 0, 1], index=0,
                                       help="1 if a centroid offset is detected; otherwise 0")
    with f3:
        v_koi_fpflag_ss = st.selectbox("Significant Secondary (0/1)", ['-- select --', 0, 1], index=0,
                                       help="1 if a significant secondary event exists; otherwise 0")
    with f4:
        v_koi_fpflag_ec = st.selectbox("Eclipsing Binary (0/1)", ['-- select --', 0, 1], index=0,
                                       help="1 if ephemeris/EB evidence; otherwise 0")

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("🔮 Classify Candidate", use_container_width=True, type="primary"):
            if pipeline is None or label_encoder is None:
                st.error("Model not loaded. Upload valid artifacts in the sidebar.")
            else:
                # Validate inputs
                errs = []
                def to_float(x, name):
                    if x is None or str(x).strip() == "":
                        errs.append(f"Please enter a value for '{name}'.")
                        return None
                    try:
                        return float(str(x).strip())
                    except ValueError:
                        errs.append(f"'{name}' must be a number.")
                        return None

                koi_score = to_float(v_koi_score, "KOI Score")
                koi_model_snr = to_float(v_koi_model_snr, "Model SNR")
                koi_impact = to_float(v_koi_impact, "Impact Parameter")
                koi_duration = to_float(v_koi_duration, "Transit Duration")
                koi_prad = to_float(v_koi_prad, "Planet Radius")
                koi_period = to_float(v_koi_period, "Orbital Period")

                def from_flag(sel, name):
                    if sel == '-- select --':
                        errs.append(f"Please choose 0 or 1 for '{name}'.")
                        return None
                    return int(sel)

                koi_fpflag_nt = from_flag(v_koi_fpflag_nt, "Not Transit-like")
                koi_fpflag_co = from_flag(v_koi_fpflag_co, "Centroid Offset")
                koi_fpflag_ss = from_flag(v_koi_fpflag_ss, "Significant Secondary")
                koi_fpflag_ec = from_flag(v_koi_fpflag_ec, "Eclipsing Binary")

                if errs:
                    for e in errs:
                        st.error(e)
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

                        time.sleep(0.2)

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

                        st.markdown(
                            f"""
                            <div style='text-align:center;padding:30px;background:linear-gradient(135deg,{color_block[0]},{color_block[0]});
                                        border-radius:20px;border:2px solid {color_block[1]};'>
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
                            data=[
                                go.Bar(
                                    x=classes,
                                    y=(proba * 100).round(2),
                                    text=[f"{p*100:.1f}%" for p in proba],
                                    textposition="auto",
                                )
                            ]
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
# Tab 3: Pretrained Model (info only)
# -------------------------
with tab3:
    st.markdown(
        """
        <div class="reveal-element" style='text-align:center;padding:25px;background:rgba(255,255,255,0.03);
                    border-radius:20px;margin-bottom:35px;border:2px solid rgba(0,255,255,0.2);box-shadow:0 10px 40px rgba(0,0,0,0.3);'>
            <h3 style='margin:0;font-size:1.8rem;'>🧠 Pretrained Model</h3>
            <p style='margin-top:15px;font-size:1.05rem;'>
                The model is already trained and loaded from <b>gb_pipeline.joblib</b> and <b>label_encoder.joblib</b>.
                Use the tabs to classify or analyze data.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🏗️ Model Architecture")
        st.markdown(
            """
            **Algorithm:** Histogram-based Gradient Boosting Classifier  
            **Features:** 10 Kepler parameters  
            **Classes:** FALSE POSITIVE, CANDIDATE, CONFIRMED  

            **Advantages:**  
            - Handles tabular features well  
            - Robust to scaling (we still standardize in-pipeline)  
            - Supports missing data via imputation
            """
        )

    with col2:
        st.markdown("### 🔩 Pipeline Stages")
        st.markdown(
            """
            - **Imputation:** `SimpleImputer(strategy="constant", fill_value=0)`  
            - **Scaling:** `StandardScaler()`  
            - **Classifier:** `HistGradientBoostingClassifier(...)`  

            Your **gb_pipeline.joblib** encapsulates all steps, so inputs here can be raw numeric values.
            """
        )

# -------------------------
# Tab 4: Visualizations (based on results in session)
# -------------------------
with tab4:
    st.markdown(
        """
        <div class="reveal-element" style='text-align:center;padding:25px;background:rgba(255,255,255,0.03);
                    border-radius:20px;margin-bottom:35px;border:2px solid rgba(0,255,255,0.2);box-shadow:0 10px 40px rgba(0,0,0,0.3);'>
            <h3 style='margin:0;font-size:1.8rem;'>📈 Data Visualizations & Analytics</h3>
            <p style='margin-top:15px;font-size:1.05rem;'>Visualize the predictions you just generated.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "results" in st.session_state:
        results = st.session_state["results"]
        viz_type = st.selectbox(
            "Select Visualization",
            ["Distribution Overview", "Feature Correlations", "Habitable Zone-ish", "3D Explorer"],
        )

        if viz_type == "Distribution Overview":
            c1, c2 = st.columns(2)
            with c1:
                vc = results["prediction"].value_counts()
                fig = px.pie(values=vc.values, names=vc.index, title="Classification Distribution")
                fig.update_traces(textposition="inside", textinfo="percent+label")
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.histogram(results, x="confidence", nbins=20, title="Confidence Distribution")
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Feature Correlations":
            numeric_cols = [c for c in SELECTED_FEATURES if c in results.columns]
            if len(numeric_cols) >= 3:
                feats = numeric_cols[:4]
                fig = px.scatter_matrix(results, dimensions=feats, color="prediction", title="Feature Correlation Matrix")
                fig.update_layout(template="plotly_dark", height=800)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough numeric features available.")

        elif viz_type == "Habitable Zone-ish":
            if all(col in results.columns for col in ["koi_prad", "koi_period"]):
                fig = px.scatter(
                    results, x="koi_period", y="koi_prad", color="prediction", size="confidence",
                    title="Radius vs Period (confidence-scaled)"
                )
                fig.update_layout(template="plotly_dark", height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Required features missing: koi_period & koi_prad")

        elif viz_type == "3D Explorer":
            have = [c for c in ["koi_model_snr", "koi_prad", "koi_period"] if c in results.columns]
            if len(have) == 3:
                fig = px.scatter_3d(
                    results, x=have[0], y=have[1], z=have[2],
                    color="prediction", size="confidence",
                    title="3D Explorer: SNR vs Radius vs Period"
                )
                fig.update_layout(template="plotly_dark", height=700)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need koi_model_snr, koi_prad, koi_period for 3D view.")
    else:
        st.info("📊 Run a batch analysis (Tab 1) to populate visualizations.")

# -------------------------
# Tab 5: About (Celestial Ai branding)
# -------------------------
with tab5:
    st.markdown(
        """
        <div class="reveal-element" style='text-align:center;padding:20px;background:rgba(255,255,255,0.03);border-radius:15px;margin-bottom:30px;'>
            <h3 style='margin:0;'>ℹ️ About Celestial Ai</h3>
            <p style='margin-top:10px;'>NASA Space Apps Challenge 2025 — Kepler 10-feature model edition</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(
            """
            ### 🌌 Project Overview
            This build integrates your trained Kepler pipeline that uses exactly these 10 features:
            `koi_score, koi_fpflag_nt, koi_model_snr, koi_fpflag_co, koi_fpflag_ss, koi_fpflag_ec, koi_impact, koi_duration, koi_prad, koi_period`.
            It supports batch analysis (10-row sampling), single-candidate classification, and interactive visuals — with no random/demo data.
            """
        )
    with col2:
        st.markdown(
            """
            ### 🏆 Highlights
            - Exact feature alignment to your pretrained model  
            - No random/demo data  
            - Clean UI with metrics & downloads  
            - Plotly visuals and 3D explorer  
            - Cinematic, Discord-style animations
            """
        )

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.03);border-radius:15px;'>
            <h4 style='color:#00ffff;margin:0;'>👨‍🚀 Built for NASA Space Apps 2025</h4>
            <p>Exploring new worlds through AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================
# Footer
# =========================
st.markdown(
    """
    <div style='text-align:center;margin-top:50px;padding:20px;border-top:1px solid rgba(255,255,255,0.1);'>
        <p style='font-size:0.95rem;'>
            Celestial Ai • NASA Space Apps 2025 • <span style='color:#00ffff;'>Kepler 10-Feature Pipeline</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

