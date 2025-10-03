# Celestial Ai (Kepler 10-feature edition) — Streamlit app
# Merges advanced visuals/animations with strict no-random-data logic
# Uses trained artifacts: gb_pipeline.joblib + label_encoder.joblib

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
# Enhanced CSS (Discord-style + brighter blues/purples everywhere)
# =========================

def inject_custom_css():
    st.markdown(
        r"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Grotesk:wght@300;400;700&display=swap');

        /* Global base: push all default text from white/black to blue/purple */
        html, body, .stApp { color: #a0a0ff !important; }

        .stApp {
            background: linear-gradient(-45deg, #0a0118, #1a0033, #0f0c29, #24243e);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            overflow-x: hidden;
        }
        @keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

        /* Stars/parallax layers */
        .stars,.stars2,.stars3{ position:fixed; top:0; left:0; width:100%; height:100%; pointer-events:none; background-repeat:repeat; z-index:0; }
        .stars{ background-image: radial-gradient(2px 2px at 20px 30px, white, transparent), radial-gradient(2px 2px at 40px 70px, white, transparent), radial-gradient(1px 1px at 50px 50px, white, transparent), radial-gradient(1px 1px at 80px 10px, white, transparent), radial-gradient(2px 2px at 130px 80px, white, transparent); background-size:200px 200px; animation: stars 60s linear infinite; opacity:0.15; }
        @keyframes stars { 0% {transform: translateY(0);} 100% {transform: translateY(-120px);} }
        .stars2{ background-image: radial-gradient(1px 1px at 60px 120px, #bbddff, transparent), radial-gradient(2px 2px at 150px 90px, #ffffff, transparent), radial-gradient(1px 1px at 200px 40px, #aaccff, transparent), radial-gradient(2px 2px at 300px 160px, #ffffff, transparent); background-size:300px 300px; animation: stars2 120s linear infinite; opacity:0.12; }
        @keyframes stars2 { 0% {transform: translateY(0);} 100% {transform: translateY(-180px);} }
        .stars3{ background-image: radial-gradient(1px 1px at 120px 60px, #88c0ff, transparent), radial-gradient(2px 2px at 240px 180px, #ffffff, transparent), radial-gradient(1px 1px at 400px 120px, #cfe8ff, transparent); background-size:500px 500px; animation: stars3 240s linear infinite; opacity:0.08; }
        @keyframes stars3 { 0% {transform: translateY(0);} 100% {transform: translateY(-220px);} }

        /* General containers */
        .main .block-container { position:relative; z-index:10; padding-top:2rem; }

        /* Title styling */
        h1 { font-family:'Orbitron', monospace !important; font-weight:900 !important; background: linear-gradient(120deg, #00ffff, #ff00ff, #00ffff); background-size:200% auto; background-clip:text; -webkit-background-clip:text; -webkit-text-fill-color:transparent; animation: shine 3s linear infinite; text-align:center; font-size:3.4rem !important; margin-bottom:1.2rem !important; text-shadow:0 0 40px rgba(0,255,255,0.6); }
        @keyframes shine { to { background-position: 200% center; } }

        h2,h3 { font-family:'Space Grotesk', sans-serif !important; color:#c6c6ff !important; text-shadow:0 0 15px rgba(100,100,255,0.4); }

        /* Orbiting planets around the title */
        .title-orbit-wrapper { position: relative; width: 100%; height: 220px; }
        .title-center { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); }
        .orbit-circle { position:absolute; left:50%; top:50%; transform:translate(-50%,-50%); border:1px dashed rgba(100,100,255,0.15); border-radius:50%; }
        .orbit-planet { position:absolute; width:26px; height:26px; border-radius:50%; filter: drop-shadow(0 0 8px rgba(0,200,255,0.5)); }
        .planet-a { background: radial-gradient(circle at 40% 30%, #7ad0ff, #2b6cb0 60%, #1a365d 100%); animation: orbitA 10s linear infinite; }
        .planet-b { background: radial-gradient(circle at 40% 30%, #ffd27a, #c05621 60%, #7b341e 100%); animation: orbitB 14s linear infinite; }
        .planet-c { background: radial-gradient(circle at 40% 30%, #e3a6ff, #8a2be2 60%, #4b0082 100%); animation: orbitC 18s linear infinite; }
        @keyframes orbitA { from { transform: rotate(0deg) translateX(120px) rotate(0deg);} to { transform: rotate(360deg) translateX(120px) rotate(-360deg);} }
        @keyframes orbitB { from { transform: rotate(0deg) translateX(90px) rotate(0deg);} to { transform: rotate(360deg) translateX(90px) rotate(-360deg);} }
        @keyframes orbitC { from { transform: rotate(0deg) translateX(60px) rotate(0deg);} to { transform: rotate(360deg) translateX(60px) rotate(-360deg);} }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border-radius:15px; padding:10px; border:1px solid rgba(255,255,255,0.1); box-shadow: 0 8px 32px rgba(0,0,0,0.2); }
        .stTabs [data-baseweb="tab"] { color:#a0a0ff !important; font-family:'Space Grotesk', sans-serif !important; font-weight:700; transition: all .3s ease; }
        .stTabs [data-baseweb="tab"]:hover { transform: translateY(-2px); color:#7c8ff0 !important; }
        .stTabs [aria-selected="true"] { background: linear-gradient(90deg, rgba(0,255,255,0.2), rgba(255,0,255,0.2)); border-radius:10px; box-shadow: 0 4px 20px rgba(0,255,255,0.3); }

        /* Inputs: enforce blue/purple labels and text */
        label, .stMarkdown p label, .stSelectbox label, .stNumberInput label, .stTextInput label, .stSlider label { color: #7aa7ff !important; }
        .stTextInput input, .stNumberInput input, .stSelectbox select, .stSlider { color:#d0d0ff !important; background: rgba(255,255,255,0.05) !important; border: 2px solid rgba(0,255,255,0.3) !important; border-radius:10px !important; backdrop-filter: blur(5px); box-shadow: inset 0 2px 4px rgba(0,0,0,0.2); }
        .stTextInput input::placeholder { color:#8aa4ff !important; }

        /* Buttons */
        .stButton > button { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color:#eef; border:none; padding:12px 35px; border-radius:30px; font-weight:800; font-family:'Space Grotesk', sans-serif; transition: all .3s cubic-bezier(0.175, 0.885, 0.32, 1.275); box-shadow:0 6px 20px rgba(102,126,234,0.4); position: relative; overflow: hidden; }
        .stButton > button:hover { transform: translateY(-3px) scale(1.05); box-shadow:0 10px 35px rgba(102,126,234,0.6); }

        /* Metrics, upload, progress */
        [data-testid="metric-container"] { background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border:2px solid rgba(255,255,255,0.1); padding:20px; border-radius:15px; box-shadow:0 6px 25px rgba(0,0,0,0.3); }
        [data-testid="stFileUploadDropzone"] { background: rgba(255,255,255,0.03); border:3px dashed rgba(0,255,255,0.3); border-radius:20px; }
        [data-testid="stFileUploadDropzone"]:hover { background: rgba(0,255,255,0.08); border-color: rgba(0,255,255,0.6); box-shadow: 0 0 30px rgba(0,255,255,0.3); }
        .stProgress > div > div > div { background: linear-gradient(90deg, #00ffff, #ff00ff); border-radius:10px; }

        /* Scrollbar */
        ::-webkit-scrollbar { width:12px; background: rgba(255,255,255,0.05); }
        ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #667eea, #764ba2); border-radius:10px; border: 2px solid rgba(255,255,255,0.1); }

        /* Expander */
        .streamlit-expanderHeader { background: rgba(255,255,255,0.05) !important; border-radius:10px !important; }
    </style>
    <div class="stars"></div>
    <div class="stars2"></div>
    <div class="stars3"></div>
    """,
        unsafe_allow_html=True,
    )

    # Reveal/parallax helpers
    components.html(
        """
    <script>
    const observer = new IntersectionObserver((entries)=>{entries.forEach(e=>{if(e.isIntersecting){e.target.classList.add('revealed');}});},{threshold:0.1});
    document.addEventListener('DOMContentLoaded',()=>{document.querySelectorAll('.reveal-element').forEach(el=>observer.observe(el));});
    window.addEventListener('scroll',()=>{const s=window.pageYOffset;['.stars','.stars2','.stars3'].forEach((cls,i)=>{const el=document.querySelector(cls); if(el){el.style.transform=`translateY(${s*(0.5-0.2*i)}px)`;}});});
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
        <div class="title-orbit-wrapper">
            <div class="orbit-circle" style="width:260px;height:260px;"></div>
            <div class="orbit-circle" style="width:200px;height:200px;"></div>
            <div class="orbit-circle" style="width:140px;height:140px;"></div>
            <div class="title-center">
                <h1>🌌 Celestial Ai</h1>
                <p style='text-align:center;color:#a0a0ff;font-family:Space Grotesk;font-size:1.2rem;margin-top:0;'>Advanced Exoplanet Detection System • NASA Space Apps 2025</p>
                <div style='text-align:center;margin:14px 0;'>
                    <span style='background:linear-gradient(90deg,#667eea,#764ba2);padding:6px 16px;border-radius:20px;color:#eef;font-size:.9rem;font-family:Space Grotesk;'>Powered by Your Kepler 10-Feature Model</span>
                </div>
            </div>
            <div class="orbit-planet planet-a" style="left:50%; top:50%;"></div>
            <div class="orbit-planet planet-b" style="left:50%; top:50%;"></div>
            <div class="orbit-planet planet-c" style="left:50%; top:50%;"></div>
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
        <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.05);
                    border-radius:15px;margin-bottom:20px;border:1px solid rgba(0,255,255,0.3);'>
            <h2 style='margin:0;font-size:1.5rem;'>🚀 Control Panel</h2>
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
    """Ensure the dataframe has exactly the required 10 features, numeric, in order.
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
# Tabs (renamed): Batch, Quick Classify, Pretrained Model, Visualizations, About
# =========================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Batch Analysis",
    "✨ Quick Classify",
    "🧠 Pretrained Model",
    "📈 Visualizations",
    "ℹ️ About",
])

# -------------------------
# Tab 1: Batch Analysis (user can sample 10 random rows from the uploaded CSV)
# -------------------------
with tab1:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown(
            """
            <div class='reveal-element' style='text-align:center;padding:20px;background:rgba(255,255,255,0.03);
                        border-radius:15px;margin-bottom:30px;border:1px solid rgba(0,255,255,0.2);'>
                <h3 style='margin:0;'>🚀 Batch Exoplanet Analysis</h3>
                <p style='color:#a0a0ff;margin-top:10px;'>Upload CSV with the <b>10 Kepler features</b>. Then analyze a random sample of <b>10 rows</b>.</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

    uploaded_file = st.file_uploader("📁 Upload CSV Dataset (must include the 10 feature columns)", type=["csv"])

    df = None
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, comment="#")
            df.columns = df.columns.str.strip()
            st.success(f"✅ Loaded {len(df)} rows from file")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    if df is not None and pipeline is not None and label_encoder is not None:
        with st.expander("📋 Dataset Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)

        c1, c2, c3 = st.columns([1,1,1])
        with c2:
            sample_size = 10
            if st.button("🎯 Analyze 10 Random Rows", use_container_width=True, type="primary"):
                with st.spinner("Sampling and analyzing 10 rows..."):
                    if len(df) < sample_size:
                        st.error("The CSV has fewer than 10 rows. Please upload a larger file.")
                    else:
                        sampled = df.sample(n=sample_size, random_state=None)
                        X = align_features_df(sampled)
                        preds = pipeline.predict(X)
                        probas = pipeline.predict_proba(X)
                        labels = decode_labels(preds)
                        results = sampled.copy()
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
                        st.session_state["results"] = results
                        st.success("✅ Analysis complete for 10 sampled rows")

    # Show results if available
    if "results" in st.session_state:
        results = st.session_state["results"]
        st.markdown("### 📊 Analysis Summary")
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

        st.markdown("### 🔍 Detailed Results (10 sampled rows)")
        st.dataframe(results, use_container_width=True)

        csv = results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name=f"celestial_ai_sampled10_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# -------------------------
# Tab 2: Quick Classify (human-friendly labels, no defaults)
# -------------------------
with tab2:
    st.markdown(
        """
        <div class='reveal-element' style='text-align:center;padding:20px;background:rgba(255,255,255,0.03); border-radius:15px;margin-bottom:30px;border:1px solid rgba(0,255,255,0.2);'>
            <h3 style='margin:0;'>✨ Quick Candidate Classification</h3>
            <p style='color:#a0a0ff;margin-top:10px;'>Enter the 10 trained features (no defaults). Labels are human-friendly; tooltips mention original Kepler fields.</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Helpers to parse numeric inputs from text fields (since we avoid defaults)
    def num_input(label, help_text, key):
        s = st.text_input(label, value="", placeholder="Enter a number", help=help_text, key=key)
        if s.strip() == "":
            return None
        try:
            return float(s)
        except ValueError:
            st.warning(f"'{label}' must be a number.")
            return None

    col1, col2 = st.columns(2)
    with col1:
        koi_score = num_input("Overall KOI Score (0-1)", "koi_score", key="koi_score")
        koi_model_snr = num_input("Transit Signal-to-Noise Ratio", "koi_model_snr", key="koi_model_snr")
        koi_impact = num_input("Impact Parameter (0-1)", "koi_impact", key="koi_impact")
        koi_duration = num_input("Transit Duration (hours)", "koi_duration", key="koi_duration")
        koi_prad = num_input("Planet Radius (Earth radii)", "koi_prad", key="koi_prad")

    with col2:
        koi_period = num_input("Orbital Period (days)", "koi_period", key="koi_period")
        koi_fpflag_nt = st.selectbox("Not Transit-like? (0/1)", ["— Select —", 0, 1], index=0, help="koi_fpflag_nt")
        koi_fpflag_co = st.selectbox("Centroid Offset Flag (0/1)", ["— Select —", 0, 1], index=0, help="koi_fpflag_co")
        koi_fpflag_ss = st.selectbox("Significant Secondary Flag (0/1)", ["— Select —", 0, 1], index=0, help="koi_fpflag_ss")
        koi_fpflag_ec = st.selectbox("Eclipsing Binary Flag (0/1)", ["— Select —", 0, 1], index=0, help="koi_fpflag_ec")

    all_numeric = all(v is not None for v in [koi_score, koi_model_snr, koi_impact, koi_duration, koi_prad, koi_period])
    all_flags_selected = all(v in (0,1) for v in [
        koi_fpflag_nt if koi_fpflag_nt != "— Select —" else None,
        koi_fpflag_co if koi_fpflag_co != "— Select —" else None,
        koi_fpflag_ss if koi_fpflag_ss != "— Select —" else None,
        koi_fpflag_ec if koi_fpflag_ec != "— Select —" else None,
    ])

    can_classify = all_numeric and all_flags_selected and (pipeline is not None and label_encoder is not None)

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button("🔮 Classify Candidate", use_container_width=True, type="primary"):
            if not can_classify:
                st.error("Please fill in all fields (numbers + choose 0/1 for each flag) and ensure the model is loaded.")
            else:
                row = pd.DataFrame([
                    {
                        "koi_score": koi_score,
                        "koi_fpflag_nt": int(koi_fpflag_nt),
                        "koi_model_snr": koi_model_snr,
                        "koi_fpflag_co": int(koi_fpflag_co),
                        "koi_fpflag_ss": int(koi_fpflag_ss),
                        "koi_fpflag_ec": int(koi_fpflag_ec),
                        "koi_impact": koi_impact,
                        "koi_duration": koi_duration,
                        "koi_prad": koi_prad,
                        "koi_period": koi_period,
                    }
                ])
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
                fig = go.Figure(data=[go.Bar(x=classes, y=(proba * 100).round(2), text=[f"{p*100:.1f}%" for p in proba], textposition='auto')])
                fig.update_layout(title="Classification Probabilities", xaxis_title="Class", yaxis_title="Probability (%)", template="plotly_dark", height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Tab 3: Pretrained Model (info only — no training UI)
# -------------------------
with tab3:
    st.markdown(
        """
        <div class='reveal-element' style='text-align:center;padding:20px;background:rgba(255,255,255,0.03); border-radius:15px;margin-bottom:30px;border:1px solid rgba(0,255,255,0.2);'>
            <h3 style='margin:0;'>🧠 Pretrained Model</h3>
            <p style='color:#a0a0ff;margin-top:10px;'>This app uses your already-trained pipeline + label encoder. Upload replacements in the sidebar if needed.</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### 📦 Artifacts
        - **Pipeline:** `gb_pipeline.joblib`
        - **Label Encoder:** `label_encoder.joblib`
        - **Features (10):** `koi_score, koi_fpflag_nt, koi_model_snr, koi_fpflag_co, koi_fpflag_ss, koi_fpflag_ec, koi_impact, koi_duration, koi_prad, koi_period`
        """)
    with col2:
        st.markdown("""
        ### ✅ Why This Setup
        - Standardization & imputation are embedded in the pipeline
        - Labels are decoded via the saved encoder
        - No random/demo data anywhere
        """)

# -------------------------
# Tab 4: Visualizations (same set as before, only when results exist)
# -------------------------
with tab4:
    st.markdown(
        """
        <div class='reveal-element' style='text-align:center;padding:20px;background:rgba(255,255,255,0.03); border-radius:15px;margin-bottom:30px;border:1px solid rgba(0,255,255,0.2);'>
            <h3 style='margin:0;'>📈 Data Visualizations</h3>
            <p style='color:#a0a0ff;margin-top:10px;'>Explore patterns in your analyzed results.</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    if "results" in st.session_state:
        results = st.session_state["results"]
        viz_type = st.selectbox("Select Visualization", ["Distribution Overview", "Feature Correlations", "Habitable Zone-ish", "3D Explorer"]) 

        if viz_type == "Distribution Overview":
            c1, c2 = st.columns(2)
            with c1:
                vc = results["prediction"].value_counts()
                fig = px.pie(values=vc.values, names=vc.index, title="Classification Distribution")
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.histogram(results, x='confidence', nbins=20, title="Confidence Distribution")
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)

        elif viz_type == "Feature Correlations":
            numeric_cols = [c for c in SELECTED_FEATURES if c in results.columns]
            if len(numeric_cols) >= 3:
                feats = numeric_cols[:4]
                fig = px.scatter_matrix(results, dimensions=feats, color='prediction', title="Feature Correlation Matrix")
                fig.update_layout(template="plotly_dark", height=800)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Not enough numeric features available.")

        elif viz_type == "Habitable Zone-ish":
            if all(col in results.columns for col in ["koi_prad","koi_period"]):
                fig = px.scatter(results, x="koi_period", y="koi_prad", color="prediction", size="confidence", title="Radius vs Period (confidence-scaled)")
                fig.update_layout(template="plotly_dark", height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Required features missing: koi_period & koi_prad")

        elif viz_type == "3D Explorer":
            have = [c for c in ["koi_model_snr","koi_prad","koi_period"] if c in results.columns]
            if len(have) == 3:
                fig = px.scatter_3d(results, x=have[0], y=have[1], z=have[2], color="prediction", size="confidence", title="3D Explorer: SNR vs Radius vs Period")
                fig.update_layout(template="plotly_dark", height=700)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need koi_model_snr, koi_prad, koi_period for 3D view.")
    else:
        st.info("📊 Run an analysis in 'Batch Analysis' to see visualizations.")

# -------------------------
# Tab 5: About
# -------------------------
with tab5:
    st.markdown(
        """
        <div class='reveal-element' style='text-align:center;padding:20px;background:rgba(255,255,255,0.03);border-radius:15px;margin-bottom:30px;'>
            <h3 style='margin:0;'>ℹ️ About Celestial Ai</h3>
            <p style='color:#a0a0ff;margin-top:10px;'>NASA Space Apps Challenge 2025 — Kepler 10-feature model edition</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""
            ### 🌌 Project Overview
            This build integrates your trained Kepler pipeline that uses exactly 10 features:
            `koi_score, koi_fpflag_nt, koi_model_snr, koi_fpflag_co, koi_fpflag_ss, koi_fpflag_ec, koi_impact, koi_duration, koi_prad, koi_period`.
            It supports batch analysis (10-row sampling), single-candidate classification, and interactive visuals.
        """)
    with col2:
        st.markdown("""
            ### 🏆 Highlights
            - Exact feature alignment to your model  
            - No random/demo data  
            - Clean UI with metrics & downloads  
            - Plotly visuals and 3D explorer
        """)

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.03);border-radius:15px;'>
            <h4 style='color:#00ffff;margin:0;'>👨‍🚀 Built for NASA Space Apps 2025</h4>
            <p style='color:#a0a0ff;'>Exploring new worlds through AI</p>
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
        <p style='color:#8890ff;font-size:0.9rem;'>
            Celestial Ai • NASA Space Apps 2025 • <span style='color:#00ffff;'>Kepler 10-Feature Pipeline</span>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
""",
    unsafe_allow_html=True,
)

