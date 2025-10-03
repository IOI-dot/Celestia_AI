# ExoHunter AI (Kepler 10-feature edition) — Streamlit app
# Integrates trained pipeline: kepler_gb_pipeline_weighted.joblib + kepler_label_encoder.joblib
# Beautiful UI, robust fallbacks, batch + single classify, visuals

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import warnings
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings('ignore')

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="ExoHunter AI - NASA Exoplanet Detection",
    layout="wide",
    page_icon="🌌",
    initial_sidebar_state="expanded"
)

# =========================
# Fixed CSS / Theme
# =========================
def inject_custom_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Grotesk:wght@300;400;700&display=swap');

        .stApp { background: linear-gradient(-45deg, #0a0118, #1a0033, #0f0c29, #24243e);
                 background-size: 400% 400%; animation: gradientShift 15s ease infinite; }
        @keyframes gradientShift { 0%{background-position:0% 50%} 50%{background-position:100% 50%} 100%{background-position:0% 50%} }

        .stars{ position:fixed; top:0; left:0; width:100%; height:100%; pointer-events:none;
                background-image:
                    radial-gradient(2px 2px at 20px 30px, white, transparent),
                    radial-gradient(2px 2px at 40px 70px, white, transparent),
                    radial-gradient(1px 1px at 50px 50px, white, transparent),
                    radial-gradient(1px 1px at 80px 10px, white, transparent),
                    radial-gradient(2px 2px at 130px 80px, white, transparent);
                background-repeat: repeat; background-size:200px 200px;
                animation: stars 10s linear infinite; opacity:0.3; z-index:0; }
        @keyframes stars { 0% {transform: translateY(0);} 100% {transform: translateY(-200px);} }

        .main .block-container { position:relative; z-index:10; padding-top:2rem; }

        h1 { font-family:'Orbitron', monospace !important; font-weight:900 !important;
             background: linear-gradient(120deg, #00ffff, #ff00ff, #00ffff);
             background-size:200% auto; background-clip:text; -webkit-background-clip:text;
             -webkit-text-fill-color:transparent; animation: shine 3s linear infinite;
             text-align:center; font-size:3rem !important; margin-bottom:2rem !important;
             text-shadow:0 0 30px rgba(0,255,255,0.5); }
        @keyframes shine { to { background-position: 200% center; } }

        h2,h3 { font-family:'Space Grotesk', sans-serif !important; color:#e0e0ff !important;
                text-shadow:0 0 10px rgba(100,100,255,0.3); }

        .stTabs [data-baseweb="tab-list"] { background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);
                                            border-radius:15px; padding:10px; border:1px solid rgba(255,255,255,0.1);}
        .stTabs [data-baseweb="tab"] { color:#a0a0ff !important; font-family:'Space Grotesk', sans-serif !important;
                                       font-weight:600; transition: all .3s ease; }
        .stTabs [aria-selected="true"] { background: linear-gradient(90deg, rgba(0,255,255,0.2), rgba(255,0,255,0.2));
                                         border-radius:10px; }

        .stTextInput input, .stNumberInput input, .stSelectbox select {
            background: rgba(255,255,255,0.05) !important; border: 1px solid rgba(0,255,255,0.3) !important;
            color:#fff !important; border-radius:10px !important; backdrop-filter: blur(5px);
            transition: all .3s ease;
        }
        .stTextInput input:focus, .stNumberInput input:focus {
            border-color:#00ffff !important; box-shadow:0 0 20px rgba(0,255,255,0.3) !important;
        }

        .stButton > button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color:white; border:none;
            padding:10px 30px; border-radius:30px; font-weight:700; font-family:'Space Grotesk', sans-serif;
            transition: all .3s ease; box-shadow:0 4px 15px rgba(102,126,234,0.3);
        }
        .stButton > button:hover { transform: translateY(-2px); box-shadow:0 6px 25px rgba(102,126,234,0.5); }

        .stAlert { background: rgba(255,255,255,0.05) !important; backdrop-filter: blur(10px);
                   border:1px solid rgba(255,255,255,0.2); border-radius:15px; animation: slideIn .5s ease; }
        @keyframes slideIn { from {opacity:0; transform: translateY(-20px);} to{opacity:1; transform: translateY(0);} }

        [data-testid="metric-container"] { background: rgba(255,255,255,0.05); backdrop-filter: blur(10px);
                                           border:1px solid rgba(255,255,255,0.1); padding:15px; border-radius:15px;
                                           box-shadow:0 4px 15px rgba(0,0,0,0.2); }

        [data-testid="stFileUploadDropzone"] { background: rgba(255,255,255,0.03);
                                               border:2px dashed rgba(0,255,255,0.3); border-radius:15px; transition:.3s; }
        [data-testid="stFileUploadDropzone"]:hover { background: rgba(0,255,255,0.05); border-color: rgba(0,255,255,0.6); }

        .stProgress > div > div > div { background: linear-gradient(90deg, #00ffff, #ff00ff);
                                         border-radius:10px; animation: pulse 2s infinite; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.8} }

        #MainMenu, header, footer { visibility: hidden; }

        ::-webkit-scrollbar { width:10px; background: rgba(255,255,255,0.05); }
        ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #667eea, #764ba2); border-radius:10px; }

        .streamlit-expanderHeader { background: rgba(255,255,255,0.05) !important; border-radius:10px !important; }
    </style>
    <div class="stars"></div>
    """, unsafe_allow_html=True)

inject_custom_css()

# =========================
# Constants / Features
# =========================
# EXACT 10 features your model was trained on
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
# Smart Demo Model (fallback)
# =========================
class SmartDummyModel:
    """Fallback model that mimics plausible behavior on the 10 Kepler features."""
    def __init__(self, classes=None):
        self.classes_ = np.array(classes or ["FALSE POSITIVE","CANDIDATE","CONFIRMED"])
        np.random.seed(42)

    def _score_row(self, row):
        score = 0.0
        # higher koi_score good
        score += float(row.get("koi_score", 0)) * 2.0
        # strong signal-to-noise
        snr = float(row.get("koi_model_snr", 0))
        if snr > 25: score += 2.5
        elif snr > 15: score += 1.5
        # plausible impact parameter
        imp = float(row.get("koi_impact", 0))
        if 0.1 <= imp <= 0.9: score += 1.0
        # reasonable duration & period (very rough heuristics)
        dur = float(row.get("koi_duration", 0))
        if 2 <= dur <= 20: score += 1.0
        period = float(row.get("koi_period", 0))
        if 0.5 <= period <= 400: score += 1.0
        # radius in a reasonable range
        prad = float(row.get("koi_prad", 0))
        if 0.8 <= prad <= 2.5: score += 2.0
        elif prad <= 4.0: score += 0.5
        # penalize certain FP flags
        for f in ["koi_fpflag_co","koi_fpflag_ss","koi_fpflag_ec"]:
            if int(row.get(f, 0)) == 1: score -= 1.0
        # NT flag = not transit like (0 means OK), so prefer 0
        if int(row.get("koi_fpflag_nt", 0)) == 0: score += 0.5
        # noise
        score += np.random.normal(0, 0.4)
        return score

    def predict(self, X):
        preds = []
        for _, r in pd.DataFrame(X, columns=SELECTED_FEATURES).iterrows():
            s = self._score_row(r)
            if s >= 5.5: preds.append(2)   # CONFIRMED
            elif s >= 2.0: preds.append(1) # CANDIDATE
            else: preds.append(0)          # FALSE POSITIVE
        return np.array(preds)

    def predict_proba(self, X):
        y = self.predict(X)
        n = len(y); k = len(self.classes_)
        P = np.zeros((n,k))
        for i, c in enumerate(y):
            base = np.random.dirichlet(np.ones(k)*0.7)
            base[c] += 0.6
            P[i] = base / base.sum()
        return P

def create_label_encoder():
    le = LabelEncoder()
    le.fit(['FALSE POSITIVE', 'CANDIDATE', 'CONFIRMED'])
    return le

# =========================
# Load your trained artifacts (safe)
# =========================
@st.cache_resource
def load_model_safe(file_or_path):
    try:
        return joblib.load(file_or_path)
    except Exception:
        return None

# =========================
# Header
# =========================
def show_header():
    st.markdown("""
        <h1 style='text-align:center;margin-bottom:0;'>🌌 ExoHunter AI</h1>
        <p style='text-align:center;color:#a0a0ff;font-family:Space Grotesk;font-size:1.2rem;margin-top:0;'>
            Advanced Exoplanet Detection System • NASA Space Apps 2025
        </p>
        <div style='text-align:center;margin:20px 0;'>
            <span style='background:linear-gradient(90deg,#667eea,#764ba2);padding:5px 15px;border-radius:20px;color:#fff;font-size:.9rem;font-family:Space Grotesk;'>
                Powered by Your Kepler 10-Feature Model
            </span>
        </div>
    """, unsafe_allow_html=True)

show_header()

# =========================
# Sidebar: Model Controls
# =========================
with st.sidebar:
    st.markdown("""
        <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.05);
                    border-radius:15px;margin-bottom:20px;border:1px solid rgba(0,255,255,0.3);'>
            <h2 style='margin:0;font-size:1.5rem;'>🚀 Control Panel</h2>
        </div>
    """, unsafe_allow_html=True)

    with st.expander("🔧 Model Configuration", expanded=True):
        # Allow uploading alternative pipeline/encoder if desired
        model_upload = st.file_uploader("Upload Pipeline (.joblib)", type=['joblib'], key='model')
        encoder_upload = st.file_uploader("Upload Encoder (.joblib)", type=['joblib'], key='encoder')

        use_demo = st.checkbox('🎮 Demo Mode (use synthetic/randomized predictions if no model found)', value=False)
        debug_mode = st.checkbox('🔍 Debug Mode', value=False, help="Show technical details in sidebar")

    # Load priority: uploaded -> local -> fallback
    pipeline = None
    label_encoder = None

    if model_upload and encoder_upload:
        pipeline = load_model_safe(model_upload)
        label_encoder = load_model_safe(encoder_upload)
        if pipeline and label_encoder:
            st.success("✅ Custom artifacts loaded (uploaded).")
    else:
        # Try your saved filenames from training script
        pipeline = load_model_safe('kepler_gb_pipeline_weighted.joblib')
        label_encoder = load_model_safe('kepler_label_encoder.joblib')
        if pipeline and label_encoder:
            st.info("📁 Using local artifacts: kepler_gb_pipeline_weighted.joblib + kepler_label_encoder.joblib")
        else:
            # Fallback
            pipeline = SmartDummyModel()
            label_encoder = create_label_encoder()
            if use_demo:
                st.warning("⚠️ Using demo fallback model (upload/load real artifacts for production).")
            else:
                st.warning("⚠️ Artifacts not found — switched to demo fallback model.")

    st.markdown("### 📊 System Status")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Model", "✅ Ready" if pipeline is not None else "❌ Missing", delta="Active" if pipeline else None)
    with c2:
        st.metric("Features", f"{len(SELECTED_FEATURES)} used", delta="Kepler 10-feature")

# =========================
# Utilities
# =========================
def align_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure the dataframe has exactly the required 10 features, numeric, in order."""
    X = df.copy()
    # create missing columns with sensible defaults
    for col in SELECTED_FEATURES:
        if col not in X.columns:
            # FP flags are binary; others numeric
            if col.startswith("koi_fpflag_"):
                X[col] = 0
            else:
                X[col] = 0.0
    # coerce numeric
    for col in SELECTED_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    # order
    X = X[SELECTED_FEATURES]
    return X

def generate_demo_data(n=60) -> pd.DataFrame:
    """Synthetic rows with those 10 features + fake labels for demos."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "koi_score": rng.uniform(0, 1, n),
        "koi_fpflag_nt": rng.integers(0, 2, n),
        "koi_model_snr": rng.normal(20, 8, n).clip(0, None),
        "koi_fpflag_co": rng.integers(0, 2, n),
        "koi_fpflag_ss": rng.integers(0, 2, n),
        "koi_fpflag_ec": rng.integers(0, 2, n),
        "koi_impact": rng.uniform(0, 1, n),
        "koi_duration": rng.normal(10, 5, n).clip(0.5, None),
        "koi_prad": rng.uniform(0.5, 6.0, n),
        "koi_period": np.exp(rng.normal(np.log(30), 1.0, n)).clip(0.5, 800),
    })
    # lightweight label heuristic for pretty demos
    lab = []
    for _, r in df.iterrows():
        s = 0
        s += r["koi_score"] * 2 + (r["koi_model_snr"] > 20) * 1.5 + (0.1 <= r["koi_impact"] <= 0.9) * 0.8
        s += (2 <= r["koi_duration"] <= 20) * 0.7 + (0.8 <= r["koi_prad"] <= 2.5) * 1.2 + (0.5 <= r["koi_period"] <= 400) * 0.8
        s -= (r["koi_fpflag_co"] + r["koi_fpflag_ss"] + r["koi_fpflag_ec"]) * 0.9
        if s >= 4.2: lab.append("CONFIRMED")
        elif s >= 2.0: lab.append("CANDIDATE")
        else: lab.append("FALSE POSITIVE")
    df["disposition"] = lab
    return df

def decode_labels(y_pred_int: np.ndarray) -> np.ndarray:
    try:
        if hasattr(label_encoder, "inverse_transform"):
            return label_encoder.inverse_transform(y_pred_int)
        else:
            classes_ = getattr(label_encoder, "classes_", np.array(["FALSE POSITIVE","CANDIDATE","CONFIRMED"]))
            return np.array([classes_[i] for i in y_pred_int])
    except Exception:
        classes_ = np.array(["FALSE POSITIVE","CANDIDATE","CONFIRMED"])
        return np.array([classes_[i] for i in y_pred_int])

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍 Batch Analysis",
    "✨ Quick Classify",
    "🧬 Model Training (simple)",
    "📈 Visualizations",
    "ℹ️ About"
])

# -------------------------
# Tab 1: Batch Analysis
# -------------------------
with tab1:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
            <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.03);
                        border-radius:15px;margin-bottom:30px;'>
                <h3 style='margin:0;'>🚀 Batch Exoplanet Analysis</h3>
                <p style='color:#a0a0ff;margin-top:10px;'>
                    Upload CSV with the <b>10 Kepler features</b> or generate demo data.
                </p>
            </div>
        """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload CSV Dataset (must include the 10 feature columns)", type=['csv'])

    df = None
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, comment="#")
            df.columns = df.columns.str.strip()
            st.success(f"✅ Loaded {len(df)} rows from file")
        except Exception as e:
            st.error(f"Error reading file: {e}")

    if df is None:
        if st.button("🎲 Generate Demo Dataset", use_container_width=True, key="gen_demo_batch"):
            df = generate_demo_data(120)
            st.session_state['demo_data'] = df
            st.success("✅ Generated 120 synthetic candidates!")
    if 'demo_data' in st.session_state and df is None:
        df = st.session_state['demo_data']

    if df is not None:
        with st.expander("📋 Dataset Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)

        c1, c2, c3 = st.columns([1,1,1])
        with c2:
            if st.button("🔬 Run Analysis", use_container_width=True, type="primary"):
                with st.spinner("Analyzing candidates..."):
                    progress_bar = st.progress(0)
                    X = align_features_df(df)
                    progress_bar.progress(30)
                    preds = pipeline.predict(X)
                    probas = pipeline.predict_proba(X)
                    progress_bar.progress(70)
                    labels = decode_labels(preds)
                    results = df.copy()
                    results['prediction'] = labels
                    results['confidence'] = (probas.max(axis=1) * 100).round(2)
                    # Assume class order matches encoder; take index for 'CONFIRMED'
                    try:
                        classes = list(getattr(label_encoder, "classes_", ["FALSE POSITIVE","CANDIDATE","CONFIRMED"]))
                        confirmed_idx = classes.index("CONFIRMED")
                    except ValueError:
                        confirmed_idx = -1
                    if confirmed_idx >= 0:
                        results['confirmed_prob'] = (probas[:, confirmed_idx] * 100).round(2)
                    else:
                        results['confirmed_prob'] = (probas.max(axis=1) * 100).round(2)
                    progress_bar.progress(100)
                    st.session_state['results'] = results
                    st.success(f"✅ Analysis complete! Processed {len(results)} candidates")
                    time.sleep(0.3)

    # Show results
    if 'results' in st.session_state:
        results = st.session_state['results']
        st.markdown("### 📊 Analysis Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            confirmed = (results['prediction'] == 'CONFIRMED').sum()
            st.metric("🌟 Confirmed", int(confirmed), delta=f"{confirmed/len(results)*100:.1f}%")
        with c2:
            cand = (results['prediction'] == 'CANDIDATE').sum()
            st.metric("🔍 Candidates", int(cand), delta=f"{cand/len(results)*100:.1f}%")
        with c3:
            fp = (results['prediction'] == 'FALSE POSITIVE').sum()
            st.metric("❌ False Positives", int(fp), delta=f"{fp/len(results)*100:.1f}%")
        with c4:
            avg_conf = float(results['confidence'].mean())
            st.metric("💪 Avg Confidence", f"{avg_conf:.1f}%", delta="High" if avg_conf > 70 else "Moderate")

        st.markdown("### 🔍 Detailed Results")
        f1, f2, f3 = st.columns(3)
        with f1:
            filt = st.selectbox("Filter by prediction:", ["All","CONFIRMED","CANDIDATE","FALSE POSITIVE"])
        with f2:
            min_conf = st.slider("Min confidence:", 0, 100, 0)
        with f3:
            sort_by = st.selectbox("Sort by:", ["confidence","confirmed_prob","koi_score","koi_model_snr","koi_period"])
        view = results.copy()
        if filt != "All":
            view = view[view['prediction'] == filt]
        view = view[view['confidence'] >= min_conf]
        if sort_by in view.columns:
            view = view.sort_values(sort_by, ascending=False)
        st.dataframe(view, use_container_width=True)

        csv = view.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name=f"exoplanet_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# -------------------------
# Tab 2: Quick Classify
# -------------------------
with tab2:
    st.markdown("""
        <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.03);
                    border-radius:15px;margin-bottom:30px;'>
            <h3 style='margin:0;'>✨ Quick Candidate Classification</h3>
            <p style='color:#a0a0ff;margin-top:10px;'>Enter the 10 trained features.</p>
        </div>
    """, unsafe_allow_html=True)

    # Inputs in logical groups
    st.markdown("#### 📡 Signal & Score")
    q1, q2, q3 = st.columns(3)
    with q1:
        koi_score = st.number_input("koi_score", min_value=0.0, step=0.01, value=0.5)
    with q2:
        koi_model_snr = st.number_input("koi_model_snr", min_value=0.0, step=0.1, value=20.0)
    with q3:
        koi_impact = st.number_input("koi_impact", min_value=0.0, max_value=1.5, step=0.01, value=0.5)

    st.markdown("#### 🪐 Geometry & Timing")
    r1, r2, r3 = st.columns(3)
    with r1:
        koi_duration = st.number_input("koi_duration (hours)", min_value=0.0, step=0.1, value=10.0)
    with r2:
        koi_prad = st.number_input("koi_prad (Earth radii)", min_value=0.0, step=0.1, value=1.5)
    with r3:
        koi_period = st.number_input("koi_period (days)", min_value=0.0, step=0.1, value=30.0)

    st.markdown("#### 🚩 False-Positive Flags (0/1)")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        koi_fpflag_nt = st.selectbox("koi_fpflag_nt", [0,1], index=0, help="Not transit-like flag")
    with f2:
        koi_fpflag_co = st.selectbox("koi_fpflag_co", [0,1], index=0, help="Centroid offset")
    with f3:
        koi_fpflag_ss = st.selectbox("koi_fpflag_ss", [0,1], index=0, help="Significant secondary")
    with f4:
        koi_fpflag_ec = st.selectbox("koi_fpflag_ec", [0,1], index=0, help="Eclipsing binary")

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button("🔮 Classify Candidate", use_container_width=True, type="primary"):
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

                time.sleep(0.3)

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
                    title = "❌ FALSE POSITIVE"
                    sub = "This signal is likely not from a real exoplanet."

                st.markdown(f"""
                    <div style='text-align:center;padding:30px;background:linear-gradient(135deg,{color_block[0]},{color_block[0]});
                                border-radius:20px;border:2px solid {color_block[1]};'>
                        <h1 style='margin:0;'>{title}</h1>
                        <h2>Confidence: {confidence:.1f}%</h2>
                        <p style='color:#cfe3ff;'>{sub}</p>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown("### 📊 Probability Distribution")
                classes = list(getattr(label_encoder, "classes_", ["FALSE POSITIVE","CANDIDATE","CONFIRMED"]))
                fig = go.Figure(data=[go.Bar(
                    x=classes,
                    y=(proba * 100).round(2),
                    text=[f"{p*100:.1f}%" for p in proba],
                    textposition='auto'
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
# Tab 3: Model Training (simple, optional)
# -------------------------
with tab3:
    st.markdown("""
        <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.03);
                    border-radius:15px;margin-bottom:30px;'>
            <h3 style='margin:0;'>🧬 Train Custom Model (with 10 features)</h3>
            <p style='color:#a0a0ff;margin-top:10px;'>
                Upload a CSV containing the 10 feature columns and a 'koi_disposition' column.
            </p>
        </div>
    """, unsafe_allow_html=True)

    train_file = st.file_uploader("Upload Training Dataset (CSV)", type=['csv'], key='training_file')
    if train_file is not None:
        try:
            train_df = pd.read_csv(train_file, comment="#")
            train_df.columns = train_df.columns.str.strip()
            st.success(f"✅ Loaded {len(train_df)} training samples")
            with st.expander("📋 Training Data Preview"):
                st.dataframe(train_df.head(), use_container_width=True)

            # Quick train controls
            colA, colB = st.columns(2)
            with colA:
                test_size = st.slider("Test size", 0.1, 0.4, 0.2, step=0.05)
            with colB:
                max_iter = st.number_input("Max Iterations", 50, 1000, 500, step=50)

            if st.button("🚀 Start Training", type="primary"):
                with st.spinner("Training model..."):
                    # Prepare data
                    # ensure disposition column exists
                    if "koi_disposition" not in train_df.columns:
                        st.error("Training CSV must include 'koi_disposition'.")
                    else:
                        # encode labels
                        train_le = LabelEncoder()
                        y = train_le.fit_transform(train_df["koi_disposition"].astype(str))
                        X = align_features_df(train_df)  # will pick/clean the 10 features

                        # Build a simple pipeline mirroring your preprocessing
                        numeric_transformer = Pipeline(steps=[
                            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                            ("scaler", StandardScaler()),
                        ])
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
                        # Simple split (no report here to keep things snappy)
                        from sklearn.model_selection import train_test_split
                        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
                        new_pipeline.fit(Xtr, ytr)
                        acc = new_pipeline.score(Xte, yte)
                        # Save
                        joblib.dump(new_pipeline, "kepler_gb_pipeline_weighted.joblib")
                        joblib.dump(train_le, "kepler_label_encoder.joblib")
                        st.success(f"✅ Training complete. Test accuracy: {acc:.2%}")
                        c1, c2 = st.columns(2)
                        with c1:
                            with open("kepler_gb_pipeline_weighted.joblib", "rb") as f:
                                st.download_button("📥 Download Pipeline", f.read(), "kepler_gb_pipeline_weighted.joblib")
                        with c2:
                            with open("kepler_label_encoder.joblib", "rb") as f:
                                st.download_button("📥 Download Encoder", f.read(), "kepler_label_encoder.joblib")
        except Exception as e:
            st.error(f"Error during training: {e}")
    else:
        st.info("Upload training data to (re)train a compatible 10-feature model.")

# -------------------------
# Tab 4: Visualizations
# -------------------------
with tab4:
    st.markdown("""
        <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.03);
                    border-radius:15px;margin-bottom:30px;'>
            <h3 style='margin:0;'>📈 Data Visualizations</h3>
            <p style='color:#a0a0ff;margin-top:10px;'>Explore patterns in your analyzed results.</p>
        </div>
    """, unsafe_allow_html=True)

    if 'results' in st.session_state:
        results = st.session_state['results']
        viz_type = st.selectbox("Select Visualization", ["Distribution Overview","Feature Correlations","Habitable Zone-ish","3D Explorer"])

        if viz_type == "Distribution Overview":
            c1, c2 = st.columns(2)
            with c1:
                vc = results['prediction'].value_counts()
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
            # Not true HZ, but radius vs period proxy with confidence scaling
            if all(col in results.columns for col in ["koi_prad","koi_period"]):
                fig = px.scatter(
                    results, x="koi_period", y="koi_prad", color="prediction", size="confidence",
                    title="Radius vs Period (confidence-scaled)"
                )
                fig.update_layout(template="plotly_dark", height=600)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Required features missing: koi_period & koi_prad")

        elif viz_type == "3D Explorer":
            have = [c for c in ["koi_model_snr","koi_prad","koi_period"] if c in results.columns]
            if len(have) == 3:
                fig = px.scatter_3d(results, x=have[0], y=have[1], z=have[2],
                                    color="prediction", size="confidence",
                                    title="3D Explorer: SNR vs Radius vs Period")
                fig.update_layout(template="plotly_dark", height=700)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Need koi_model_snr, koi_prad, koi_period for 3D view.")
    else:
        st.info("📊 Run an analysis on Tab 1 to see visualizations.")
        if st.button("🎲 Generate Demo Visualization Data", use_container_width=True):
            demo = generate_demo_data(60)
            # simulate model pass
            X = align_features_df(demo)
            preds = pipeline.predict(X)
            probas = pipeline.predict_proba(X)
            labels = decode_labels(preds)
            results = demo.copy()
            results["prediction"] = labels
            results["confidence"] = (probas.max(axis=1) * 100).round(2)
            try:
                classes = list(getattr(label_encoder, "classes_", ["FALSE POSITIVE","CANDIDATE","CONFIRMED"]))
                ci = classes.index("CONFIRMED")
            except ValueError:
                ci = -1
            results["confirmed_prob"] = (probas[:, ci] * 100).round(2) if ci >= 0 else (probas.max(axis=1) * 100).round(2)
            st.session_state["results"] = results
            st.success("✅ Demo data generated — open visualizations again.")

# -------------------------
# Tab 5: About
# -------------------------
with tab5:
    st.markdown("""
        <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.03);
                    border-radius:15px;margin-bottom:30px;'>
            <h3 style='margin:0;'>ℹ️ About ExoHunter AI</h3>
            <p style='color:#a0a0ff;margin-top:10px;'>NASA Space Apps Challenge 2025 — Kepler 10-feature model edition</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])
    with col1:
        st.markdown("""
            ### 🌌 Project Overview
            This build integrates your trained Kepler pipeline that uses exactly 10 features:
            `koi_score, koi_fpflag_nt, koi_model_snr, koi_fpflag_co, koi_fpflag_ss, koi_fpflag_ec, koi_impact, koi_duration, koi_prad, koi_period`.
            It supports batch analysis, single-candidate classification, light retraining, and interactive visuals.
        """)
    with col2:
        st.markdown("""
            ### 🏆 Highlights
            - Exact feature alignment to your model  
            - Robust fallbacks if artifacts are missing  
            - Clean UI with metrics & downloads  
            - Plotly visuals and 3D explorer
        """)

    st.markdown("---")
    st.markdown("""
        <div style='text-align:center;padding:20px;background:rgba(255,255,255,0.03);border-radius:15px;'>
            <h4 style='color:#00ffff;margin:0;'>👨‍🚀 Built for NASA Space Apps 2025</h4>
            <p style='color:#a0a0ff;'>Exploring new worlds through AI</p>
        </div>
    """, unsafe_allow_html=True)

# =========================
# Debug info
# =========================
if 'debug_mode' in locals() and debug_mode:
    with st.sidebar.expander("🔍 Debug Information", expanded=False):
        st.write("**Session State Keys:**", list(st.session_state.keys()))
        st.write("**Pipeline Type:**", type(pipeline).__name__)
        if hasattr(pipeline, 'classes_'):
            st.write("**Model Classes:**", getattr(pipeline, 'classes_', None))
        if label_encoder and hasattr(label_encoder, 'classes_'):
            st.write("**Encoder Classes:**", label_encoder.classes_)

# =========================
# Footer
# =========================
st.markdown("""
    <div style='text-align:center;margin-top:50px;padding:20px;border-top:1px solid rgba(255,255,255,0.1);'>
        <p style='color:#6666ff;font-size:0.9rem;'>
            ExoHunter AI • NASA Space Apps 2025 •
            <span style='color:#00ffff;'>Kepler 10-Feature Pipeline</span>
        </p>
    </div>
""", unsafe_allow_html=True)
