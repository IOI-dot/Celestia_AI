# Celestia AI - Streamlit Edition
# NASA Kepler 10-feature Weighted Pipeline Integration
# Author: Automated Rebuild by GPT-5 Mini

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import time
import warnings
from datetime import datetime
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix

warnings.filterwarnings("ignore")

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Celestia AI - NASA Exoplanet Detection",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# Custom CSS
# =========================
def inject_css():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Space+Grotesk:wght@300;400;700&display=swap');
        .stApp { background: linear-gradient(-45deg, #2a0040, #7f3fbf, #c4a1ff, #e0e0ff); background-size: 400% 400%; animation: gradientShift 20s ease infinite; }
        @keyframes gradientShift { 0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%} }
        .stars{ position:fixed; top:0; left:0; width:100%; height:100%; pointer-events:none;
                background-image:
                    radial-gradient(2px 2px at 20px 30px, white, transparent),
                    radial-gradient(2px 2px at 40px 70px, white, transparent),
                    radial-gradient(1px 1px at 50px 50px, white, transparent),
                    radial-gradient(1px 1px at 80px 10px, white, transparent),
                    radial-gradient(2px 2px at 130px 80px, white, transparent);
                background-repeat: repeat; background-size:200px 200px;
                animation: stars 10s linear infinite; opacity:0.2; z-index:0; }
        @keyframes stars { 0% {transform: translateY(0);} 100% {transform: translateY(-200px);} }
        h1 { font-family:'Orbitron', monospace !important; font-weight:900 !important;
             background: linear-gradient(120deg, #b19cd9, #ffffff, #b19cd9);
             background-size:200% auto; background-clip:text; -webkit-background-clip:text;
             -webkit-text-fill-color:transparent; animation: shine 3s linear infinite;
             text-align:center; font-size:3rem !important; margin-bottom:2rem !important;
             text-shadow:0 0 15px rgba(255,255,255,0.3); }
        @keyframes shine { to { background-position: 200% center; } }
        h2,h3 { font-family:'Space Grotesk', sans-serif !important; color:#3f00ff !important; text-shadow:0 0 5px rgba(200,200,255,0.2);}
        .stButton > button { background: linear-gradient(90deg, #7f3fbf 0%, #c4a1ff 100%); color:white; border:none; padding:10px 30px; border-radius:20px; font-weight:700; font-family:'Space Grotesk', sans-serif; transition: all .3s ease; box-shadow:0 4px 15px rgba(127,63,191,0.3);}
        .stButton > button:hover { transform: translateY(-2px); box-shadow:0 6px 25px rgba(127,63,191,0.5); }
        [data-testid="stFileUploadDropzone"] { background: rgba(255,255,255,0.03); border:2px dashed rgba(127,63,191,0.3); border-radius:15px; transition:.3s;}
        [data-testid="stFileUploadDropzone"]:hover { background: rgba(255,255,255,0.05); border-color: rgba(127,63,191,0.6);}
        ::-webkit-scrollbar { width:10px; background: rgba(255,255,255,0.05); }
        ::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #7f3fbf, #c4a1ff); border-radius:10px; }
        #MainMenu, header, footer { visibility: hidden; }
    </style>
    <div class="stars"></div>
    """, unsafe_allow_html=True)

inject_css()

# =========================
# Constants
# =========================
SELECTED_FEATURES = [
    "koi_score", "koi_fpflag_nt", "koi_model_snr", "koi_fpflag_co",
    "koi_fpflag_ss", "koi_fpflag_ec", "koi_impact", "koi_duration",
    "koi_prad", "koi_period"
]

FRIENDLY_NAMES = {
    "koi_score":"Transit Score",
    "koi_fpflag_nt":"Not-Transit Flag (0=Good,1=Bad)",
    "koi_model_snr":"Signal-to-Noise Ratio",
    "koi_fpflag_co":"Centroid Offset Flag",
    "koi_fpflag_ss":"Secondary Signal Flag",
    "koi_fpflag_ec":"Eclipsing Binary Flag",
    "koi_impact":"Transit Impact",
    "koi_duration":"Transit Duration (hrs)",
    "koi_prad":"Planet Radius (Earth radii)",
    "koi_period":"Orbital Period (days)"
}

# =========================
# Load Model & Encoder
# =========================
@st.cache_resource
def load_artifact(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        return None

pipeline = load_artifact("kepler_gb_pipeline_weighted.joblib")
label_encoder = load_artifact("kepler_label_encoder.joblib")

if pipeline is None or label_encoder is None:
    st.error("❌ Model or Encoder not found. Train your model first!")
    st.stop()

# =========================
# Header
# =========================
st.markdown("<h1>🪐 Celestia AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#3f00ff;'>NASA Kepler 10-Feature Exoplanet Detection</p>", unsafe_allow_html=True)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("## 🚀 Control Panel")
    debug_mode = st.checkbox("🔍 Debug Mode", value=False)
    uploaded_file = st.file_uploader("📁 Upload CSV Dataset (must include 10 features)", type=['csv'])

# =========================
# Utilities
# =========================
def align_features_df(df):
    X = df.copy()
    for col in SELECTED_FEATURES:
        if col not in X.columns:
            X[col] = 0.0 if not col.startswith("koi_fpflag") else 0
    for col in SELECTED_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0)
    return X[SELECTED_FEATURES]

def decode_labels(y_pred):
    return label_encoder.inverse_transform(y_pred)

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Batch Analysis","✨ Quick Classify","📈 Visualizations","ℹ️ About"])

# -------------------------
# Tab 1: Batch Analysis
# -------------------------
with tab1:
    st.markdown("### Upload your dataset for batch prediction")
    df = None
    if uploaded_file:
        df = pd.read_csv(uploaded_file, comment="#")
        df.columns = df.columns.str.strip()
        st.success(f"✅ Loaded {len(df)} rows")
        st.dataframe(df.head(), use_container_width=True)

    if df is not None:
        if st.button("🔬 Run Analysis"):
            with st.spinner("Predicting..."):
                X = align_features_df(df)
                y_pred = pipeline.predict(X)
                proba = pipeline.predict_proba(X)
                labels = decode_labels(y_pred)
                df["prediction"] = labels
                df["confidence"] = (proba.max(axis=1)*100).round(2)
                classes = label_encoder.classes_
                try: confirmed_idx = list(classes).index("CONFIRMED")
                except: confirmed_idx=-1
                df["confirmed_prob"] = (proba[:,confirmed_idx]*100).round(2) if confirmed_idx>=0 else df["confidence"]
                st.session_state["batch_results"] = df
                st.success(f"✅ Predictions complete for {len(df)} candidates!")

    if "batch_results" in st.session_state:
        results = st.session_state["batch_results"]
        st.markdown("### 📊 Summary Metrics")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("🌟 CONFIRMED", int((results['prediction']=="CONFIRMED").sum()))
        c2.metric("🔍 CANDIDATES", int((results['prediction']=="CANDIDATE").sum()))
        c3.metric("❌ FALSE POSITIVES", int((results['prediction']=="FALSE POSITIVE").sum()))
        c4.metric("💪 Avg Confidence", f"{results['confidence'].mean():.1f}%")

        st.markdown("### 🔍 Filter & View Results")
        filt = st.selectbox("Filter by Prediction", ["All","CONFIRMED","CANDIDATE","FALSE POSITIVE"])
        min_conf = st.slider("Min Confidence", 0, 100, 0)
        view = results.copy()
        if filt != "All": view = view[view['prediction']==filt]
        view = view[view['confidence']>=min_conf]
        st.dataframe(view, use_container_width=True)

        csv = view.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, file_name=f"celestia_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

# -------------------------
# Tab 2: Quick Classify
# -------------------------
with tab2:
    st.markdown("### Enter Candidate Features")
    inputs = {}
    for f in SELECTED_FEATURES:
        if "fpflag" in f:
            inputs[f] = st.selectbox(FRIENDLY_NAMES[f],[0,1], index=0)
        else:
            inputs[f] = st.number_input(FRIENDLY_NAMES[f],value=0.5,step=0.01)

    if st.button("🔮 Classify Candidate"):
        row = pd.DataFrame([inputs])
        X = align_features_df(row)
        y_pred = pipeline.predict(X)[0]
        proba = pipeline.predict_proba(X)[0]
        label = decode_labels(np.array([y_pred]))[0]
        conf = float(proba.max()*100)

        color = "rgba(127,63,191,0.3)" if label=="CONFIRMED" else ("rgba(255,255,0,0.3)" if label=="CANDIDATE" else "rgba(255,0,0,0.3)")
        st.markdown(f"<div style='padding:20px;background:{color};border-radius:15px;text-align:center;'>Prediction: <b>{label}</b><br>Confidence: {conf:.1f}%</div>", unsafe_allow_html=True)

        # Probability plot
        fig = go.Figure([go.Bar(x=label_encoder.classes_, y=(proba*100).round(2), text=[f"{p*100:.1f}%" for p in proba], textposition='auto')])
        fig.update_layout(title="Class Probabilities",template="plotly_dark",height=400)
        st.plotly_chart(fig,use_container_width=True)

# -------------------------
# Tab 3: Visualizations
# -------------------------
with tab3:
    st.markdown("### Visualize Results")
    if "batch_results" in st.session_state:
        results = st.session_state["batch_results"]
        fig1 = px.pie(results, names="prediction", title="Prediction Distribution")
        fig1.update_layout(template="plotly_dark",height=400)
        st.plotly_chart(fig1,use_container_width=True)

        fig2 = px.histogram(results, x="confidence", nbins=20, title="Confidence Distribution")
        fig2.update_layout(template="plotly_dark",height=400)
        st.plotly_chart(fig2,use_container_width=True)

        if all(c in results.columns for c in ["koi_prad","koi_period"]):
            fig3 = px.scatter(results,x="koi_period",y="koi_prad",color="prediction",size="confidence",title="Planet Radius vs Period")
            fig3.update_layout(template="plotly_dark",height=600)
            st.plotly_chart(fig3,use_container_width=True)
    else:
        st.info("Upload or run batch analysis to see visualizations.")

# -------------------------
# Tab 4: About
# -------------------------
with tab4:
    st.markdown("### ℹ️ About Celestia AI")
    st.markdown("""
    Celestia AI is a NASA Kepler 10-feature exoplanet detection tool.
    - Uses exactly 10 features from Kepler data
    - Powered by HistGradientBoostingClassifier with weighted classes
    - Supports batch CSV upload and single candidate classification
    - Interactive Plotly visualizations and confidence metrics
    """)
    st.markdown("<p style='text-align:center;color:#7f3fbf;'>🪐 Built for NASA Kepler Enthusiasts & Scientists</p>", unsafe_allow_html=True)

# =========================
# Footer
# =========================
st.markdown("<div style='position:fixed;bottom:0;width:100%;text-align:center;color:#ffffff;padding:10px;font-size:12px;'>© 2025 Celestia AI | All rights reserved</div>", unsafe_allow_html=True)
