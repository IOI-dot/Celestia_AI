# app.py
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, f1_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

# -------------------------
# Page config & global CSS
# -------------------------
st.set_page_config(page_title="ExoClass — Exoplanet Classifier", layout="wide", page_icon="🚀")
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #05032d 0%, #071a3a 40%, #0b2b4a 100%);
    color: #e6f0ff;
}
[data-testid="stSidebar"] > div:first-child {
    background: linear-gradient(180deg,#04122b,#08243f);
    border-right: 1px solid rgba(255,255,255,0.04);
    padding: 16px;
    color: #dbefff;
}
.sidebar-title {
    color: #ffd966;
    font-weight: 700;
    font-size: 18px;
    margin-bottom: 6px;
}
.card {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 12px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.45);
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Utilities
# -------------------------
@st.cache_data
def load_label_encoder(path="label_encoder.joblib"):
    try:
        return joblib.load(path)
    except Exception:
        return None

@st.cache_resource
def load_model(path="gb_pipeline.joblib"):
    try:
        return joblib.load(path)
    except Exception:
        return None

def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    st.pyplot(fig)

def plot_feature_importance(importances, feature_names):
    fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8,4))
    fi.plot(kind="bar", ax=ax, color="#6ea8fe")
    ax.set_title("Top 20 Feature Importances")
    st.pyplot(fig)

def plot_roc_curve(y_true, y_score, classes):
    y_bin = label_binarize(y_true, classes=range(len(classes)))
    n_classes = y_bin.shape[1]
    fig, ax = plt.subplots(figsize=(6,5))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"{classes[i]} (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (One-vs-Rest)")
    ax.legend(loc="lower right")
    st.pyplot(fig)

def get_expected_columns_from_pipeline(model):
    preproc = model.named_steps.get('preprocessor', None)
    if preproc is None:
        raise ValueError("Model pipeline has no 'preprocessor' step.")
    transformers = preproc.transformers_
    numeric_cols = []
    cat_cols = []
    for name, transformer, cols in transformers:
        if name == 'num':
            numeric_cols = list(cols)
        elif name == 'cat':
            cat_cols = list(cols)
    return numeric_cols, cat_cols

def align_features(X_new, model):
    try:
        expected_num, expected_cat = get_expected_columns_from_pipeline(model)
    except Exception as e:
        st.warning(f"Could not extract expected columns from pipeline: {e}")
        return X_new
    expected = expected_num + expected_cat
    Xp = X_new.copy()

    for col in expected_num:
        if col not in Xp.columns:
            # Respect dtype by filling 0 with int or float depending on other data
            Xp[col] = 0
        elif not np.issubdtype(Xp[col].dtype, np.number):
            Xp[col] = pd.to_numeric(Xp[col], errors='coerce').fillna(0)

    for col in expected_cat:
        if col not in Xp.columns:
            Xp[col] = "missing"
        else:
            Xp[col] = Xp[col].astype(str)

    # Reorder and keep only expected
    existing = [c for c in expected if c in Xp.columns]
    Xp = Xp.loc[:, existing]

    for c in expected:
        if c not in Xp.columns:
            Xp[c] = 0 if c in expected_num else "missing"

    # Cast numeric columns to the same type they had in training if possible
    for col in expected_num:
        if col in Xp.columns:
            Xp[col] = Xp[col].astype(np.float64)
    return Xp[expected]


# -------------------------
# Animated hero + starfield
# -------------------------
def render_animated_hero(height=400):
    html = f"""
    <div style="position:relative;width:100%;height:{height}px;overflow:hidden;border-radius:12px;margin-bottom:10px;">
      <canvas id="starfield" style="width:100%;height:100%;display:block;"></canvas>
      <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">
        <div style="width:200px;height:200px;border-radius:50%;
          background: radial-gradient(circle at 30% 25%, #ffd27a, #e06b00 35%, #8b2b00 70%);
          box-shadow: 0 0 60px rgba(255,215,0,0.3), inset -10px -10px 40px rgba(255,255,255,0.05);
          animation: spin 15s linear infinite;"></div>
      </div>
    </div>

    <style>
    @keyframes spin {{
      0% {{ transform: rotate(0deg); }}
      100% {{ transform: rotate(360deg); }}
    }}
    </style>

    <script>
    const canvas = document.getElementById('starfield');
    const ctx = canvas.getContext('2d');
    function resize() {{
        canvas.width = canvas.clientWidth*devicePixelRatio;
        canvas.height = canvas.clientHeight*devicePixelRatio;
        ctx.scale(devicePixelRatio, devicePixelRatio);
    }}
    resize(); window.addEventListener('resize', resize);

    const stars = [];
    for(let i=0;i<200;i++) {{
        stars.push({{
            x: Math.random()*canvas.clientWidth,
            y: Math.random()*canvas.clientHeight,
            r: Math.random()*1.5+0.5,
            a: Math.random(),
            blinkSpeed: 0.02 + Math.random()*0.03
        }});
    }}
    let t=0;
    function draw() {{
        ctx.clearRect(0,0,canvas.clientWidth, canvas.clientHeight);
        for(const s of stars){{
            ctx.beginPath();
            ctx.globalAlpha = 0.5 + 0.5*Math.sin(t*s.blinkSpeed*50 + s.a*10);
            ctx.fillStyle = 'white';
            ctx.arc(s.x,s.y,s.r,0,Math.PI*2);
            ctx.fill();
        }}
        ctx.globalAlpha = 1;
        t++;
        requestAnimationFrame(draw);
    }}
    draw();
    </script>
    """
    components.html(html, height=height+20)


# -------------------------
# Sidebar: model loader
# -------------------------
st.sidebar.markdown('<div class="sidebar-title">🚀 ExoClass Model</div>', unsafe_allow_html=True)
st.sidebar.write("Load a pretrained pipeline (.joblib) and label encoder, or retrain in app.")

uploaded_model = st.sidebar.file_uploader("Upload gb_pipeline.joblib", type=["joblib"])
uploaded_le = st.sidebar.file_uploader("Upload label_encoder.joblib", type=["joblib"])

gb_pipeline = joblib.load(uploaded_model) if uploaded_model else load_model("gb_pipeline.joblib")
le = joblib.load(uploaded_le) if uploaded_le else load_label_encoder("label_encoder.joblib")

if gb_pipeline is None or le is None:
    st.sidebar.warning("Model or label encoder not found. Use Retrain or upload model + encoder.")
else:
    st.sidebar.success("Model & LabelEncoder ready ✅")

st.sidebar.markdown("---")
st.sidebar.markdown("**Mission:** NASA Space Apps — Hunting for Exoplanets with AI")
st.sidebar.markdown("- Upload K2/Kepler/TESS CSVs\n- Retrain with custom FP weight\n- Inspect metrics & download models")
st.sidebar.markdown("---")
st.sidebar.markdown("🔗 **Tip:** Fill all features in manual input or upload CSV for batch prediction.")

# -------------------------
# Header + hero
# -------------------------
col_left, col_right = st.columns([3,1])
with col_left:
    st.title("🌌 CelestiaAI — Exoplanet Classifier")
    st.markdown("**NASA Space Apps 2025** — classify K2/Kepler/TESS entries as `CONFIRMED`, `CANDIDATE`, or `FALSE POSITIVE`.")
with col_right:
    render_animated_hero(height=200)

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "🔧 Retrain", "📊 Insights"])

# -------------------------
# PREDICT TAB — full dynamic input
# -------------------------
with tab1:
    st.header("Predict — Full-featured manual input & Batch CSV")
    st.markdown("Provide values for **all features expected by the model** or upload a CSV for batch prediction.")

    if gb_pipeline and le:
        try:
            num_cols, cat_cols = get_expected_columns_from_pipeline(gb_pipeline)
        except Exception as e:
            st.error(f"Could not extract expected features: {e}")
            num_cols, cat_cols = [], []

        with st.expander("🪐 Numeric Features", expanded=True):
         for col in num_cols:
             st.markdown(f"<div style='font-size:18px;font-weight:500;color:#e6f0ff'>{col}</div>", unsafe_allow_html=True)
             numeric_inputs[col] = st.number_input("", value=0.0, format="%.4f", key=f"num_{col}")

with st.expander("🛸 Categorical Features", expanded=False):
        for col in cat_cols:
            st.markdown(f"<div style='font-size:18px;font-weight:500;color:#e6f0ff'>{col}</div>", unsafe_allow_html=True)
            categorical_inputs[col] = st.text_input("", value="missing", key=f"cat_{col}")

        if st.button("Predict with current inputs"):
            try:
                df_input = pd.DataFrame([{**numeric_inputs, **categorical_inputs}])
                df_aligned = align_features(df_input, gb_pipeline)
                pred_idx = gb_pipeline.predict(df_aligned)[0]
                proba = gb_pipeline.predict_proba(df_aligned)[0]
                label = le.inverse_transform([pred_idx])[0]

                st.subheader("🚀 Prediction Result")
                st.metric("Predicted Class", label)

                st.subheader("📊 Prediction Probabilities")
                prob_df = pd.DataFrame(proba.reshape(1, -1), columns=le.classes_).T
                prob_df.columns = ["Probability"]
                st.dataframe(prob_df.style.format("{:.3f}"))

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    st.markdown("---")
    # Batch CSV upload
    st.subheader("Batch prediction (CSV upload)")
    upload_csv = st.file_uploader("Upload CSV (K2-style or minimal 3 cols)", type=["csv"], key="batch_predict")
    if upload_csv is not None:
        try:
            df_new = pd.read_csv(upload_csv, comment="#")
            df_new.columns = df_new.columns.str.strip()
            st.write("Preview of upload:")
            st.dataframe(df_new.head(6))

            X_new = df_new.drop(columns=["disposition"]) if "disposition" in df_new.columns else df_new.copy()
            X_new_aligned = align_features(X_new, gb_pipeline)
            preds = gb_pipeline.predict(X_new_aligned)
            probs = gb_pipeline.predict_proba(X_new_aligned)
            decoded = le.inverse_transform(preds)

            results = df_new.reset_index(drop=True).copy()
            results["prediction"] = decoded
            results["probability"] = probs.max(axis=1)
            st.subheader("Predictions (first rows)")
            st.dataframe(results.head(20))

            csv_out = results.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv_out, file_name="predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

# -------------------------
# RETRAIN TAB
# -------------------------
with tab2:
    st.header("Retrain model (advanced)")
    st.markdown("Upload a CSV containing `disposition` (CONFIRMED/CANDIDATE/FALSE POSITIVE).")
    retrain_file = st.file_uploader("Upload CSV for retraining", type=["csv"], key="retrain")
    fp_weight = st.slider("False positive weight multiplier", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
    n_estimators = st.slider("Iterations (max_iter)", min_value=100, max_value=2000, value=200, step=50)
    lr = st.slider("Learning rate", min_value=0.01, max_value=1.0, value=0.1, step=0.01)

    if retrain_file is not None and st.button("Start Retraining"):
        try:
            df_retrain = pd.read_csv(retrain_file, comment="#")
            df_retrain = df_retrain.iloc[98:] if df_retrain.shape[0]>200 else df_retrain
            df_retrain.columns = df_retrain.columns.str.strip()
            df_retrain = df_retrain[df_retrain['disposition'].isin(['CONFIRMED','CANDIDATE','FALSE POSITIVE'])]
            useless_cols = ['pl_name','hostname','disp_refname','pl_refname','st_refname','sy_refname','rastr','decstr','rowupdate','pl_pubdate','releasedate']
            df_retrain = df_retrain.drop(columns=[c for c in useless_cols if c in df_retrain.columns])

            Xr = df_retrain.drop(columns=['disposition'])
            yr_text = df_retrain['disposition']
            le_new = LabelEncoder().fit(yr_text)
            y = le_new.transform(yr_text)

            cat_cols = Xr.select_dtypes(include=['object']).columns.tolist()
            num_cols = Xr.select_dtypes(include=['int64','float64']).columns.tolist()

            preprocessor_new = ColumnTransformer(transformers=[
                ('num', SimpleImputer(strategy='constant', fill_value=0), num_cols),
                ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                  ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
            ])

            class_weights = {i: (fp_weight if c == "FALSE POSITIVE" else 1.0) for i,c in enumerate(le_new.classes_)}
            gb_new = Pipeline(steps=[
                ('preprocessor', preprocessor_new),
                ('classifier', HistGradientBoostingClassifier(learning_rate=float(lr), max_iter=int(n_estimators), class_weight=class_weights, random_state=42))
            ])

            Xtr, Xval, ytr, yval = train_test_split(Xr, y, test_size=0.2, stratify=y, random_state=42)
            gb_new.fit(Xtr, ytr)

            yval_pred = gb_new.predict(Xval)
            yval_score = gb_new.predict_proba(Xval) if hasattr(gb_new.named_steps['classifier'], 'predict_proba') else None

            acc = accuracy_score(yval, yval_pred)
            f1 = f1_score(yval, yval_pred, average='weighted')
            prec = precision_score(yval, yval_pred, average='weighted')
            rec = recall_score(yval, yval_pred, average='weighted')

            st.success("✅ Retraining complete")
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Accuracy", f"{acc:.3f}")
            mcol2.metric("F1 (weighted)", f"{f1:.3f}")
            mcol3.metric("Precision (weighted)", f"{prec:.3f}")
            mcol4.metric("Recall (weighted)", f"{rec:.3f}")

            st.subheader("Classification Report (validation)")
            st.text(classification_report(yval, yval_pred, target_names=le_new.classes_))

            st.subheader("Confusion Matrix (validation)")
            plot_confusion_matrix(confusion_matrix(yval, yval_pred), le_new.classes_)

            if yval_score is not None:
                st.subheader("ROC Curves (validation)")
                plot_roc_curve(yval, yval_score, le_new.classes_)

            joblib.dump(gb_new, "gb_pipeline_retrained.joblib")
            joblib.dump(le_new, "label_encoder_retrained.joblib")
            joblib.dump(gb_new, "gb_pipeline.joblib")
            joblib.dump(le_new, "label_encoder.joblib")
            st.success("Saved retrained model to disk and updated active model files")
            st.download_button("Download retrained model (.joblib)", open("gb_pipeline_retrained.joblib","rb"), "gb_pipeline_retrained.joblib")
            st.download_button("Download retrained label encoder (.joblib)", open("label_encoder_retrained.joblib","rb"), "label_encoder_retrained.joblib")

            gb_pipeline = gb_new
            le = le_new

        except Exception as e:
            st.error(f"Retraining failed: {e}")

# -------------------------
# INSIGHTS TAB
# -------------------------
with tab3:
    st.header("Model Insights")
    if gb_pipeline is None or le is None:
        st.info("Load or retrain a model to view insights.")
    else:
        try:
            clf = gb_pipeline.named_steps["classifier"]
            preproc = gb_pipeline.named_steps["preprocessor"]

            # Fetch numeric & categorical features
            try:
                num_cols, cat_cols = get_expected_columns_from_pipeline(gb_pipeline)
            except Exception:
                num_cols = st.text_input("Numeric feature names (comma-separated)", value="pl_orbper,pl_rade,pl_trandur").split(",")
                cat_cols = []

            # Prepare dummy data for permutation importance
            X_dummy = pd.DataFrame([{c: 0 if c in num_cols else "missing" for c in num_cols+cat_cols}])
            X_aligned = align_features(X_dummy, gb_pipeline)
            
            # Compute permutation importance (faster, works for any model)
            r = permutation_importance(gb_pipeline, X_aligned, np.array([0]), n_repeats=10, random_state=42, scoring='accuracy')
            feature_names = num_cols + cat_cols
            importances = r.importances_mean

            plot_feature_importance(importances, feature_names)

        except Exception as e:
            st.warning("Could not extract feature importances: " + str(e))

    st.markdown("---")
    st.write("🌍 Built for NASA Space Apps Challenge 2025 — explore exoplanets with AI 🚀")

