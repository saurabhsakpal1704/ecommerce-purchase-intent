from pathlib import Path
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Purchase Intent Demo", page_icon="🛒", layout="centered")
st.title("🛒 E-Commerce Purchase Intent")
st.caption("Synthetic browsing-session demo with interpretable model feature importance.")
model_path = Path("models/purchase_intent_pipeline.joblib")
if not model_path.exists():
    st.warning("Run `python generate_data.py` and `python train.py` first.")
    st.stop()
model = joblib.load(model_path)
with st.form("session"):
    a,b = st.columns(2)
    pages = a.slider("Page views",1,40,8)
    duration = b.slider("Session duration (minutes)",0.2,60.0,8.0)
    products = a.slider("Product views",0,25,4)
    carts = b.slider("Cart additions",0,8,1)
    returning = a.selectbox("Returning visitor",["No","Yes"])
    device = b.selectbox("Device",["mobile","desktop","tablet"])
    source = a.selectbox("Traffic source",["search","social","direct","email","referral"])
    go = st.form_submit_button("Estimate demo purchase intent")
if go:
    row = pd.DataFrame([{"page_views":pages,"session_duration_min":duration,
        "product_views":products,"cart_additions":carts,"returning_visitor":int(returning=="Yes"),
        "device_type":device,"traffic_source":source}])
    p = model.predict_proba(row)[0,1]
    st.metric("Model-estimated purchase probability (synthetic demo)",f"{p:.1%}")
    st.info("This is an educational model output, not a reliable prediction about a real customer.")
importance_path = Path("models/permutation_importance.csv")
if importance_path.exists():
    imp = pd.read_csv(importance_path)
    st.subheader("Permutation feature importance")
    st.plotly_chart(px.bar(imp, x="importance", y="feature", orientation="h",
        title="Change in ROC-AUC when a feature is shuffled"), use_container_width=True)
    st.caption("Importance describes this trained demo model; it does not establish causation.")
