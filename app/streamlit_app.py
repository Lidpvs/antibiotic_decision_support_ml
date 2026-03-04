# app/streamlit_app.py

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


from src.data_prep import load_data, prepare_long_df
from src.model import train_logreg
from src.recommender import recommend, PenaltyConfig, DEFAULT_PENALTY

st.set_page_config(page_title="Antibiotic Decision Support (ML)", layout="wide")
st.title("Antibiotic Decision Support (ML)")
st.caption("Logistic Regression + Rule-based penalty for reserve antibiotics")
st.sidebar.header("Settings")

data_path = st.sidebar.text_input("Path to CSV", value="data/raw/antibiotics.csv")
top_k = st.sidebar.slider("Top-K recommendations", 3, 15, 10)

reserve_penalty = st.sidebar.slider("Reserve penalty (subtract from score)", 0.0, 0.5, 0.15, 0.01)

reserve_text = st.sidebar.text_area(
    "Reserve antibiotics (one per line)",
    value="colistine\nIPM"
)

reserve_list = [x.strip() for x in reserve_text.splitlines() if x.strip()]

penalty_cfg = PenaltyConfig(
    reserve_groups={"reserve": reserve_list},
    penalties={"reserve": float(reserve_penalty)}
)

st.sidebar.divider()
train_now = st.sidebar.button("Tran / Re-Train model")

@st.cache_data
def _prepare_long(path: str) -> pd.DataFrame:
    return prepare_long_df(path)

@st.cache_resource
def _train_model(long_df: pd.DataFrame):
    model, metrics = train_logreg(long_df)
    return model, metrics

try:
    long_df = _prepare_long(data_path)

    if train_now:
        _train_model.clear()

    model, metrics = _train_model(long_df)

except Exception as e:
    st.error(f"ERROR: {e}")
    st.stop()

st.subheader("Model metrics")
col1, col2 = st.columns(2)

with col1:
    st.write("ROC-AUC:", metrics.get("roc_auc"))
with col2:
    st.write("Samples:", metrics.get("n_samples"))

bacteria_list = sorted(long_df["bacteria"].dropna().unique().tolist())
bacteria = st.selectbox("Select bacteria", options=bacteria_list)

if st.button("Recommend antibiotics"):
    rec = recommend(
        model, long_df,
        bacteria=bacteria, top_k=top_k,
        penalty_config=penalty_cfg
        )
    st.subheader("Recommendations")
    st.dataframe(rec, use_container_width=True)

    st.caption("score = mean P(susceptible) from model; reserve penalty substracts from score")