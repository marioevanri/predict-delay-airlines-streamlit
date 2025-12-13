import streamlit as st
import pandas as pd
import joblib
import numpy as np

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="Airline Delay Prediction",
    layout="centered"
)

st.title("✈️ Airline Delay Prediction")

# =========================
# LOAD MODEL & METADATA
# =========================
@st.cache_resource
def load_pipeline():
    pipeline = joblib.load("airlines_final_pipeline.joblib")
    metadata = joblib.load("model_metadata.joblib")
    return pipeline, metadata

pipeline, metadata = load_pipeline()

THRESHOLD = float(metadata.get("threshold", 0.5))

# =========================
# GET CATEGORY VALUES FROM PIPELINE
# =========================
preprocess = pipeline.named_steps["preprocess"]
cat_pipe = preprocess.named_transformers_["cat"]
ohe = cat_pipe.named_steps["onehot"]

CATEGORIES = dict(
    zip(
        metadata["categorical_features"],
        ohe.categories_
    )
)

# =========================
# UI INPUT
# =========================
st.subheader("Masukkan detail penerbangan")

airline = st.selectbox(
    "Airline",
    sorted(CATEGORIES["Airline"])
)

route = st.selectbox(
    "Route",
    sorted(CATEGORIES["Rute"])
)

day_of_week = st.selectbox(
    "Day of Week",
    ["Weekday", "Weekend"]
)

departure_period = st.selectbox(
    "Departure Period",
    sorted(CATEGORIES["Departure_period"])
)

# =========================
# FEATURE ENGINEERING (USER → MODEL)
# =========================
if st.button("Predict Delay"):
    with st.spinner("Predicting..."):

        # Map weekday
        day_map = {"Weekday": 1, "Weekend": 7}
        is_weekend = 1 if day_of_week == "Weekend" else 0

        # =========================
        # BUILD INPUT DATAFRAME
        # =========================
        input_df = pd.DataFrame([{
            # ===== NUMERIC (DEFAULT SAFE) =====
            "Flight": 1,
            "Time": 0,
            "Length": 0,
            "Distance_km": 0,
            "Arrival_Time": 0,

            # ===== CATEGORICAL =====
            "Airline": airline,
            "Rute": route,
            "DayOfWeek": day_map[day_of_week],
            "Departure_period": departure_period,
            "is_weekend": is_weekend,
            "Arrival_period": "Unknown"
        }])

        # =========================
        # PREDICTION
        # =========================
        proba = pipeline.predict_proba(input_df)[0][1]
        pred = int(proba >= THRESHOLD)

        # =========================
        # OUTPUT
        # =========================
        st.subheader("Hasil Prediksi")

        if pred == 1:
            st.error(f"⏱️ **Delay Predicted** (probability = {proba:.2%})")
        else:
            st.success(f"✅ **On Time** (probability = {1 - proba:.2%})")

        st.caption(f"Decision threshold = {THRESHOLD:.2f}")
