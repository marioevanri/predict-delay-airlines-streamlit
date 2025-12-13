import streamlit as st
import pandas as pd
import joblib

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Airline Delay Prediction",
    layout="centered"
)

# ===============================
# LOAD MODEL & METADATA
# ===============================
@st.cache_resource
def load_pipeline():
    return joblib.load("airlines_final_pipeline.joblib")

@st.cache_data
def load_metadata():
    return joblib.load("model_metadata.joblib")

pipeline = load_pipeline()
metadata = load_metadata()

CATEGORIES = metadata.get("categorical_features", {})

# ===============================
# SAFE CATEGORY LOAD
# ===============================
AIRLINES = sorted(CATEGORIES.get("Airline", []))
RUTES = sorted(CATEGORIES.get("Rute", []))   # ⬅️ PENTING: Rute (bukan Route)

# ===============================
# UI
# ===============================
st.title("✈️ Airline Delay Prediction")
st.markdown("Masukkan detail penerbangan di bawah ini")

# ===============================
# USER INPUT
# ===============================
airline = st.selectbox(
    "Airline",
    AIRLINES if AIRLINES else ["UNKNOWN"]
)

route = st.selectbox(
    "Rute",
    RUTES if RUTES else ["UNKNOWN"]
)

day_of_week = st.selectbox(
    "Day of Week",
    {
        "Weekday": 1,
        "Weekend": 7
    }.items(),
    format_func=lambda x: f"{x[0]} ({x[1]})"
)[1]

departure_period = st.selectbox(
    "Departure Period",
    ["Morning", "Afternoon", "Evening", "Night"]
)

# ===============================
# PREDICT
# ===============================
if st.button("🚀 Predict Delay"):

    # Minimal user input
    input_df = pd.DataFrame([{
        "Airline": airline,
        "Rute": route,
        "DayOfWeek": day_of_week,
        "Departure_period": departure_period
    }])

    # ===============================
    # AUTO-FILL MISSING FEATURES
    # ===============================
    required_features = pipeline.feature_names_in_

    input_df = input_df.reindex(
        columns=required_features,
        fill_value=0
    )

    # ===============================
    # PREDICTION
    # ===============================
    proba = pipeline.predict_proba(input_df)[0][1]

    label = "DELAYED ❌" if proba >= 0.5 else "ON TIME ✅"

    # ===============================
    # OUTPUT
    # ===============================
    st.subheader("📊 Prediction Result")
    st.metric(
        label="Delay Probability",
        value=f"{proba:.2%}"
    )
    st.success(label)
