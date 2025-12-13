import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Airline Delay Prediction", layout="centered")
st.title("✈️ Airline Delay Prediction")

# =========================
# LOAD PIPELINE & CATEGORIES
# =========================
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("airlines_final_pipeline.joblib")
    categories = joblib.load("categories.joblib")
    return pipeline, categories

pipeline, CATEGORIES = load_artifacts()

# 🔒 PAKSA jadi list of STRING
AIRLINES = [str(x) for x in CATEGORIES["Airline"]]
RUTES = [str(x) for x in CATEGORIES["Rute"]]
DEPARTURE_PERIODS = [str(x) for x in CATEGORIES["Departure_period"]]

# =========================
# UI INPUT (STRING ONLY)
# =========================
airline = st.selectbox(
    "Airline",
    options=AIRLINES,
    index=0
)

rute = st.selectbox(
    "Rute",
    options=RUTES,
    index=0
)

day_map = {
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 7
}

day_label = st.selectbox(
    "Day of Week",
    options=list(day_map.keys())
)

departure_period = st.selectbox(
    "Departure Period",
    options=DEPARTURE_PERIODS
)

# =========================
# PREDICT
# =========================
if st.button("Predict Delay"):

    dayofweek = day_map[day_label]

    input_df = pd.DataFrame([{
        # numeric (default aman)
        "Flight": 0,
        "Time": 0,
        "Length": 0,
        "Distance_km": 0,
        "Arrival_Time": 0,

        # categorical (STRING ASLI)
        "Airline": airline,
        "Rute": rute,
        "DayOfWeek": dayofweek,
        "Departure_period": departure_period,
        "is_weekend": int(dayofweek >= 6),
        "Arrival_period": "Unknown"
    }])

    proba = pipeline.predict_proba(input_df)[0][1]

    st.markdown("---")
    st.write("### Prediction Result")

    if proba >= 0.32:
        st.error(f"⏱️ **DELAYED** — Probability: {proba:.2%}")
    else:
        st.success(f"✅ **ON TIME** — Probability: {1 - proba:.2%}")
