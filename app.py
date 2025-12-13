import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL & CATEGORIES
# =========================
@st.cache_resource
def load_pipeline():
    return joblib.load("airlines_final_pipeline.joblib")

@st.cache_resource
def load_categories():
    return joblib.load("categories.joblib")

pipeline = load_pipeline()
CATEGORIES = load_categories()

# =========================
# DROPDOWN OPTIONS
# =========================
AIRLINES = CATEGORIES["Airline"]
ROUTES = CATEGORIES["Rute"]
DEPARTURE_PERIODS = CATEGORIES["Departure_period"]

# =========================
# UI
# =========================
st.title("✈️ Airline Delay Prediction")

airline = st.selectbox("Airline", AIRLINES)
route = st.selectbox("Route", ROUTES)

day_map = {
    "Monday": 1, "Tuesday": 2, "Wednesday": 3,
    "Thursday": 4, "Friday": 5, "Saturday": 6, "Sunday": 7
}
day_label = st.selectbox("Day of Week", list(day_map.keys()))
dayofweek = day_map[day_label]

departure_period = st.selectbox("Departure Period", DEPARTURE_PERIODS)

# =========================
# PREDICT
# =========================
if st.button("Predict Delay"):
    input_df = pd.DataFrame([{
        "Flight": 0,
        "Time": 0,
        "Length": 0,
        "Distance_km": 0,
        "Arrival_Time": 0,
        "Airline": airline,
        "Rute": route,
        "DayOfWeek": dayofweek,
        "Departure_period": departure_period,
        "is_weekend": int(dayofweek >= 6),
        "Arrival_period": "Unknown"
    }])

    proba = pipeline.predict_proba(input_df)[0][1]

    if proba >= 0.5:
        st.error(f"⏱️ Delay Predicted ({proba:.2%})")
    else:
        st.success(f"✅ On Time ({1 - proba:.2%})")
