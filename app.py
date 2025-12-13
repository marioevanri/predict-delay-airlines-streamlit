import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD PIPELINE
# =========================
@st.cache_resource
def load_pipeline():
    return joblib.load("airlines_final_pipeline.joblib")

pipeline = load_pipeline()

# =========================
# FEATURE BUILDER
# =========================
def build_input_df(
    Airline,
    Route,
    DayOfWeek,
    Departure_period
):
    # Derived features
    is_weekend = 1 if DayOfWeek in [6, 7] else 0

    # Default / safe values (median-like)
    Flight = 1
    Time = 12
    Length = 120
    Distance_km = 800
    Arrival_Time = 14

    arrival_period_map = {
        "Morning": "Afternoon",
        "Afternoon": "Evening",
        "Evening": "Night",
        "Night": "Morning"
    }
    Arrival_period = arrival_period_map.get(Departure_period, "Afternoon")

    return pd.DataFrame([{
        "Flight": Flight,
        "Time": Time,
        "Length": Length,
        "Distance_km": Distance_km,
        "Arrival_Time": Arrival_Time,
        "Airline": Airline,
        "Route": Route,
        "DayOfWeek": DayOfWeek,
        "Departure_period": Departure_period,
        "is_weekend": is_weekend,
        "Arrival_period": Arrival_period
    }])

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Airline Delay Prediction", layout="centered")
st.title("✈️ Airline Delay Prediction")

st.write("Masukkan data penerbangan sederhana di bawah ini:")

Airline = st.text_input("Airline (contoh: AA, DL, UA)")
Route = st.text_input("Route (contoh: JFK-LAX)")
DayOfWeek = st.selectbox(
    "Day Of Week",
    options=[1, 2, 3, 4, 5, 6, 7],
    format_func=lambda x: f"{x} ({'Weekend' if x in [6,7] else 'Weekday'})"
)
Departure_period = st.selectbox(
    "Departure Period",
    ["Morning", "Afternoon", "Evening", "Night"]
)

# =========================
# PREDICT
# =========================
if st.button("Predict Delay"):
    input_df = build_input_df(
        Airline=Airline,
        Route=Route,
        DayOfWeek=DayOfWeek,
        Departure_period=Departure_period
    )

    proba = pipeline.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")
    st.metric("Delay Probability", f"{proba*100:.1f}%")

    if proba >= 0.5:
        st.error("⚠️ Flight is likely to be delayed")
    else:
        st.success("✅ Flight is likely to be on time")

