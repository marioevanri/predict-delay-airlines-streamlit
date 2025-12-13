import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL & METADATA
# =========================
@st.cache_resource
def load_assets():
    pipeline = joblib.load("airlines_final_pipeline.joblib")
    metadata = joblib.load("model_metadata.joblib")
    return pipeline, metadata

pipeline, metadata = load_assets()
THRESHOLD = metadata["threshold"]

# =========================
# UI
# =========================
st.set_page_config(page_title="Airline Delay Prediction", layout="centered")
st.title("✈️ Airline Delay Prediction")
st.write("Predict whether a flight will be delayed")

# =========================
# INPUT FORM
# =========================
with st.form("prediction_form"):
    Airline = st.text_input("Airline", "AA")
    Rute = st.text_input("Route", "JFK-LAX")
    Departure_period = st.selectbox(
        "Departure Period", ["Morning", "Afternoon", "Evening", "Night"]
    )
    Arrival_period = st.selectbox(
        "Arrival Period", ["Morning", "Afternoon", "Evening", "Night"]
    )

    Flight = st.number_input("Flight Number", value=100)
    DayOfWeek = st.slider("Day of Week (1=Mon, 7=Sun)", 1, 7, 1)
    Time = st.number_input("Departure Time", value=1200)
    Length = st.number_input("Flight Duration (minutes)", value=120)
    Distance_km = st.number_input("Distance (km)", value=500)
    Arrival_Time = st.number_input("Arrival Time", value=1400)

    submitted = st.form_submit_button("Predict")

# =========================
# PREDICTION
# =========================
if submitted:
    input_df = pd.DataFrame([{
        "Airline": Airline,
        "Rute": Rute,
        "Departure_period": Departure_period,
        "Arrival_period": Arrival_period,
        "Flight": Flight,
        "DayOfWeek": DayOfWeek,
        "Time": Time,
        "Length": Length,
        "Distance_km": Distance_km,
        "Arrival_Time": Arrival_Time
    }])

    proba = pipeline.predict_proba(input_df)[0][1]
    pred = int(proba >= THRESHOLD)

    st.subheader("📊 Prediction Result")
    st.write(f"Delay Probability: **{proba:.2%}**")

    if pred == 1:
        st.error("⏱️ Flight is likely to be DELAYED")
    else:
        st.success("✅ Flight is likely ON TIME")
