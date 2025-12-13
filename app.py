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

st.title("✈️ Airline Delay Prediction")
st.write("Masukkan data penerbangan (Airline & Rute dalam bentuk teks)")

# ===============================
# LOAD MODEL & METADATA
# ===============================
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("airlines_final_pipeline.joblib")
    meta = joblib.load("model_metadata.joblib")
    return pipeline, meta

pipeline, meta = load_artifacts()
THRESHOLD = float(meta.get("threshold", 0.5))

# ===============================
# USER INPUT (STRING ONLY)
# ===============================
airline = st.text_input(
    "Airline (contoh: AA, WN, F9)",
    value=""
)

rute = st.text_input(
    "Rute (contoh: JFK-LAX, ANC-ORD)",
    value=""
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
    list(day_map.keys())
)
dayofweek = day_map[day_label]

departure_period = st.selectbox(
    "Departure Period",
    ["Morning", "Afternoon", "Evening", "Night"]
)

# ===============================
# PREDICTION
# ===============================
if st.button("Predict Delay"):

    # VALIDASI SEDERHANA
    if airline.strip() == "" or rute.strip() == "":
        st.warning("⚠️ Airline dan Rute tidak boleh kosong")
    else:
        input_df = pd.DataFrame([{
            # numeric (default aman)
            "Flight": 0,
            "Time": 0,
            "Length": 0,
            "Distance_km": 0,
            "Arrival_Time": 0,

            # categorical (STRING ASLI)
            "Airline": airline.strip(),
            "Rute": rute.strip(),
            "DayOfWeek": dayofweek,
            "Departure_period": departure_period,
            "is_weekend": int(dayofweek >= 6),
            "Arrival_period": "Unknown"
        }])

        proba = pipeline.predict_proba(input_df)[0][1]

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        if proba >= THRESHOLD:
            st.error(f"⏱️ **DELAYED** — Probability: {proba:.2%}")
        else:
            st.success(f"✅ **ON TIME** — Probability: {1 - proba:.2%}")

        st.caption(f"Decision threshold = {THRESHOLD:.2f}")
