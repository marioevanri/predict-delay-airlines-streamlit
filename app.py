import streamlit as st
import pandas as pd
import joblib

# =====================
# LOAD MODEL
# =====================
@st.cache_resource
def load_assets():
    pipeline = joblib.load("airlines_final_pipeline.joblib")
    metadata = joblib.load("model_metadata.joblib")
    return pipeline, metadata

pipeline, metadata = load_assets()
THRESHOLD = float(metadata["threshold"])


# =====================
# BUILD MODEL INPUT
# =====================
def build_model_input(user_input: dict) -> pd.DataFrame:
    df = pd.DataFrame([user_input])

    # Numeric defaults (tidak diinput user)
    df["Flight"] = 1
    df["Length"] = 500
    df["Distance_km"] = df["Length"]
    df["Arrival_Time"] = df["Time"]

    # Derived features
    df["is_weekend"] = df["DayOfWeek"].isin(
        ["Saturday", "Sunday"]
    ).astype(int)

    df["Departure_period"] = "Unknown"

    ordered_cols = [
        'Flight',
        'Time',
        'Length',
        'Distance_km',
        'Arrival_Time',
        'Airline',
        'Rute',
        'DayOfWeek',
        'Departure_period',
        'is_weekend',
        'Arrival_period'
    ]

    return df[ordered_cols]


# =====================
# STREAMLIT UI
# =====================
st.set_page_config(page_title="Airline Delay Prediction", layout="centered")

st.title("✈️ Airline Delay Prediction")
st.write("Masukkan data penerbangan")

# =====================
# USER INPUT
# =====================
airline = st.text_input("Maskapai (contoh: AA, B6, WN)")
route = st.text_input("Rute (contoh: JFK-LAX)")

day_of_week = st.selectbox(
    "Hari Keberangkatan",
    ["Senin", "Selasa", "Rabu", "Kamis",
     "Jumat", "Sabtu", "Minggu"]
)

time = st.slider("Jam Keberangkatan", 0, 23, 8)

arrival_period = st.selectbox(
    "Periode Kedatangan",
    ["Pagi", "Siang", "Petang", "Malam"]
)

# =====================
# PREDICTION
# =====================
if st.button("Predict Delay"):
    user_input = {
        "Airline": airline,
        "Rute": route,
        "DayOfWeek": day_of_week,
        "Time": time,
        "Arrival_period": arrival_period
    }

    input_df = build_model_input(user_input)

    proba = pipeline.predict_proba(input_df)[0][1]
    pred = int(proba >= THRESHOLD)

    st.markdown("---")
    st.subheader("Prediction Result")

    if pred == 1:
        st.error(f"⚠️ Flight likely delayed (Probability: {proba:.2%})")
    else:
        st.success(f"✅ Flight likely on time (Probability: {(1 - proba):.2%})")
