import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD PIPELINE
# =========================
pipeline = joblib.load("airlines_final_pipeline.joblib")

# =========================
# EXTRACT CATEGORY VALUES
# =========================
preprocessor = pipeline.named_steps["preprocess"]
cat_encoder = (
    preprocessor
    .named_transformers_["cat"]
    .named_steps["onehot"]
)

CATEGORICAL_FEATURES = preprocessor.transformers_[1][2]
CATEGORIES = dict(zip(CATEGORICAL_FEATURES, cat_encoder.categories_))

AIRLINES = sorted(CATEGORIES["Airline"])
ROUTES = sorted(CATEGORIES["Route"])
DEPARTURE_PERIODS = sorted(CATEGORIES["Departure_period"])

DAY_OF_WEEK = {
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 7
}

st.title("✈️ Airline Delay Prediction")

st.subheader("Masukkan detail penerbangan")

airline = st.selectbox("Airline", AIRLINES)
route = st.selectbox("Rute", Rute)
day_label = st.selectbox("Day of Week", DAY_OF_WEEK.keys())
departure_period = st.selectbox("Departure Period", DEPARTURE_PERIODS)

if st.button("Predict Delay"):
    input_df = pd.DataFrame([{
        "Airline": airline,
        "Route": route,
        "DayOfWeek": DAY_OF_WEEK[day_label],
        "Departure_period": departure_period,

        # DEFAULT / AUTO FILL (user tidak perlu isi)
        "Flight": 0,
        "Time": 0,
        "Length": 120,
        "Distance_km": 500,
        "Arrival_Time": 0,
        "is_weekend": int(DAY_OF_WEEK[day_label] >= 6),
        "Arrival_period": "Afternoon"
    }])

    proba = pipeline.predict_proba(input_df)[0][1]

    st.success(f"🕒 Probability of Delay: **{proba:.2%}**")
