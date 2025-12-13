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
# LOAD CATEGORY OPTIONS
# =========================
@st.cache_data
def load_categories():
    # ambil kategori dari encoder di pipeline
    ohe = pipeline.named_steps["preprocess"] \
                  .named_transformers_["cat"] \
                  .named_steps["onehot"]

    feature_names = ohe.get_feature_names_out()

    airlines = sorted(
        set(f.split("_", 1)[1] for f in feature_names if f.startswith("Airline_"))
    )
    routes = sorted(
        set(f.split("_", 1)[1] for f in feature_names if f.startswith("Route_"))
    )

    return airlines, routes

AIRLINES, ROUTES = load_categories()

# =========================
# FEATURE BUILDER (FULL)
# =========================
def build_input_df(Airline, Route, DayOfWeek, Departure_period):
    is_weekend = 1 if DayOfWeek in [6, 7] else 0

    # default numeric (aman & konsisten)
    data = {
        "Flight": 1,
        "Time": 12,
        "Length": 120,
        "Distance_km": 800,
        "Arrival_Time": 14,
        "Airline": Airline,
        "Route": Route,
        "DayOfWeek": DayOfWeek,
        "Departure_period": Departure_period,
        "is_weekend": is_weekend,
        "Arrival_period": {
            "Morning": "Afternoon",
            "Afternoon": "Evening",
            "Evening": "Night",
            "Night": "Morning"
        }[Departure_period]
    }

    return pd.DataFrame([data])

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Airline Delay Prediction", layout="centered")
st.title("✈️ Airline Delay Prediction")

st.markdown("### Masukkan detail penerbangan")

Airline = st.selectbox("Airline", AIRLINES)
Route = st.selectbox("Route", ROUTES)

DayOfWeek = st.selectbox(
    "Day of Week",
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
        Airline, Route, DayOfWeek, Departure_period
    )

    proba = pipeline.predict_proba(input_df)[0][1]

    st.subheader("📊 Prediction Result")
    st.metric("Delay Probability", f"{proba*100:.1f}%")

    if proba >= 0.5:
        st.error("⚠️ Flight is likely to be delayed")
    else:
        st.success("✅ Flight is likely to be on time")
