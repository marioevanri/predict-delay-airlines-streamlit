import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL & METADATA
# =========================
@st.cache_resource
def load_pipeline():
    return joblib.load("airlines_final_pipeline.joblib")

@st.cache_resource
def load_metadata():
    return joblib.load("model_metadata.joblib")

pipeline = load_pipeline()
meta = load_metadata()

# =========================
# GET CATEGORIES SAFELY
# =========================
CATS = meta["categorical_features"]

AIRLINES = sorted(pipeline.named_steps["preprocess"]
                  .named_transformers_["cat"]
                  .named_steps["onehot"]
                  .categories_[CATS.index("Airline")])

ROUTES = sorted(pipeline.named_steps["preprocess"]
                .named_transformers_["cat"]
                .named_steps["onehot"]
                .categories_[CATS.index("Rute")])

# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Airline Delay Prediction", layout="centered")

st.title("✈️ Airline Delay Prediction")
st.markdown("Masukkan detail penerbangan:")

airline = st.selectbox("Airline", AIRLINES)
route = st.selectbox("Rute", ROUTES)

day_map = {
    "Weekday": 1,
    "Weekend": 6
}
day_label = st.selectbox("Day of Week", list(day_map.keys()))
dayofweek = day_map[day_label]

departure_period = st.selectbox(
    "Departure Period",
    ["Morning", "Afternoon", "Evening", "Night"]
)

# =========================
# AUTO-FILL FEATURES
# =========================
is_weekend = 1 if dayofweek >= 6 else 0

input_df = pd.DataFrame([{
    "Flight": 1,
    "Time": 1,
    "Length": 1,
    "Distance_km": 500,
    "Arrival_Time": 1,

    "Airline": airline,
    "Rute": route,
    "DayOfWeek": dayofweek,
    "Departure_period": departure_period,
    "is_weekend": is_weekend,
    "Arrival_period": "Afternoon"
}])

# =========================
# PREDICTION
# =========================
if st.button("Predict Delay"):
    proba = pipeline.predict_proba(input_df)[0][1]
    threshold = float(meta["threshold"])

    st.subheader("📊 Prediction Result")
    st.write(f"**Delay Probability:** `{proba:.2%}`")
    st.write(f"**Decision Threshold:** `{threshold:.2f}`")

    if proba >= threshold:
        st.error("⏱️ Flight is likely to be **DELAYED**")
    else:
        st.success("✅ Flight is likely **ON TIME**")
