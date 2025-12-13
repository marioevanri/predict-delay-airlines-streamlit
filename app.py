import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ===============================
# LOAD MODEL & METADATA
# ===============================
@st.cache_resource
def load_artifacts():
    pipeline = joblib.load("airlines_final_pipeline.joblib")
    meta = joblib.load("model_metadata.joblib")
    return pipeline, meta

pipeline, meta = load_artifacts()
threshold = float(meta.get("threshold", 0.5))

# ===============================
# AMBIL KATEGORI DARI PIPELINE
# ===============================
preprocessor = pipeline.named_steps["preprocess"]
cat_pipeline = preprocessor.named_transformers_["cat"]
onehot = cat_pipeline.named_steps["onehot"]

cat_features = preprocessor.transformers_[1][2]  # kolom kategorikal
cat_categories = dict(zip(cat_features, onehot.categories_))

AIRLINES = sorted(cat_categories["Airline"])
ROUTES = sorted(cat_categories["Rute"])

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Airline Delay Prediction", layout="centered")

st.title("✈️ Airline Delay Prediction")
st.write("Masukkan detail penerbangan:")

# -------- INPUT USER ----------
airline = st.selectbox("Airline", AIRLINES)
route = st.selectbox("Route", ROUTES)

day_map = {
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
    "Sunday": 7,
}
day_label = st.selectbox("Day of Week", list(day_map.keys()))
dayofweek = day_map[day_label]

departure_period = st.selectbox(
    "Departure Period",
    ["Morning", "Afternoon", "Evening", "Night"]
)

# ===============================
# PREDICTION
# ===============================
if st.button("Predict Delay"):
    # Buat input lengkap sesuai pipeline
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

    # Prediksi
    proba = pipeline.predict_proba(input_df)[0][1]
    pred = int(proba >= threshold)

    # Output
    st.markdown("---")
    if pred == 1:
        st.error(f"⏱️ **Delay Predicted**\n\nProbability: **{proba:.2%}**")
    else:
        st.success(f"✅ **On Time**\n\nProbability: **{1 - proba:.2%}**")
