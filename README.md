# ✈️ Airline Delay Prediction – Streamlit App

A web application to predict airline flight delays using a machine learning model.

## 🚀 Model Overview
- Algorithm: XGBoost (best tuned model)
- Evaluation metric: F1-score (threshold optimized)
- Pipeline includes:
  - Missing value handling
  - Scaling
  - Categorical encoding
  - Final trained model

## 📦 Artifacts
- `airlines_final_pipeline.joblib` → Full preprocessing + model pipeline
- `model_metadata.joblib` → Threshold & feature metadata

## 🖥️ How the App Works
1. User inputs flight information
2. Data is processed using the trained pipeline
3. Model predicts delay probability
4. Final decision based on optimized threshold

## 🧪 Features Used
- Airline
- Route
- Departure period
- Arrival period
- Flight number
- Day of week
- Time
- Flight length
- Distance (km)
- Arrival time

## ⚙️ Tech Stack
- Python
- Streamlit
- Scikit-learn
- XGBoost
- Joblib

## ▶️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
