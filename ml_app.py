# ml_app.py (revisi)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import xgboost as xgb
import os
from sklearn.preprocessing import LabelEncoder

# -------------------------
# Helper: load encoders if saved, else build from CSV (fallback)
# -------------------------
@st.cache_data(show_spinner=False)
def _load_training_metadata(encoders_path_local="encoders.joblib"):
    possible_paths = [
        encoders_path_local,
        os.path.join("models", os.path.basename(encoders_path_local)),
        os.path.join(".", encoders_path_local),
    ]

    loaded = None
    used_path = None
    for p in possible_paths:
        if os.path.exists(p):
            loaded = joblib.load(p)
            used_path = p
            break

    if loaded is None:
        raise FileNotFoundError(
            f"encoders.joblib not found. Place `encoders.joblib` in the project root or models/ folder."
        )

    # Normalize loaded object
    if isinstance(loaded, dict) and 'encoders' in loaded:
        encoders = loaded.get('encoders', {})
        feature_columns = loaded.get('feature_columns')
        raw_unique = loaded.get('raw_unique', {})
    else:
        encoders = loaded
        feature_columns = None
        raw_unique = {}

    # Build raw_unique from LabelEncoder classes if not provided
    for col, le in encoders.items():
        try:
            raw_unique.setdefault(col, list(le.classes_))
        except Exception:
            raw_unique.setdefault(col, [])

    # Derive a conservative default feature list if not included in artifact
    if feature_columns is None:
        numeric_engineered = [
            'Length', 'Time', 'DayOfWeek', 'Departure_period',
            'Holiday', 'Arrival_Time', 'Arrival_period'
        ]
        feature_columns = list(encoders.keys()) + [c for c in numeric_engineered if c not in encoders]

    return None, encoders, feature_columns, raw_unique


# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource
def _load_model_cached(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = joblib.load(model_path)

    warn_msgs = [str(wi.message) for wi in w]
    compat_warning = any('If you are loading a serialized model' in m for m in warn_msgs)
    if compat_warning:
        try:
            if hasattr(model, 'get_booster'):
                booster = model.get_booster()
                converted = model_path.replace('.pkl', '_converted.json')
                booster.save_model(converted)
                new_model = xgb.XGBClassifier()
                new_model.load_model(converted)
                model = new_model
        except Exception:
            pass

    return model


# -------------------------
# Preprocess single input (same as before)
# -------------------------
def _preprocess_input(input_data: dict, encoders: dict, feature_columns: list):
    """Given a single input dict, apply same feature engineering and encoding.
    Returns a DataFrame with columns ordered to feature_columns.
    """
    row = {}
    row['Airline'] = input_data['Airline']
    row['AirportFrom'] = input_data['AirportFrom']
    row['AirportTo'] = input_data['AirportTo']
    row['Length'] = input_data['Length_min'] / 60.0
    row['Time'] = input_data['Time_hour']
    row['DayOfWeek'] = input_data['DayOfWeek']

    df_row = pd.DataFrame([row])
    if 'Rute_override' in input_data:
        df_row['Rute'] = input_data['Rute_override']
    else:
        df_row['Rute'] = df_row['AirportFrom'].astype(str) + '-' + df_row['AirportTo'].astype(str)

    def get_period(t):
        if (t >= 5) & (t < 12):
            return 0
        elif (t >= 12) & (t < 17):
            return 1
        elif (t >= 17) & (t < 21):
            return 2
        else:
            return 3

    df_row['Departure_period'] = df_row['Time'].apply(get_period)
    df_row['Holiday'] = df_row['DayOfWeek'].apply(lambda x: 1 if (x == 6 or x == 7) else 0)

    at = df_row['Time'].iloc[0] + df_row['Length'].iloc[0]
    if at >= 24:
        at = at - 24
    df_row['Arrival_Time'] = at
    df_row['Arrival_period'] = get_period(at)

    # apply encoders: if unseen -> -1
    for col, le in encoders.items():
        if col in df_row.columns:
            val = df_row.loc[0, col]
            try:
                df_row.loc[0, col] = le.transform([str(val)])[0]
            except ValueError:
                df_row.loc[0, col] = -1

    # Do NOT enforce final feature ordering here. Return a single-row DF
    # with the engineered/encoded fields present. The caller will align
    # columns to the model's expected feature_names before prediction.

    # set dtypes for known encoded columns (integers) and numeric columns
    for col in list(df_row.columns):
        if col in encoders:
            df_row[col] = pd.to_numeric(df_row[col], errors='coerce').fillna(-1).astype(int)
        else:
            df_row[col] = pd.to_numeric(df_row[col], errors='coerce').fillna(0).astype(float)

    return df_row


def _get_model_feature_names(model):
    """Return a list of feature names expected by the model, or None."""
    try:
        # sklearn wrapper (XGBClassifier / XGBRegressor)
        if hasattr(model, 'get_booster'):
            booster = model.get_booster()
            if hasattr(booster, 'feature_names'):
                return booster.feature_names
            # some versions store as attribute
            return getattr(booster, 'feature_names', None)

        # raw Booster
        if isinstance(model, xgb.core.Booster):
            if hasattr(model, 'feature_names'):
                return model.feature_names
            return getattr(model, 'feature_names', None)

        # fallback
        return getattr(model, 'feature_names', None)
    except Exception:
        return None


def _align_input_to_model(X_df: pd.DataFrame, model_features: list, encoders: dict):
    """Ensure X_df has exactly the columns in model_features in the same order.
    - Add missing features with safe defaults: -1 for encoded categorical, 0 for numeric.
    - Drop any extra columns not in model_features.
    - Re-order columns to match model_features.
    Returns a new DataFrame aligned to model_features.
    """
    if model_features is None:
        return X_df

    X = X_df.copy()

    # Add missing
    for feat in model_features:
        if feat not in X.columns:
            if feat in encoders:
                X[feat] = -1
            else:
                # assume numeric default
                X[feat] = 0

    # Drop extras
    for col in list(X.columns):
        if col not in model_features:
            X = X.drop(columns=[col])

    # Reorder
    try:
        X = X[model_features]
    except Exception:
        # fallback: align intersection then append missing
        inter = [c for c in model_features if c in X.columns]
        X = X[inter]

    # ensure dtypes: encoded columns -> int, others -> float
    for col in X.columns:
        if col in encoders:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(-1).astype(int)
        else:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0).astype(float)

    return X


# -------------------------
# Streamlit app entry
# -------------------------
def run_ml_app():
    st.subheader("Machine Learning — Delay Prediction")

    # load metadata (encoders and feature order)
    try:
        df_train, encoders, feature_columns, raw_unique = _load_training_metadata()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # load model
    model_path_default = "Tuned_Best_XGBoost.pkl"
    try:
        model = _load_model_cached(model_path_default)
    except FileNotFoundError:
        st.error(f"Model tidak ditemukan. Letakkan `{model_path_default}` di folder proyek agar prediksi dapat berjalan.")
        return
    except Exception as e:
        st.warning(f"Gagal memuat model: {e}")
        model = None

    st.markdown("---")
    st.markdown("**Masukkan informasi penerbangan untuk prediksi risiko delay**")

    # UI options
    airline_opt = raw_unique.get('Airline', []) if raw_unique else []
    airport_from_opt = raw_unique.get('AirportFrom', []) if raw_unique else []
    airport_to_opt = raw_unique.get('AirportTo', []) if raw_unique else []

    col1, col2 = st.columns(2)
    with col1:
        airline = st.selectbox('Airline', options=airline_opt)
        airport_from = st.selectbox('Airport From', options=airport_from_opt)
        airport_to = st.selectbox('Airport To', options=airport_to_opt)

    with col2:
        length_min = st.slider('Flight Duration (minutes)', min_value=30, max_value=600, value=120)
        time_hour = st.slider('Departure Time (hour 0-23)', min_value=0.0, max_value=23.0, value=8.0, step=0.5)
        dow_map = {1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat', 7: 'Sun'}
        dow = st.selectbox('Day of Week', options=list(dow_map.keys()), format_func=lambda x: f"{x} - {dow_map[x]}")

    st.markdown("---")
    if st.button('Predict Delay'):
        if model is None:
            st.error('Tidak ada model — letakkan `Tuned_Best_XGBoost.pkl` di folder proyek.')
        else:
            rute_str = f"{airport_from}-{airport_to}"
            rute_classes = raw_unique.get('Rute', []) if raw_unique else []
            used_rute = rute_str
            if rute_str not in rute_classes:
                # fallback
                if df_train is not None and 'AirportFrom' in df_train.columns:
                    candidates = df_train[df_train['AirportFrom'].astype(str) == str(airport_from)]['Rute']
                    if not candidates.empty:
                        used_rute = candidates.mode().iloc[0]
                    else:
                        used_rute = df_train['Rute'].mode().iloc[0]
                else:
                    # fallback to first known route (from encoders) if available
                    r_classes = raw_unique.get('Rute', [])
                    used_rute = r_classes[0] if r_classes else rute_str
                # silently use fallback route when unseen

            input_row = {
                'Airline': airline,
                'AirportFrom': airport_from,
                'AirportTo': airport_to,
                'Length_min': length_min,
                'Time_hour': time_hour,
                'DayOfWeek': dow,
                'Rute_override': used_rute
            }

            X_input = _preprocess_input(input_row, encoders, feature_columns)

            # Align created input to the model's expected feature names (prevent feature_names mismatch)
            model_feature_names = _get_model_feature_names(model)
            if model_feature_names is None:
                # fallback to metadata-derived feature list
                model_feature_names = feature_columns

            X_input = _align_input_to_model(X_input, model_feature_names, encoders)

            try:
                if isinstance(model, xgb.core.Booster):
                    dmat = xgb.DMatrix(X_input)
                    proba_arr = model.predict(dmat)
                    prob_delay = float(proba_arr[0])
                    pred = int(prob_delay >= 0.5)
                else:
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_input)[0]
                        prob_delay = proba[1]
                    else:
                        prob_delay = None
                    pred = int(model.predict(X_input)[0])

                if prob_delay is not None:
                    st.metric(label='Probabilitas Delay', value=f"{prob_delay:.2%}", delta=None)

                if pred == 1:
                    st.error('Prediction: POTENSI DELAY')
                else:
                    st.success('Prediction: ON-TIME')

                # do not display preprocessed input in UI

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memprediksi: {e}")

    st.markdown('---')
