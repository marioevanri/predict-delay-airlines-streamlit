import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import xgboost as xgb
import os
from sklearn.preprocessing import LabelEncoder


@st.cache_data(show_spinner=False)
def _load_training_metadata(csv_path="Airlines.csv"):
    """Load dataset to recreate preprocessing artifacts (encoders, feature order).
    Returns: df_processed, encoders dict, feature_columns, raw_unique_values
    """
    if not os.path.exists(csv_path):
        # cached function must be pure; raise so caller can handle UI
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Needed to build encoders.")

    df = pd.read_csv(csv_path)
    # mirror notebook steps
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    df = df.drop_duplicates()

    # create route
    df['Rute'] = df['AirportFrom'].astype(str) + '-' + df['AirportTo'].astype(str)

    # convert Length and Time from minutes to hours (as in notebook)
    df['Length'] = df['Length'] / 60.0
    df['Time'] = df['Time'] / 60.0

    # departure period
    departure_period = []
    for i in range(len(df)):
        t = df['Time'].iloc[i]
        if (t >= 5) & (t < 12):
            departure_period.append(0)
        elif (t >= 12) & (t < 17):
            departure_period.append(1)
        elif (t >= 17) & (t < 21):
            departure_period.append(2)
        else:
            departure_period.append(3)
    df['Departure_period'] = departure_period

    # holiday: 6 or 7
    holiday = []
    for i in range(len(df)):
        if (df['DayOfWeek'].iloc[i] == 6) | (df['DayOfWeek'].iloc[i] == 7):
            holiday.append(1)
        else:
            holiday.append(0)
    df['Holiday'] = holiday

    # arrival time and period
    arrival_time = []
    for i in range(len(df)):
        at = df['Time'].iloc[i] + df['Length'].iloc[i]
        if at >= 24:
            arrival_time.append(at - 24)
        else:
            arrival_time.append(at)
    df['Arrival_Time'] = arrival_time

    arrival_period = []
    for i in range(len(df)):
        t = df['Arrival_Time'].iloc[i]
        if (t >= 5) & (t < 12):
            arrival_period.append(0)
        elif (t >= 12) & (t < 17):
            arrival_period.append(1)
        elif (t >= 17) & (t < 21):
            arrival_period.append(2)
        else:
            arrival_period.append(3)
    df['Arrival_period'] = arrival_period

    # encode categorical columns with LabelEncoder per column (fit on training data)
    cat_cols = ['Airline', 'AirportFrom', 'AirportTo', 'Rute']
    encoders = {}
    raw_unique = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        raw_unique[col] = list(le.classes_)

    # features used during training (X = df.drop(columns=['Delay']))
    if 'Delay' in df.columns:
        X = df.drop(columns=['Delay'])
    else:
        X = df.copy()

    feature_columns = list(X.columns)
    return df, encoders, feature_columns, raw_unique


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
        # try best-effort conversion
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


def _preprocess_input(input_data: dict, encoders: dict, feature_columns: list):
    """Given a single input dict, apply same feature engineering and encoding.
    Returns a DataFrame with columns ordered to feature_columns.
    """
    # Build DataFrame for 1 row based on expected raw inputs
    row = {}
    # raw inputs expected keys: Airline, AirportFrom, AirportTo, Length_min, Time_hour, DayOfWeek
    row['Airline'] = input_data['Airline']
    row['AirportFrom'] = input_data['AirportFrom']
    row['AirportTo'] = input_data['AirportTo']
    # convert minutes to hours like notebook
    row['Length'] = input_data['Length_min'] / 60.0
    row['Time'] = input_data['Time_hour']
    row['DayOfWeek'] = input_data['DayOfWeek']

    df_row = pd.DataFrame([row])

    # Rute (allow override from input_data when fallback selected)
    if 'Rute_override' in input_data:
        df_row['Rute'] = input_data['Rute_override']
    else:
        df_row['Rute'] = df_row['AirportFrom'].astype(str) + '-' + df_row['AirportTo'].astype(str)

    # departure period
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

    # holiday
    df_row['Holiday'] = df_row['DayOfWeek'].apply(lambda x: 1 if (x == 6 or x == 7) else 0)

    # arrival time and period
    at = df_row['Time'].iloc[0] + df_row['Length'].iloc[0]
    if at >= 24:
        at = at - 24
    df_row['Arrival_Time'] = at
    df_row['Arrival_period'] = get_period(at)

    # apply encoders
    for col, le in encoders.items():
        val = df_row.loc[0, col]
        # if value not in training classes, show -1 (avoid crash)
        try:
            df_row.loc[0, col] = le.transform([str(val)])[0]
        except ValueError:
            # unseen category: assign -1
            df_row.loc[0, col] = -1

    # ensure all feature columns exist
    for c in feature_columns:
        if c not in df_row.columns:
            df_row[c] = 0

    # order columns
    df_row = df_row[feature_columns]

    # Ensure numeric dtypes: encoded categorical columns -> int, others -> float
    for col in df_row.columns:
        if col in encoders:
            # encoded categorical
            df_row[col] = pd.to_numeric(df_row[col], errors='coerce').fillna(-1).astype(int)
        else:
            # numeric features
            df_row[col] = pd.to_numeric(df_row[col], errors='coerce').fillna(0).astype(float)

    return df_row


def run_ml_app():
    st.subheader("Machine Learning — Delay Prediction")

    # load metadata (encoders and feature order) from training CSV (cached)
    try:
        df_train, encoders, feature_columns, raw_unique = _load_training_metadata()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # load model (cached) and handle errors
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

    # populate UI options from raw_unique
    airline_opt = raw_unique['Airline'] if raw_unique and 'Airline' in raw_unique else []
    airport_from_opt = raw_unique['AirportFrom'] if raw_unique and 'AirportFrom' in raw_unique else []
    airport_to_opt = raw_unique['AirportTo'] if raw_unique and 'AirportTo' in raw_unique else []

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
            # check Rute existence and fallback if unseen
            rute_str = f"{airport_from}-{airport_to}"
            rute_classes = raw_unique.get('Rute', []) if raw_unique else []
            used_rute = rute_str
            rute_warning = False
            if rute_str not in rute_classes:
                # try to find a fallback route that has same AirportFrom
                candidates = df_train[df_train['AirportFrom'].astype(str) == str(airport_from)]['Rute']
                if not candidates.empty:
                    used_rute = candidates.mode().iloc[0]
                else:
                    # fallback to global most common route
                    used_rute = df_train['Rute'].mode().iloc[0]
                rute_warning = True

            if rute_warning:
                st.warning(f"Rute {rute_str} tidak ditemukan di data training — menggunakan rute fallback: {used_rute}")

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

            try:
                # handle XGBoost Booster fallback (if model is xgboost.core.Booster)
                if isinstance(model, xgb.core.Booster):
                    dmat = xgb.DMatrix(X_input)
                    proba_arr = model.predict(dmat)
                    # proba_arr may be shape (n,) for binary classification
                    prob_delay = float(proba_arr[0])
                    pred = int(prob_delay >= 0.5)
                else:
                    # sklearn-like API (XGBClassifier wrapper or other scikit models)
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_input)[0]
                        prob_delay = proba[1]
                    else:
                        prob_delay = None

                    pred = int(model.predict(X_input)[0])

                # show result
                if prob_delay is not None:
                    st.metric(label='Probabilitas Delay', value=f"{prob_delay:.2%}", delta=None)

                if pred == 1:
                    st.error('Prediction: POTENSI DELAY')
                else:
                    st.success('Prediction: ON-TIME')

                with st.expander('Input fitur (setelah preprocessing)'):
                    st.write(X_input.transpose())

            except Exception as e:
                st.error(f"Terjadi kesalahan saat memprediksi: {e}")

    st.markdown('---')