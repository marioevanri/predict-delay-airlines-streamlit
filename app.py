import streamlit as st

st.set_page_config(
    page_title="Airline Delay Prediction",
    page_icon="✈️",
    layout="centered"
)

st.title("✈️ Airline Delay Prediction")

st.markdown("""
### 👋 Selamat Datang!

Aplikasi ini bertujuan untuk **memprediksi kemungkinan keterlambatan penerbangan**
menggunakan **Machine Learning**.

Model memanfaatkan informasi seperti:
- Maskapai
- Rute
- Hari
- Waktu Keberangkatan
- Waktu Tiba

➡️ Silakan buka **menu di sidebar** untuk masuk ke halaman prediksi.
""")

st.info("📌 Gunakan sidebar untuk berpindah halaman.")
