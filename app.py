import streamlit as st

st.set_page_config(
    page_title="Airline Delay Prediction",
    layout="centered"
)

st.title("✈️ Airline Delay Prediction")

st.markdown("""
### 👋 Selamat Datang!

Aplikasi ini bertujuan untuk **memprediksi kemungkinan keterlambatan penerbangan**
menggunakan **Machine Learning**.

Model memanfaatkan informasi seperti:
- Maskapai
- Rute penerbangan
- Hari keberangkatan
- Waktu keberangkatan
- Waktu tiba

➡️ **Silakan gunakan sidebar di kiri** untuk masuk ke halaman prediksi ML.
""")

st.info("📌 Sidebar akan muncul otomatis jika folder `pages/` terdeteksi.")
