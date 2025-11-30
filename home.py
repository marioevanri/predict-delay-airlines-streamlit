import streamlit as st


def run_home():
    st.title("Predict Delay — Airlines")
    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Ringkasan Proyek")
        st.write(
            "Aplikasi ini memprediksi potensi keterlambatan penerbangan (Delay) menggunakan model machine learning yang dilatih pada dataset maskapai. "
            "Model utama yang digunakan adalah XGBoost (hasil tuning) yang menyajikan trade-off antara precision dan recall untuk kebutuhan early-warning."
        )

        st.subheader("Fitur Utama")
        st.markdown(
            """- Prediksi probabilitas delay untuk satu penerbangan.
    - Preprocessing sesuai pipeline (Rute, konversi waktu, periodisasi, Holiday, label encoding)."""
        )

    with col2:
        st.header("About")
        st.markdown("**Author:** Kelompok 2 - Batch 53")
        st.markdown("---")
        st.markdown("_Teknologi: Python, scikit-learn, XGBoost, Streamlit_")

    st.markdown("---")
