import streamlit as st
from ml_app import run_ml_app
from home import run_home


def main():
    menu = ['Home', 'Machine Learning']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Home':
        run_home()
    elif choice == 'Machine Learning':
        run_ml_app()

if __name__ == "__main__":
    main()