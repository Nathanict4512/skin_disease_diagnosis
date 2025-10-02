# main.py

import streamlit as st
import base64
from app import home, diagnose, result, performance, benchmark, about, login

# --- Set faint background image ---
def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: top center;
            background-repeat: no-repeat;
            
        }}
        .block-container {{
            background-color: rgba(123, 123, 123, 0.88);
            padding: 2rem;
            border-radius: 12px;
            backdrop-filter: blur(4px);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# 🖼️ Set your client's image here
set_background("client_bg.jpg")  # Make sure this image is in the same directory as main.py

# --- Define pages ---
PROTECTED_PAGES = {
    "Diagnose": diagnose,
    "Model Performance": performance,
}

PUBLIC_PAGES = {
    "Home": home,
    # Add more public pages if needed
}

ALL_PAGES = {**PUBLIC_PAGES, **PROTECTED_PAGES}

# --- Initialize login state ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# --- Sidebar navigation ---
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(ALL_PAGES.keys()))

# --- Page display logic ---
if selection in PROTECTED_PAGES:
    if not st.session_state.logged_in:
        login.login()
    else:
        st.sidebar.markdown(f"👤 Logged in as: `{st.session_state.username}`")
        PROTECTED_PAGES[selection].app()

        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.success("You have been logged out.")
            st.rerun()
else:
    PUBLIC_PAGES[selection].app()
