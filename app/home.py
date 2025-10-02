# app/home.py
import streamlit as st
from PIL import Image
import os

def app():
    # st.set_page_config(page_title="AI Skin Diagnosis", layout="centered")
    
    st.title("An optimized 🧠 deep learning-based system for accurate detection and classification of skin diseases By **Makolo Daniel  (2021100000502)**")
    
    st.markdown("""
    Welcome to the **AI-based Skin Disease Diagnosis** platform. This tool leverages deep learning 
    to analyze dermoscopic images and assist with early detection of various skin conditions. 
    It's fast, accessible, and supports dermatologists with cutting-edge image analysis.
    """)

    # Sample Images Section
    st.subheader("📸 Sample Dermoscopic Images")
    sample_dir = "static/sample_images"
    image_files = [f for f in os.listdir(sample_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    cols = st.columns(len(image_files))
    for i, img_file in enumerate(image_files):
        with cols[i]:
            image = Image.open(os.path.join(sample_dir, img_file))
            st.image(image, caption=f"Sample {i+1}", use_container_width=True)

    # Highlight Section
    st.markdown("""
    ### ⚡ Why Use This Tool?
    - Quick and easy analysis of skin images.
    - Enhances **early detection** and **clinical decision-making**.
    - Combines **AI and dermatology expertise** for improved accuracy.
    """)

    # System Performance Badge (Optional)
    st.success("✅ Current Model Accuracy: **89.4%** on ISIC-2020 test dataset")

    # Call to Action