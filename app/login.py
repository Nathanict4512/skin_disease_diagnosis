# app/login.py

import streamlit as st
import json
import os

USERS_FILE = os.path.join(os.getcwd(), "users.json")

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, "r") as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)

def login():
    st.title("🔐 Authentication Page")

    menu = st.radio("Choose an option", ["Login", "Register"])

    users = load_users()

    if menu == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_btn = st.button("Login")

        if login_btn:
            if username in users and users[username]["password"] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success(f"✅ Welcome {username}!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password.")

    elif menu == "Register":
        st.subheader("Create a new account")
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        register_btn = st.button("Register")

        if register_btn:
            if new_username in users:
                st.warning("⚠️ Username already exists.")
            elif new_password != confirm_password:
                st.error("❌ Passwords do not match.")
            elif not new_username or not new_password:
                st.error("❌ Please fill in all fields.")
            else:
                users[new_username] = {"password": new_password}
                save_users(users)
                st.success("✅ Registration successful! You can now log in.")
