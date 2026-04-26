import streamlit as st
import pandas as pd
import os

# -------------------------
# FILE PATH (IMPORTANT FIX)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_FILE = os.path.join(BASE_DIR, "users.csv")

# -------------------------
# LOAD USERS
# -------------------------
def load_users():
    if not os.path.exists(USER_FILE):
        df = pd.DataFrame(columns=["username", "password"])
        df.to_csv(USER_FILE, index=False)

    df = pd.read_csv(USER_FILE)

    # 🔥 FORCE STRING (VERY IMPORTANT)
    df["username"] = df["username"].astype(str)
    df["password"] = df["password"].astype(str)

    return df

# -------------------------
# SAVE USER
# -------------------------
def save_user(username, password):
    df = load_users()

    username = str(username).strip()
    password = str(password).strip()

    new_row = pd.DataFrame([[username, password]], columns=["username", "password"])
    df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(USER_FILE, index=False)

# -------------------------
# LOGIN / REGISTER
# -------------------------
def login():
    st.sidebar.title("🔐 User System")

    menu = st.sidebar.radio("Select", ["Login", "Register"])

    if menu == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type="password")

        if st.sidebar.button("Login"):
            df = load_users()

            username = username.strip()
            password = password.strip()

            user = df[
                (df["username"] == username) &
                (df["password"] == password)
            ]

            if not user.empty:
                st.session_state["logged_in"] = True
                st.session_state["user"] = username
                st.sidebar.success("Login successful ✅")
            else:
                st.sidebar.error("Invalid credentials ❌")

    elif menu == "Register":
        new_user = st.sidebar.text_input("New Username")
        new_pass = st.sidebar.text_input("New Password", type="password")

        if st.sidebar.button("Register"):
            df = load_users()

            new_user = str(new_user).strip()
            new_pass = str(new_pass).strip()

            if new_user in df["username"].values:
                st.sidebar.warning("User already exists ⚠️")
            else:
                save_user(new_user, new_pass)
                st.sidebar.success("Registered successfully ✅")

# -------------------------
# CHECK LOGIN
# -------------------------
def check_login():
    return st.session_state.get("logged_in", False)

# -------------------------
# LOGOUT
# -------------------------
def logout():
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.experimental_rerun()