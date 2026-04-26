import streamlit as st
import pandas as pd
import pickle
import json
from auth import login, check_login, logout

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Mental Health Predictor", layout="wide")

# -------------------------
# CLEAN MINIMAL CSS
# -------------------------
st.markdown("""
<style>
/* Base */
.stApp {
    background: #f5f7fb;
    color: #1f2937;
}

/* Header */
.header {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 4px;
}
.subtext {
    color: #6b7280;
    margin-bottom: 20px;
}

/* Card */
.card {
    background: white;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06);
}

/* KPI */
.kpi {
    font-size: 22px;
    font-weight: 600;
}

/* Buttons */
.stButton>button {
    background: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 10px 18px;
    border: none;
}
.stButton>button:hover {
    background: #1d4ed8;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff;
}

/* Divider */
.hr {
    border-top: 1px solid #e5e7eb;
    margin: 20px 0;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# LOGIN
# -------------------------
login()
if not check_login():
    st.warning("Please login")
    st.stop()

st.sidebar.success(f"👤 {st.session_state.get('user')}")
logout()

# -------------------------
# HEADER
# -------------------------
st.markdown("<div class='header'>🧠 Mental Health Predictor</div>", unsafe_allow_html=True)
st.markdown("<div class='subtext'>Clean AI-based mental health assessment using student data</div>", unsafe_allow_html=True)

# -------------------------
# LOAD DATA
# -------------------------
data = pd.read_csv("data/dataset.csv")
data.columns = data.columns.str.strip().str.lower()
target = "depression"

# -------------------------
# KPI CARDS
# -------------------------
c1, c2, c3 = st.columns(3)

c1.markdown(f"<div class='card'><div>📊 Records</div><div class='kpi'>{len(data)}</div></div>", unsafe_allow_html=True)
c2.markdown(f"<div class='card'><div>🎯 Features</div><div class='kpi'>{len(data.columns)-1}</div></div>", unsafe_allow_html=True)
c3.markdown(f"<div class='card'><div>🤖 Models</div><div class='kpi'>4</div></div>", unsafe_allow_html=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# -------------------------
# MODEL SELECT
# -------------------------
model_name = st.sidebar.selectbox(
    "Choose Model",
    ["RandomForest", "LogisticRegression", "SVM", "HGB"]
)

model = pickle.load(open(f"models/{model_name}.pkl", "rb"))

# -------------------------
# DASHBOARD
# -------------------------
st.markdown("<div class='card'><h4>📊 Insights</h4></div>", unsafe_allow_html=True)

d1, d2 = st.columns(2)

with d1:
    st.bar_chart(data[target].value_counts())

with d2:
    if "gender" in data.columns:
        st.bar_chart(data["gender"].value_counts())

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# -------------------------
# INPUT FORM
# -------------------------
st.markdown("<div class='card'><h4>📝 Enter Details</h4></div>", unsafe_allow_html=True)

input_data = {}
cols = st.columns(2)

i = 0
for col in data.columns:
    if col == target:
        continue

    numeric_col = pd.to_numeric(data[col], errors='coerce')

    with cols[i % 2]:
        if numeric_col.notnull().sum() > 0:
            input_data[col] = st.number_input(col, value=float(numeric_col.mean()))
        else:
            input_data[col] = st.selectbox(col, data[col].dropna().unique())
    i += 1

# -------------------------
# PREDICTION
# -------------------------
if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    result = model.predict(input_df)[0]

    st.markdown("<div class='card'><h4>🎯 Result</h4></div>", unsafe_allow_html=True)

    if result <= 2:
        st.success(f"Low Stress Level ({result})")
    elif result == 3:
        st.warning(f"Moderate Stress Level ({result})")
    else:
        st.error(f"High Stress Level ({result})")

    # Probability
    if hasattr(model, "predict_proba"):
        st.write("### Confidence")
        st.bar_chart(model.predict_proba(input_df)[0])

    # Suggestions
    if result >= 4:
        st.write("### Suggestions")
        st.write("- Improve sleep schedule")
        st.write("- Talk to someone")
        st.write("- Exercise regularly")

    # Download
    result_data = {"prediction": int(result), "input": input_data}

    st.download_button(
        "Download Report",
        data=json.dumps(result_data, indent=4),
        file_name="result.json"
    )
