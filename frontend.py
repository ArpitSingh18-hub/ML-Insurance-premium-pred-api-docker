import streamlit as st
import requests
import pandas as pd

API_URL = "http://16.176.174.195:8000/predict"

# 🎨 Page config
st.set_page_config(page_title="Insurance AI Predictor", page_icon="💡", layout="centered")

# 🎨 Styling
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #1f4037, #99f2c8);
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

# 🏆 Header
st.title("💡 Insurance Premium Predictor")
st.markdown("### Simple tool to estimate your insurance category")

# 📦 Input Layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 1, 119, 30)
    height = st.number_input("Height (m)", 0.5, 2.5, 1.7)
    smoker = st.selectbox("Smoker?", [True, False])

with col2:
    weight = st.number_input("Weight (kg)", 1.0, 150.0, 65.0)
    income_lpa = st.number_input("Income (LPA)", 0.1, 100.0, 10.0)

city = st.text_input("City", value="Mumbai")
occupation = st.selectbox(
    "Occupation",
    ['retired', 'freelancer', 'student', 'government job', 'business_owner', 'unemployed']
)

# 🧠 BMI Feature
bmi = weight / (height ** 2)
st.metric("Your BMI", f"{bmi:.2f}")

# 🚀 Predict Button
if st.button("🚀 Predict My Insurance Category"):

    data = {
        "age": age,
        "weight": weight,
        "height": height,
        "income_lpa": income_lpa,
        "smoker": smoker,
        "city": city.lower(),
        "occupation": occupation
    }

    try:
        with st.spinner("Analyzing your data..."):
            response = requests.post(API_URL, json=data)

        if response.status_code == 200:
            result = response.json()

            prediction = result["response"]["predicted_category"]
            confidence = result["response"]["confidence"]
            probs = result["response"]["class_probabilities"]

            # 🎯 Color Output
            if prediction == "High":
                st.error(f"🚨 High Risk Category")
            elif prediction == "Medium":
                st.warning(f"⚠️ Medium Risk Category")
            else:
                st.success(f"✅ Low Risk Category")

            st.info(f"Confidence: {confidence:.2f}")

            # 📊 Graph
            df = pd.DataFrame({
                "Category": list(probs.keys()),
                "Probability": list(probs.values())
            })

            st.bar_chart(df.set_index("Category"))

            # 📦 Expand
            with st.expander("See detailed data"):
                st.json(probs)

        else:
            st.error("Server error. Try again.")

    except Exception as e:
        st.error("Cannot connect to server. Check internet or API.")

# 🔗 Help Button
st.markdown("---")
st.link_button("📘 View API Docs (Advanced)", "http://16.176.174.195:8000/docs")