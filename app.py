import streamlit as st
import pandas as pd
import joblib
import os

# Load saved model and vectorizer
model = joblib.load("personality_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Personality info
personality_info = {
    "INTP": ("🧠", "Logical, analytical thinker", "Data Scientist, Engineer"),
    "ENFP": ("🌟", "Creative and enthusiastic", "Marketing, HR, Content Creator"),
    "INTJ": ("♟️", "Strategic and independent", "Architect, Analyst"),
    "INFJ": ("🔮", "Visionary and thoughtful", "Psychologist, Writer"),
    "ESTJ": ("📊", "Organized and leader", "Manager, Administrator"),
    "ISFP": ("🎨", "Artistic and sensitive", "Designer, Artist")
}

st.set_page_config(page_title="Personality Predictor", layout="centered")

st.title("🧠 Personality Prediction System")
st.write("ML-based MBTI Personality Prediction App")

text = st.text_area("Enter your text")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Enter text")
    else:
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]

        emoji, desc, career = personality_info.get(
            prediction,
            ("🙂", "Personality type", "Various careers")
        )

        st.success(f"{emoji} **Predicted Personality: {prediction}**")
        st.write("**Description:**", desc)
        st.write("**Suggested Careers:**", career)

        # Feedback
        st.subheader("Was this correct?")
        feedback = st.radio("Your opinion", ["Correct", "Wrong"])
        if st.button("Submit Feedback"):
            with open("feedback.csv", "a") as f:
                f.write(f"{text},{prediction},{feedback}\n")
            st.success("Feedback saved")

# Dashboard
if os.path.exists("feedback.csv"):
    st.subheader("Feedback Dashboard")
    df = pd.read_csv("feedback.csv", header=None, names=["Text","Prediction","Feedback"])
    st.bar_chart(df["Feedback"].value_counts())