import streamlit as st
import pandas as pd
import re
import nltk
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# ---------------- PERSONALITY DATA ----------------
personality_info = {
    "INTP": ("🧠", "Logical, analytical thinker", "Data Scientist, Engineer"),
    "ENFP": ("🌟", "Creative and enthusiastic", "Marketing, HR, Content Creator"),
    "INTJ": ("♟️", "Strategic and independent", "Architect, Analyst"),
    "INFJ": ("🔮", "Visionary and thoughtful", "Psychologist, Writer"),
    "ESTJ": ("📊", "Organized and leader", "Manager, Administrator"),
    "ISFP": ("🎨", "Artistic and sensitive", "Designer, Artist")
}

# ---------------- TRAIN MODELS ----------------
@st.cache_resource
def train_models():
    data = pd.read_csv("mbti_1.csv")

    data = data.groupby("type").apply(lambda x: x.sample(min(len(x), 150))).reset_index(drop=True)

    data["cleaned"] = data["posts"].apply(clean_text)

    X = data["cleaned"]
    y = data["type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    lr = LogisticRegression(max_iter=3000)
    nb = MultinomialNB()
    svm = LinearSVC()

    lr.fit(X_train_vec, y_train)
    nb.fit(X_train_vec, y_train)
    svm.fit(X_train_vec, y_train)

    return lr, nb, svm, vectorizer

lr, nb, svm, vectorizer = train_models()

# ---------------- PREDICTION ----------------
def predict_all(text):
    text = clean_text(text)
    vec = vectorizer.transform([text])

    p1 = lr.predict(vec)[0]
    p2 = nb.predict(vec)[0]
    p3 = svm.predict(vec)[0]

    ensemble = max(set([p1, p2, p3]), key=[p1, p2, p3].count)

    return p1, p2, p3, ensemble

# ---------------- UI ----------------
st.set_page_config(page_title="Personality Predictor", layout="centered")

theme = st.toggle("🌙 Dark Mode")

if theme:
    st.markdown("<style>body{background-color:#121212;color:white;}</style>", unsafe_allow_html=True)

st.title("🧠 Personality Prediction System")
st.write("Unique ML app using NLP, multi-model ensemble and dashboard")

text = st.text_area("Enter your text")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Enter text")
    else:
        lr_p, nb_p, svm_p, final = predict_all(text)

        emoji, desc, career = personality_info.get(final, ("🙂", "Personality type", "Various careers"))

        st.success(f"{emoji} **Final Personality: {final}**")
        st.write("**Description:**", desc)
        st.write("**Suggested Careers:**", career)

        st.subheader("Model Comparison")
        st.write("Logistic Regression:", lr_p)
        st.write("Naive Bayes:", nb_p)
        st.write("SVM:", svm_p)
        st.write("**Ensemble Result:**", final)

        # Feedback
        st.subheader("Was this correct?")
        feedback = st.radio("Your opinion", ["Correct", "Wrong"])
        if st.button("Submit Feedback"):
            with open("feedback.csv", "a") as f:
                f.write(f"{text},{final},{feedback}\n")
            st.success("Feedback saved")

# ---------------- DASHBOARD ----------------
if os.path.exists("feedback.csv"):
    st.subheader("Feedback Dashboard")
    df = pd.read_csv("feedback.csv", header=None, names=["Text","Prediction","Feedback"])
    st.bar_chart(df["Feedback"].value_counts())