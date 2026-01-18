import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
data = pd.read_csv("mbti_1.csv")
print(data.head())
print("Dataset loaded")

# clean the text
data = pd.read_csv("mbti_1.csv")
print("Dataset loaded")
data = data.sample(2000)
stop_words = set(stopwords.words('english'))
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)
print("Cleaning started...")
data['cleaned_posts'] = data['posts'].apply(clean_text)
print("Cleaning finished")

# split input and output 
X = data['cleaned_posts']
y = data['type']

# Train-Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# convert text to numbers
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("Vectorization done")

# Train ML Model
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)
print("Model trained")

# Test Accuracy
from sklearn.metrics import accuracy_score
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Predict custom text
def predict_personality(text):
    text = clean_text(text)
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]
print(predict_personality("I enjoy deep thinking and solving problems"))