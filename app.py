import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("📰 Fake News Detector (2025)")
st.markdown("Enter a news article to check if it's **real or fake**.")

user_input = st.text_area("Paste the article content here:")

if st.button("Check Now"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some content.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        result = model.predict(vectorized)[0]

        if result == 1:
            st.success("🟢 This news is predicted as REAL.")
        else:
            st.error("🔴 This news is predicted as FAKE.")
