import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load datasets
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("Real.csv")

# Add labels
fake_df['label'] = 0
real_df['label'] = 1

# Select relevant columns and clean
fake_df = fake_df[['text', 'label']].dropna()
real_df = real_df[['text', 'label']].dropna()

fake_df['text'] = fake_df['text'].apply(clean_text)
real_df['text'] = real_df['text'].apply(clean_text)

# Combine and shuffle
data = pd.concat([fake_df, real_df])
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
data.to_csv("news.csv", index=False)
print("✅ Dataset cleaned and saved as 'news.csv'")
