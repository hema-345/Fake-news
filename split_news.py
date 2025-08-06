import pandas as pd

# Load the combined dataset (must be in same folder)
df = pd.read_csv("present_news_large.csv")

# Filter real and fake news
real_news = df[df['label'] == 1]
fake_news = df[df['label'] == 0]

# Save to separate files
real_news.to_csv("real_news_2023_2025.csv", index=False)
fake_news.to_csv("fake_news_2023_2025.csv", index=False)

print("✅ Done! Files created:")
print(" - real_news_2023_2025.csv")
print(" - fake_news_2023_2025.csv")
