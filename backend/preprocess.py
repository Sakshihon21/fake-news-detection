import pandas as pd
import os

# ✅ Absolute path to dataset folder
BASE_DIR = r"C:\Users\honsa\fake-news-detection\dataset"

# Load datasets
true_news = pd.read_csv(os.path.join(BASE_DIR, "True.csv"))
fake_news = pd.read_csv(os.path.join(BASE_DIR, "Fake.csv"))

# Add labels: 1 = True news, 0 = Fake news
true_news["label"] = 1
fake_news["label"] = 0

# Merge both datasets
data = pd.concat([true_news, fake_news], axis=0)

# Shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# Show basic info
print("Dataset shape:", data.shape)
print("Columns:", data.columns.tolist())
print("\nSample data:\n", data.head())

# Save preprocessed dataset
OUTPUT_PATH = os.path.join(BASE_DIR, "news_dataset.csv")
data.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ Preprocessed dataset saved at: {OUTPUT_PATH}")
