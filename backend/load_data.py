import pandas as pd
import os

# Path to dataset
DATASET_PATH = os.path.join("dataset")

# Load datasets
fake = pd.read_csv(os.path.join(DATASET_PATH, "Fake.csv"))
true = pd.read_csv(os.path.join(DATASET_PATH, "True.csv"))

print("✅ Dataset loaded successfully!")
print(f"Fake news samples: {len(fake)}")
print(f"True news samples: {len(true)}")

# Show first 5 rows from each
print("\n--- Fake News Sample ---")
print(fake.head())

print("\n--- True News Sample ---")
print(true.head())
dir C:\Users\honsa\fake-news-detection\model
