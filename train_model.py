import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Path to dataset
DATA_PATH = r"C:\Users\honsa\fake-news-detection\dataset\news_dataset.csv"

# Load dataset
data = pd.read_csv(DATA_PATH)

# Combine title + text into one field for better context
data["content"] = data["title"] + " " + data["text"]

X = data["content"]
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert text to numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model (Logistic Regression)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate model
y_pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save trained model and vectorizer
MODEL_DIR = r"C:\Users\honsa\fake-news-detection\model"
os.makedirs(MODEL_DIR, exist_ok=True)

with open(os.path.join(MODEL_DIR, "fake_news_model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(vectorizer, f)

print(f"\n✅ Model and vectorizer saved in: {MODEL_DIR}")
