import pickle
import os

# Path to model directory
MODEL_DIR = r"C:\Users\honsa\fake-news-detection\model"

# Load trained model
with open(os.path.join(MODEL_DIR, "fake_news_model.pkl"), "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

def predict_news(news_text):
    """
    Predict if a news article is Real (1) or Fake (0).
    """
    # Transform input text
    transformed_text = vectorizer.transform([news_text])
    
    # Make prediction
    prediction = model.predict(transformed_text)[0]
    confidence = model.predict_proba(transformed_text).max()

    result = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
    return result, round(confidence*100, 2)

# Example test
if __name__ == "__main__":
    test_news = "Donald Trump won the 2020 US Presidential election."
    result, confidence = predict_news(test_news)
    print(f"News: {test_news}")
    print(f"Prediction: {result} (Confidence: {confidence}%)")
