import streamlit as st
import pickle

# Correct paths to your saved model/vectorizer
MODEL_PATH = r"C:\Users\HP\fake-news-detection\model\fake_news_model.pkl"
VECTORIZER_PATH = r"C:\Users\HP\fake-news-detection\model\tfidf_vectorizer.pkl"

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

st.title("📰 Fake News Detection App")

user_input = st.text_area("Enter News Article Text:")

if st.button("Predict"):
    if user_input.strip():
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        prob = model.predict_proba(X)[0][prediction] * 100

        if prediction == 1:
            st.success(f"🟢 Real News (Confidence: {prob:.2f}%)")
        else:
            st.error(f"🔴 Fake News (Confidence: {prob:.2f}%)")
    else:
        st.warning("⚠️ Please enter some text before predicting.")
