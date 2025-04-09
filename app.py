import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Streamlit UI
st.title("🌪️ Fake Disaster Tweet Detection")
st.write("Enter a tweet below to check if it's real or fake.")

tweet = st.text_area("Tweet Text")

if st.button("Predict"):
    if tweet.strip() == "":
        st.warning("Please enter a tweet.")
    else:
        vec = vectorizer.transform([tweet])
        prediction = model.predict(vec)[0]
        label = "✅ Real" if prediction == 1 else "❌ Fake"
        st.subheader(f"Prediction: {label}")
