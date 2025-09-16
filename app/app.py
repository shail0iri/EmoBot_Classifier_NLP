# Streamlit App for Emotion Classification using a Pre-Trained Model and spaCy
%%writefile app.py
import streamlit as st
import joblib
import spacy
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np # import numpy

# Load the saved model and vectorizer
model = joblib.load('/content/best_model.pkl')
vectorizer = pickle.load(open('/content/vectorizer.pkl', 'rb'))

# Load the spaCy English language model
nlp = spacy.load('en_core_web_sm')

# Preprocess function (same as in your notebook)
def preprocess(text):
    doc = nlp(text)
    filtered_text = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_text.append(token.lemma_)
    return ' '.join(filtered_text)

# Streamlit app
st.title("Emotion Classifier")

# Input text
comment = st.text_area("Enter your comment:")

if st.button("Classify"):
    if comment:
        # Preprocess the input text
        processed_comment = preprocess(comment)
        # Vectorize the preprocessed text
        comment_vec = vectorizer.transform([processed_comment])
        # Make a prediction
        prediction = model.predict(comment_vec)[0]  # Convert prediction to scalar

        # Ensure prediction is an integer
        if isinstance(prediction, np.ndarray):
            prediction = prediction.item()  # Convert numpy array to a scalar

        # Define emotion labels (you might need to adjust these based on your encoding)
        emotion_labels = {
            0: 'anger',
            1: 'fear',
            2: 'joy',  # Added surprise if it's in your data
        }

        # Display the predicted emotion
        st.write(f"Predicted Emotion: {emotion_labels.get(prediction, 'Unknown')}")
    else:
        st.write("Please enter a comment.")