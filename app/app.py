# Streamlit App for Emotion Classification using a Pre-Trained Model and spaCy
import streamlit as st
import joblib
import spacy
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Load the saved model and vectorizer from the model folder
model_path = PROJECT_ROOT / "model" / "best_model.pkl"
vectorizer_path = PROJECT_ROOT / "model" / "vectorizer.pkl"

@st.cache_resource
def load_model():
    """Load the pre-trained model and vectorizer with caching"""
    try:
        model = joblib.load(model_path)
        vectorizer = pickle.load(open(vectorizer_path, 'rb'))
        return model, vectorizer
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_resource
def load_spacy():
    """Load spaCy model with caching"""
    try:
        return spacy.load('en_core_web_sm')
    except OSError:
        st.error("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
        st.stop()

# Load everything
model, vectorizer = load_model()
nlp = load_spacy()

# Preprocess function (same as in your notebook)
def preprocess(text):
    """Preprocess text by removing stopwords, punctuation, and applying lemmatization"""
    doc = nlp(text)
    filtered_text = []
    for token in doc:
        if token.is_stop or token.is_punct:
            continue
        filtered_text.append(token.lemma_)
    return ' '.join(filtered_text)

# Emotion labels mapping
emotion_labels = {
    0: '😠 Anger',
    1: '😨 Fear',
    2: '😊 Joy'
}

# Emoji mapping for display
emotion_icons = {
    'anger': '😠',
    'fear': '😨',
    'joy': '😊'
}

# Streamlit app configuration
st.set_page_config(
    page_title="Emotion Classifier",
    page_icon="😊",
    layout="centered"
)

# App title and description
st.title("🎭 Emotion Classifier")
st.markdown("""
This app uses a machine learning model to detect emotions in text.
Simply enter your text below and click **Classify** to see the predicted emotion.
""")

# Sidebar with information
with st.sidebar:
    st.header("📋 About")
    st.markdown("""
    This emotion classifier can detect:
    - **Anger** 😠
    - **Fear** 😨
    - **Joy** 😊
    
    The model uses:
    - CatBoost Classifier
    - TF-IDF Vectorization
    - spaCy for text preprocessing
    """)
    
    st.header("💡 Tips")
    st.markdown("""
    - Write natural language
    - Longer texts may give better predictions
    - The model works best on emotional content
    """)

# Input area
st.markdown("---")
comment = st.text_area(
    "📝 Enter your text:", 
    placeholder="e.g., I'm so excited about this! I can't wait...",
    height=150
)

# Classification button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    classify_button = st.button("🔍 Classify Emotion", type="primary", use_container_width=True)

# Process classification
if classify_button:
    if comment and comment.strip():
        with st.spinner("Analyzing your text..."):
            try:
                # Preprocess the input text
                processed_comment = preprocess(comment)
                
                # Vectorize the preprocessed text
                comment_vec = vectorizer.transform([processed_comment])
                
                # Make prediction
                prediction = model.predict(comment_vec)[0]
                
                # Get prediction probabilities
                probabilities = model.predict_proba(comment_vec)[0]
                confidence = np.max(probabilities) * 100
                
                # Ensure prediction is scalar
                if isinstance(prediction, np.ndarray):
                    prediction = prediction.item()
                
                # Get emotion label
                emotion = emotion_labels.get(prediction, 'Unknown')
                emotion_name = emotion.split()[-1].lower() if emotion != 'Unknown' else 'unknown'
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Results")
                
                # Show emotion with emoji
                emotion_emoji = emotion_icons.get(emotion_name, '🎭')
                st.markdown(f"### {emotion_emoji} Predicted Emotion: **{emotion}**")
                
                # Show confidence
                st.markdown(f"### 📈 Confidence: **{confidence:.1f}%**")
                
                # Show confidence bar
                st.progress(confidence / 100)
                
                # Show all probabilities in expander
                with st.expander("🔍 View Detailed Probabilities"):
                    prob_df = pd.DataFrame({
                        'Emotion': [emotion_labels.get(i, 'Unknown') for i in range(len(probabilities))],
                        'Probability': [f"{p*100:.1f}%" for p in probabilities],
                        'Score': probabilities
                    })
                    prob_df = prob_df.sort_values('Score', ascending=False)
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
                
                # Show preprocessing steps in expander
                with st.expander("📝 View Text Processing"):
                    st.markdown("**Original Text:**")
                    st.info(comment)
                    st.markdown("**Processed Text:**")
                    st.code(processed_comment if processed_comment else "[Empty after preprocessing]")
                
                # Show processing stats
                st.markdown("---")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("📝 Original Length", len(comment))
                with col2:
                    st.metric("🔧 Processed Length", len(processed_comment))
                with col3:
                    st.metric("📊 Words Removed", len(comment.split()) - len(processed_comment.split()))
                
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")
                st.info("Please try again with different text.")
                
    else:
        st.warning("⚠️ Please enter some text to classify.")

# Add example texts in expander
with st.expander("💡 Try these examples"):
    st.markdown("""
    **Anger Examples:**
    - "This is absolutely frustrating! I can't believe this happened!"
    - "I'm so angry right now, nothing is working!"
    
    **Fear Examples:**
    - "I'm really scared about my presentation tomorrow."
    - "What if something bad happens? I'm so worried."
    
    **Joy Examples:**
    - "I'm so happy! Today was the best day ever!"
    - "I'm excited about my new job, this is wonderful!"
    """)
    
    if st.button("📋 Copy Example"):
        st.info("Click in the text area above and type/paste your own text!")

# Footer
st.markdown("---")
st.caption("🤖 Emotion Classifier | Powered by CatBoost, spaCy, and Streamlit")