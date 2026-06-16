import os
import pickle
from pathlib import Path

import numpy as np
import spacy
import streamlit as st
from dotenv import load_dotenv

from rag.response_generator import ResponseGenerator
from rag.vector_store import EmotionVectorStore


load_dotenv()


class EmoBotRAG:
    def __init__(self, knowledge_base_path=None, api_key=None, use_llm=False, model_name="llama-3.3-70b-versatile", timeout_seconds=20):
        print("Initializing EmoBot RAG...")

        self.project_root = Path(__file__).parent.parent
        self.knowledge_base_path = (
            Path(knowledge_base_path)
            if knowledge_base_path
            else self.project_root / "knowledge_base"
        )

        model_path = self.project_root / "model" / "best_model.pkl"
        vectorizer_path = self.project_root / "model" / "vectorizer.pkl"

        if not model_path.exists():
            st.error(f"Model file not found at {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not vectorizer_path.exists():
            st.error(f"Vectorizer file not found at {vectorizer_path}")
            raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

        with open(model_path, "rb") as f:
            self.classifier = pickle.load(f)

        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        print(f"Loaded classifier and vectorizer from {model_path}")

        self.nlp = spacy.load("en_core_web_sm")
        self.emotion_map = {0: "anger", 1: "fear", 2: "joy"}

        self.vector_store = EmotionVectorStore(
            persist_directory=self.project_root / "data" / "vector_store"
        )
        if self.vector_store.collection.count() == 0:
            print(f"Building vector store from {self.knowledge_base_path}")
            self.vector_store.build_from_text_files(str(self.knowledge_base_path))
        else:
            print(f"Vector store has {self.vector_store.collection.count()} documents")

        self.response_gen = ResponseGenerator(api_key=api_key, use_llm=use_llm, model_name=model_name, timeout_seconds=timeout_seconds)
        print("EmoBot RAG ready")

    def classify_emotion(self, text):
        doc = self.nlp(text)
        filtered = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct
        ]
        processed = " ".join(filtered)
        features = self.vectorizer.transform([processed])

        prediction = self.classifier.predict(features)[0]
        if isinstance(prediction, np.ndarray):
            prediction = prediction.item()

        emotion = self.emotion_map.get(int(prediction), "unknown")
        probabilities = self.classifier.predict_proba(features)[0]
        confidence = float(np.max(probabilities))

        return emotion, confidence

    def classify_only(self, user_input):
        emotion, confidence = self.classify_emotion(user_input)
        return {
            "emotion": emotion,
            "confidence": confidence,
        }

    def process_query(self, user_input):
        emotion, confidence = self.classify_emotion(user_input)
        search_query = f"{user_input} {emotion} coping support"
        retrieved_docs, distances, metadatas = self.vector_store.search(
            search_query,
            n_results=3,
        )

        result = self.response_gen.generate_response(
            user_query=user_input,
            emotion=emotion,
            retrieved_docs=retrieved_docs,
            confidence=confidence,
        )
        result["retrieved_docs"] = retrieved_docs
        result["retrieved_metadata"] = metadatas
        result["emotion"] = emotion
        result["confidence"] = confidence
        result["context_sources"] = len(retrieved_docs)
        return result


def get_bot(api_key, use_llm):
    if "emobot" not in st.session_state:
        with st.spinner("Loading EmoBot RAG. First load can take a minute."):
            st.session_state.emobot = EmoBotRAG(api_key=api_key, use_llm=use_llm)

    st.session_state.emobot.response_gen = ResponseGenerator(
        api_key=api_key,
        use_llm=use_llm,
        model_name="llama-3.3-70b-versatile",
        timeout_seconds=20
    )

    return st.session_state.emobot


def main():
    st.set_page_config(page_title="EmoBot RAG", page_icon="E", layout="wide")

    st.title("EmoBot RAG")
    st.caption("Emotion classification plus retrieval-based emotional support")

    with st.sidebar:
        st.header("Settings")
        api_key_input = st.text_input(
            "API Key (Groq or Gemini)",
            type="password",
            value= "",
            help="Enter your Groq or Gemini API key. Get Groq key from https://console.groq.com",
        )
        use_llm = st.checkbox(
            "Use LLM responses (Groq/Gemini)",
            value=False,
            help="Turn this on only when your API key and internet connection are working.",
        )

        st.header("Knowledge Base")
        st.markdown(
            """
Add or edit `.txt` files in `knowledge_base/`.

After changing those files, rebuild the vector store:

`python scripts/build_vector_store.py --reset`
"""
        )

        st.header("Supported Emotions")
        st.markdown("Anger, Fear, Joy")

    if not api_key_input:
        api_key_input = os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY", "")

    mode = st.radio(
        "Mode",
        ["Classify Only", "RAG Support"],
        horizontal=True,
    )

    user_input = st.text_area(
        "How are you feeling today?",
        placeholder="Example: I feel nervous about my presentation tomorrow.",
        height=120,
    )

    button_label = "Classify Emotion" if mode == "Classify Only" else "Get Support"
    if not st.button(button_label, type="primary") or not user_input.strip():
        return

    try:
        bot = get_bot(api_key_input.strip(), use_llm)
        if mode == "Classify Only":
            result = bot.classify_only(user_input)
        else:
            result = bot.process_query(user_input)
    except Exception as e:
        st.error(f"Error processing your message: {e}")
        return

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Emotion", result["emotion"].title())
    col2.metric("Confidence", f"{result['confidence']:.1%}")
    col3.metric("Sources Used", result.get("context_sources", 0))

    if mode == "RAG Support":
        st.subheader("EmoBot Response")
        st.success(result["response"])

        with st.expander("Retrieved knowledge used for this answer"):
            for index, doc in enumerate(result.get("retrieved_docs", []), start=1):
                metadata = result.get("retrieved_metadata", [{}])[index - 1] or {}
                source = metadata.get("source", "knowledge_base")
                st.markdown(f"**Source {index}: {source}**")
                st.info(doc[:600] + "..." if len(doc) > 600 else doc)

        if result.get("mock"):
            st.warning(
                "Using offline mock responses. Add a working API key (Groq or Gemini) for generated responses."
            )


if __name__ == "__main__":
    main()