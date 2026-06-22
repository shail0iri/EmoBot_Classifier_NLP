# EmoBot — Emotion-Aware RAG Chatbot

An end-to-end NLP project that combines **emotion classification** with a **Retrieval-Augmented Generation (RAG) pipeline** to deliver context-aware emotional support responses.

---

## Overview

EmoBot detects emotions in user text using a CatBoost classifier (95% accuracy), then retrieves relevant coping strategies from a curated knowledge base via ChromaDB and Sentence Transformers. A Groq LLM generates the final empathetic response — with an offline fallback mode when no API key is available.

---

## Features

- **Multi-Model Benchmarking** — 5 ML models trained and evaluated; CatBoost selected as best performer
- **RAG Pipeline** — ChromaDB vector store + Sentence Transformers (`all-MiniLM-L6-v2`) for semantic retrieval
- **Groq LLM Integration** — `llama-3.3-70b-versatile` for fast, context-aware response generation
- **Dual-Mode App** — Online (Groq LLM) and offline fallback (retrieval-only) via Streamlit
- **Curated Knowledge Base** — Structured `.txt` files covering anger, fear, joy, anxiety, coping strategies, and crisis resources
- **Full Pipeline** — Text input → emotion classification → semantic retrieval → LLM response

---

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.9+ |
| NLP & ML | spaCy, scikit-learn, CatBoost, LightGBM, XGBoost |
| RAG | ChromaDB, Sentence Transformers |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| Deployment | Streamlit |
| Experiment Tracking | — |

---

## Model Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| Logistic Regression | — | — |
| Random Forest | — | — |
| LightGBM | — | — |
| XGBoost | — | — |
| **CatBoost** | **~95%** | **0.94** |

> Emotions supported: **Anger**, **Fear**, **Joy**

---

## Project Structure

```
EmoBot_Classifier_NLP/
├── app/
│   ├── app.py                  # Streamlit classifier app
│   ├── rag_main.py             # RAG pipeline + Streamlit RAG app
│   └── rag/
│       ├── response_generator.py   # Groq LLM integration
│       └── vector_store.py         # ChromaDB + Sentence Transformers
├── data/
│   └── vector_store/           # Persisted ChromaDB embeddings (auto-generated)
├── knowledge_base/
│   ├── anger_support.txt
│   ├── anxiety_support.txt
│   ├── fear_support.txt
│   ├── joy_support.txt
│   ├── coping_strategies.txt
│   ├── crisis_and_boundaries.txt
│   └── professional_resources.txt
├── model/
│   ├── best_model.pkl          # Trained CatBoost model
│   └── vectorizer.pkl          # TF-IDF vectorizer
├── scripts/
│   └── build_vector_store.py   # Script to build/rebuild ChromaDB index
├── .env.example                # API key template
├── .gitignore
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/shail0iri/EmoBot_Classifier_NLP.git
cd EmoBot_Classifier_NLP
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Setup

1. Copy the environment template and add your API key:

```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=your_groq_key_here
```

Get a free Groq API key at https://console.groq.com

2. Build the vector store (first run only):

```bash
python scripts/build_vector_store.py
```

To rebuild from scratch:
```bash
python scripts/build_vector_store.py --reset
```

---

## Usage

**Classifier-only app:**
```bash
streamlit run app/app.py
```

**Full RAG app (with LLM support):**
```bash
streamlit run app/rag_main.py
```

In the sidebar, enter your Groq API key and toggle **"Use LLM responses"** for full RAG mode. Without an API key, the app runs in offline fallback mode using retrieval-only responses.

---

## How It Works

```
User Input
    ↓
spaCy Preprocessing (lemmatization, stopword removal)
    ↓
TF-IDF Vectorization
    ↓
CatBoost Classifier → Emotion + Confidence Score
    ↓
ChromaDB Semantic Search (Sentence Transformers)
    ↓
Groq LLM (llama-3.3-70b-versatile) → Empathetic Response
    ↓
Streamlit UI
```

---

## Knowledge Base

The RAG pipeline retrieves from a curated set of `.txt` files in `knowledge_base/`. To add or update content, edit the `.txt` files and rebuild the vector store:

```bash
python scripts/build_vector_store.py --reset
```

---

## .gitignore Recommendations

Make sure your `.gitignore` includes:
```
.env
data/vector_store/
*.pkl        # optional — remove this line if you want to commit model files
```

---
