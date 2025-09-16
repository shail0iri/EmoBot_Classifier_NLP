# 🤖 EmoBot Classifier – NLP Emotion Detection

An **end-to-end NLP project** for detecting emotions in text using **machine learning**.  
This repository covers the full pipeline — from preprocessing and model training to evaluation and an interactive **Streamlit web app**.

---

## 🚀 Features
- 📊 **Dataset Preprocessing** with spaCy & scikit-learn  
- 🔎 **Feature Engineering** using TF-IDF  
- 🧠 **Multiple ML Models** trained: Logistic Regression, Random Forest, LightGBM, XGBoost, CatBoost  
- 🏆 **CatBoost achieved ~95% accuracy** (best performer)  
- 🎨 **Visualizations**: emotion distribution, confusion matrix, word clouds  
- 🌐 **Streamlit App** for real-time emotion classification  
---
🔧 Tech Stack

Python 3.9+

NLP: spaCy, scikit-learn

Models: Logistic Regression, Random Forest, LightGBM, XGBoost, CatBoost

Visualization: Matplotlib, Seaborn, WordCloud

Deployment: Streamlit
---

## 📂 Project Structure
EmoBot_Classifier_NLP/
│── app/
│ └── app.py # Streamlit app
│
│── data/
│ └── Emotion_classify_Data.csv # Dataset
│
│── models/ # Saved ML models (.pkl files)
│
│── notebooks/
│ └── EmoBot_notebook.ipynb # Training notebook (95% accuracy)
│
│── requirements.txt # Dependencies
│── README.md

---

## ⚙️ Installation

Clone this repository:
```bash
git clone https://github.com/shail0iri/EmoBot_Classifier_NLP.git
cd EmoBot_Classifier_NLP

Install dependencies --
pip install -r requirements.txt

▶️ Usage
Run the Streamlit App

📊 Results

Best model: CatBoost

Accuracy: 95%

Macro F1-score: 0.94

