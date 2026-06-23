# NLP Text Classification: Question vs Declarative Sentence

A comparative NLP study that benchmarks multiple text vectorization techniques and machine learning classifiers on the task of distinguishing **questions** from **declarative sentences**.

---

## 🎯 Problem Statement

Given a sentence, automatically classify it as either:
- `1` → Question
- `0` → Declarative sentence

This is a foundational NLP task with applications in chatbot routing, search query understanding, and dialogue systems.

---

## 🔬 What This Project Covers

### Vectorization Methods Compared
| Method | Description |
|--------|-------------|
| **TF-IDF** | Term Frequency–Inverse Document Frequency |
| **Bag of Words** | Count-based vectorization |
| **LSA** | Latent Semantic Analysis (TruncatedSVD on BoW) |
| **LDA** | Latent Dirichlet Allocation (topic modelling) |
| **Doc2Vec** | Paragraph-level neural embeddings (Gensim) |
| **Word2Vec** | Word-level neural embeddings averaged to sentence vectors |

### Classifiers Evaluated
- Logistic Regression
- Naive Bayes (MultinomialNB — used with BoW and TF-IDF only, as it cannot handle negative values)
- Random Forest
- Decision Tree
- SVM
- LSTM (Keras/TensorFlow)

### Evaluation Metrics
- Accuracy score per vectorizer × classifier combination
- ROC curves with AUC scores (all classifiers overlaid)
- Confusion matrices

---

## 📁 Repository Structure

```
nlp-text-classifier/
│
├── notebooks/
│   └── text_classification.ipynb   # Full pipeline: EDA → vectorization → modelling → evaluation
│
├── src/
│   ├── preprocessing.py            # Text cleaning utilities
│   └── app.py                      # Flask API for real-time inference
│
├── data/
│   └── README.md                   # Dataset info and source
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/asrithvarma/nlp-text-classifier.git
cd nlp-text-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add the dataset
Download the dataset (see `data/README.md`) and place it at `data/dataset.csv`.

### 4. Run the notebook
```bash
jupyter notebook notebooks/text_classification.ipynb
```

### 5. Run the Flask API (optional)
```bash
python src/app.py
```
Then send a POST request:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "What is the capital of France?"}'
```

---

## 📊 Key Results

| Vectorizer | Best Classifier | Accuracy |
|------------|----------------|----------|
| TF-IDF     | Logistic Regression | ~94% |
| LSA        | Logistic Regression | ~92% |
| BoW        | Random Forest       | ~93% |
| Doc2Vec    | SVM                 | ~89% |
| LDA        | Logistic Regression | ~87% |

> Results are approximate — run the notebook to reproduce exact numbers with the dataset.

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **NLP:** NLTK, spaCy, Gensim (Word2Vec, Doc2Vec)
- **ML:** scikit-learn (LR, NB, RF, DT, SVM), TensorFlow/Keras (LSTM)
- **Vectorization:** CountVectorizer, TfidfVectorizer, TruncatedSVD, LDA
- **Visualization:** Matplotlib, Seaborn
- **API:** Flask

---

## 💡 Why This Is Interesting

Most NLP tutorials pick one vectorizer and one model. This project systematically cross-evaluates **6 vectorization strategies × 5 classifiers = 30 combinations**, making it a comprehensive benchmark rather than a one-shot demo. Key insight: TF-IDF with Logistic Regression is surprisingly competitive with more complex approaches (Doc2Vec + LSTM) at a fraction of the compute cost.

---

## 📂 Dataset

The dataset contains labelled sentences (`question` / `sentence`) sourced from public NLP benchmark data. See `data/README.md` for details and download link.

---

## 👤 Author

**Samanthapudi Asrith Varma**  
BS-MS Physics + Data Science Minor, IISER Mohali  
[LinkedIn](https://www.linkedin.com/in/samanthapudi-asrith-varma-577068263) · [GitHub](https://github.com/asrithvarma)
