"""
app.py
------
Flask REST API for real-time question vs sentence classification.

Usage:
    python src/app.py

Endpoint:
    POST /predict
    Body: {"text": "Is this a question?"}
    Response: {"text": "Is this a question?", "prediction": "question", "label": 1}

Note:
    The model and vectorizer must be trained and saved first (run the notebook
    and save them with joblib). See README for instructions.
"""

import os
import joblib
from flask import Flask, request, jsonify
from preprocessing import clean_sentence

app = Flask(__name__)

# ── Load model and vectorizer ────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'logreg_tfidf.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.pkl')

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    model = None
    vectorizer = None
    print("Warning: model files not found. Run the notebook and save models first.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'model_loaded': model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict whether input text is a question or declarative sentence.

    Request body (JSON):
        {"text": "Your sentence here"}

    Response (JSON):
        {
            "text": "Your sentence here",
            "prediction": "question" | "sentence",
            "label": 1 | 0
        }
    """
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded. Run the notebook and save models first.'}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Request body must contain a "text" field.'}), 400

    raw_text = data['text']
    cleaned = clean_sentence(raw_text)
    vectorized = vectorizer.transform([cleaned])
    label = int(model.predict(vectorized)[0])
    prediction = 'question' if label == 1 else 'sentence'

    return jsonify({
        'text': raw_text,
        'prediction': prediction,
        'label': label
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
