from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.tokenize import word_tokenize
import string
from flask_cors import CORS

nltk.download('punkt')
nltk.download('stopwords')

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": ["http://localhost:3000", "http://localhost:5173"]}})

MODEL_PATH = './model/svm_model.pkl'
VECTORIZER_PATH = './model/vectorizer.pkl'

# --- Load model and vectorizer at startup ---
model = None
vectorizer = None

def load_model():
    global model, vectorizer
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

# --- Preprocessing function ---
def preprocess_text(text):
    stemmer = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stopwords.words('english')]
    return ' '.join(text)

def is_gibberish(text):
    words = text.translate(str.maketrans('', '', string.punctuation)).split()
    meaningful_words = [word for word in words if word.isalpha()]
    return len(meaningful_words) < 3


# --- Model retraining logic ---
def retrain_model():
    real_path = './data/real_news.json'
    fake_path = './data/fake_news_sample.csv'

    with open(real_path, 'r', encoding='utf-8') as f:
        real_news = json.load(f)
    real_df = pd.DataFrame(real_news)
    real_df['label'] = 0

    fake_df = pd.read_csv(fake_path, encoding='ISO-8859-1')
    fake_df['label'] = 1

    df = pd.concat([real_df, fake_df]).sample(frac=1).reset_index(drop=True)
    df['text'] = (df['title'] + ' ' + df['description']).apply(preprocess_text)

    X = df['text']
    y = df['label']

    new_vectorizer = TfidfVectorizer()
    X_vec = new_vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

    new_model = SVC(kernel='linear', probability=True)
    new_model.fit(X_train, y_train)

    acc = accuracy_score(y_test, new_model.predict(X_test))

    # Save the updated model & vectorizer
    joblib.dump(new_model, MODEL_PATH)
    joblib.dump(new_vectorizer, VECTORIZER_PATH)
    
    print(f"✅ Retrained model with accuracy: {acc:.4f}")
    return acc

@app.route('/')
def home():
    return "🚩 Jay Shree Ram! Flask ML Server is running successfully."

# --- Flask Endpoints ---
@app.route('/predict' , methods=['GET'])
def predict_with_svm():
    model = './model/svm_model.pkl'
    model = joblib.load(model)
    vectorizer = joblib.load('./model/vectorizer.pkl')
    text = request.args.get('text')
    processed_text = preprocess_text(text)
    input_vector = vectorizer.transform([processed_text])
    
    prediction = model.predict(input_vector)[0]
    if not text:
        return jsonify({"error": "No text provided"}), 400
    else:
        return jsonify({"prediction": str(prediction)})


# @app.route('/predict', methods=['GET'])
# def predict_with_svm():
#     global model, vectorizer

#     text = request.args.get('text', '')

#     if not text.strip():
#         return jsonify({"error": "Missing or empty 'text' parameter."}), 400

#     if is_gibberish(text):
#         return jsonify({"error": "Input text too short or nonsensical for prediction."}), 400

#     processed_text = preprocess_text(text)
#     input_vector = vectorizer.transform([processed_text])
#     prediction = model.predict(input_vector)[0]

#     try:
#         proba = model.predict_proba(input_vector)[0]
#         confidence = max(proba)
#     except AttributeError:
#         confidence = 1.0

#     if confidence < 0.7:
#         return jsonify({
#             "prediction": "Uncertain",
#             "confidence": round(confidence, 3)
#         }), 200

#     return jsonify({
#         "prediction": "Fake" if prediction == 1 else "Real",
#         "confidence": round(confidence, 3)
#     })

@app.route('/retrain', methods=['POST'])
def retrain():
    try:
        acc = retrain_model()
        load_model()  # Reload the updated model
        return jsonify({'success': True, 'accuracy': round(acc, 4)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# --- Startup ---
if __name__ == '__main__':
    load_model()
    port = int(os.environ.get("PORT", 5001))
    print(f"🚀 Flask ML Server running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
