"""
Flask Application — NLP Sentiment Analysis Dashboard
=====================================================
Serves the dashboard and provides REST API endpoints
for statistics, predictions, and batch processing.
"""

import os
import json
import re
import string
import io
import csv

import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request, jsonify, Response
from spellchecker import SpellChecker

# Initialize spellchecker and load domain-specific words
spell = SpellChecker()
spell.word_frequency.load_words([
    "app", "apps", "teams", "microsoft", "chat", "meeting", "meetings", 
    "android", "ios", "zoom", "skype", "wifi", "internet", "notification", "notifications"
])

# Custom slang mapping
CUSTOM_WORDS = {
    "thiz": "this",
    "iz": "is",
    "zuckk": "suck",
    "zuck": "suck",
    "zuckin": "sucking",
    "zucks": "sucks",
    "suckk": "suck",
    "sux": "suck",
}

# ── Initialize Flask ───────────────────────────────────────────────────────────
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "reviews.csv")
MODEL_FILE = os.path.join(BASE_DIR, "models", "model.pkl")
TFIDF_FILE = os.path.join(BASE_DIR, "models", "tfidf.pkl")
METRICS_FILE = os.path.join(BASE_DIR, "models", "metrics.json")
KEYWORDS_FILE = os.path.join(BASE_DIR, "models", "keywords.json")


# ── Load Resources ─────────────────────────────────────────────────────────────
def load_resources():
    """Load model, vectorizer, data, and metrics."""
    resources = {}

    # Load dataset
    if os.path.exists(DATA_FILE):
        resources["df"] = pd.read_csv(DATA_FILE)
        print(f"✅ Loaded dataset: {len(resources['df']):,} reviews")
    else:
        resources["df"] = pd.DataFrame()
        print("⚠️  Dataset not found!")

    # Load model
    if os.path.exists(MODEL_FILE):
        resources["model"] = joblib.load(MODEL_FILE)
        print("✅ Loaded model")
    else:
        resources["model"] = None
        print("⚠️  Model not found!")

    # Load TF-IDF vectorizer
    if os.path.exists(TFIDF_FILE):
        resources["tfidf"] = joblib.load(TFIDF_FILE)
        print("✅ Loaded TF-IDF vectorizer")
    else:
        resources["tfidf"] = None
        print("⚠️  TF-IDF vectorizer not found!")

    # Load metrics
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE, "r") as f:
            resources["metrics"] = json.load(f)
        print("✅ Loaded metrics")
    else:
        resources["metrics"] = {}
        print("⚠️  Metrics not found!")

    # Load keywords
    if os.path.exists(KEYWORDS_FILE):
        with open(KEYWORDS_FILE, "r") as f:
            resources["keywords"] = json.load(f)
        print("✅ Loaded keywords")
    else:
        resources["keywords"] = {}
        print("⚠️  Keywords not found!")

    # Download NLTK resources
    import tempfile
    nltk_dir = os.path.join(tempfile.gettempdir(), 'nltk_data')
    os.makedirs(nltk_dir, exist_ok=True)
    if nltk_dir not in nltk.data.path:
        nltk.data.path.append(nltk_dir)
        
    for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        nltk.download(resource, download_dir=nltk_dir, quiet=True)

    resources["stop_words"] = set(stopwords.words("english"))
    resources["lemmatizer"] = WordNetLemmatizer()

    return resources


# Load resources on startup
res = load_resources()


# ── Text Preprocessing ────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Apply the same preprocessing pipeline used during training."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = word_tokenize(text)
    
    # spelling & slang correction
    corrected_tokens = []
    for t in tokens:
        if t in CUSTOM_WORDS:
            corrected_tokens.append(CUSTOM_WORDS[t])
        elif t in spell:
            corrected_tokens.append(t)
        else:
            corrected = spell.correction(t)
            corrected_tokens.append(corrected if corrected else t)
    tokens = corrected_tokens
    
    tokens = [t for t in tokens if t not in res["stop_words"]]
    tokens = [res["lemmatizer"].lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 1]
    return " ".join(tokens)


def predict_sentiment(text: str) -> dict:
    """
    Predict sentiment for a single text.

    Returns:
        Dictionary with sentiment, confidence, and probabilities.
    """
    if res["model"] is None or res["tfidf"] is None:
        return {"error": "Model not loaded"}

    cleaned = clean_text(text)

    if not cleaned.strip():
        return {
            "original_text": text,
            "cleaned_text": "",
            "sentiment": "Neutral",
            "confidence": 0.0,
            "probabilities": {"Negative": 0.33, "Neutral": 0.34, "Positive": 0.33},
        }

    features = res["tfidf"].transform([cleaned])
    prediction = res["model"].predict(features)[0]

    # Get probabilities
    if hasattr(res["model"], "predict_proba"):
        proba = res["model"].predict_proba(features)[0]
        classes = res["model"].classes_
        prob_dict = {cls: round(float(p) * 100, 2) for cls, p in zip(classes, proba)}
        confidence = round(float(max(proba)) * 100, 2)
    else:
        prob_dict = {prediction: 100.0}
        confidence = 100.0

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "sentiment": prediction,
        "confidence": confidence,
        "probabilities": prob_dict,
    }


# ── Page Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    """Render the main dashboard page."""
    return render_template("dashboard.html")


@app.route("/checker")
def sentiment_checker():
    """Render the sentiment checker page."""
    return render_template("sentiment_checker.html")


# ── API Routes ─────────────────────────────────────────────────────────────────
@app.route("/api/statistics")
def api_statistics():
    """Return dataset KPI statistics."""
    df = res["df"]

    if df.empty:
        return jsonify({"error": "No data available"})

    stats = {
        "total_reviews": int(len(df)),
        "total_positive": int(len(df[df["sentiment"] == "Positive"])),
        "total_neutral": int(len(df[df["sentiment"] == "Neutral"])),
        "total_negative": int(len(df[df["sentiment"] == "Negative"])),
        "average_rating": round(float(df["rating"].mean()), 2),
    }

    return jsonify(stats)


@app.route("/api/sentiment-distribution")
def api_sentiment_distribution():
    """Return sentiment distribution for pie chart."""
    df = res["df"]

    if df.empty:
        return jsonify({"error": "No data available"})

    distribution = df["sentiment"].value_counts().to_dict()

    return jsonify({
        "labels": list(distribution.keys()),
        "values": list(distribution.values()),
    })


@app.route("/api/rating-distribution")
def api_rating_distribution():
    """Return rating distribution for bar chart."""
    df = res["df"]

    if df.empty:
        return jsonify({"error": "No data available"})

    rating_counts = df["rating"].value_counts().sort_index().to_dict()

    # Ensure all ratings 1-5 are present
    full_distribution = {i: rating_counts.get(i, 0) for i in range(1, 6)}

    return jsonify({
        "labels": [f"{k} Star" for k in full_distribution.keys()],
        "values": list(full_distribution.values()),
    })


@app.route("/api/model-performance")
def api_model_performance():
    """Return model evaluation metrics."""
    metrics = res["metrics"]

    if not metrics:
        return jsonify({"error": "No metrics available"})

    best_model = metrics.get("best_model", "Unknown")
    best_metrics = metrics.get("models", {}).get(best_model, {})

    # All models comparison
    all_models = []
    for name, m in metrics.get("models", {}).items():
        all_models.append({
            "name": name,
            "accuracy": m.get("accuracy", 0),
            "precision": m.get("precision", 0),
            "recall": m.get("recall", 0),
            "f1_score": m.get("f1_score", 0),
            "is_best": name == best_model,
        })

    return jsonify({
        "best_model": best_model,
        "accuracy": best_metrics.get("accuracy", 0),
        "precision": best_metrics.get("precision", 0),
        "recall": best_metrics.get("recall", 0),
        "f1_score": best_metrics.get("f1_score", 0),
        "all_models": all_models,
        "confusion_matrix_url": "/static/images/confusion_matrix.png",
    })


@app.route("/api/top-keywords")
def api_top_keywords():
    """Return top keywords for positive and negative sentiments."""
    keywords = res["keywords"]

    if not keywords:
        return jsonify({"error": "No keywords available"})

    return jsonify(keywords)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict sentiment for a single text input."""
    data = request.get_json()

    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text provided"}), 400

    result = predict_sentiment(text)
    return jsonify(result)


@app.route("/api/batch-predict", methods=["POST"])
def api_batch_predict():
    """Predict sentiment for multiple reviews (batch)."""
    data = request.get_json()

    if not data or "reviews" not in data:
        return jsonify({"error": "No reviews provided"}), 400

    reviews = data["reviews"]

    if not isinstance(reviews, list) or len(reviews) == 0:
        return jsonify({"error": "Invalid reviews format"}), 400

    results = []
    for review_text in reviews:
        text = str(review_text).strip()
        if text:
            result = predict_sentiment(text)
            results.append(result)

    return jsonify({"results": results, "total": len(results)})


@app.route("/api/batch-predict-csv", methods=["POST"])
def api_batch_predict_csv():
    """Return batch prediction results as downloadable CSV."""
    data = request.get_json()

    if not data or "reviews" not in data:
        return jsonify({"error": "No reviews provided"}), 400

    reviews = data["reviews"]
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["review", "cleaned_text", "sentiment", "confidence", "positive_prob", "neutral_prob", "negative_prob"])

    for review_text in reviews:
        text = str(review_text).strip()
        if text:
            result = predict_sentiment(text)
            probs = result.get("probabilities", {})
            writer.writerow([
                result.get("original_text", ""),
                result.get("cleaned_text", ""),
                result.get("sentiment", ""),
                result.get("confidence", 0),
                probs.get("Positive", 0),
                probs.get("Neutral", 0),
                probs.get("Negative", 0),
            ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_predictions.csv"},
    )


# ── Run Server ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 Starting NLP Sentiment Analysis Dashboard...")
    print("   Dashboard : http://localhost:5000")
    print("   Checker   : http://localhost:5000/checker")
    print("=" * 50)
    app.run(debug=True, host="0.0.0.0", port=5000)
