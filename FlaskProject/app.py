from flask import Flask, render_template, request
import pickle
import re
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ==========================
# Load Model and Vectorizer
# ==========================
model = load_model("fake_news_detector_ann_fast.keras")

with open("tokenizer_fast.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ==========================
# Suspicious words
# ==========================
SUSPICIOUS_WORDS = [
    'shocking', 'breaking', 'exclusive', 'viral', 'unbelievable',
    'clickbait', 'alert', 'fake', 'exposed', 'secret'
]

# ==========================
# Preprocessing
# ==========================
def preprocess_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\!\?]', '', text)
    return ' '.join(text.split())

# ==========================
# Prediction function
# ==========================
def predict_news(text):
    processed = preprocess_text(text)
    if len(processed.split()) < 5:
        return "INSUFFICIENT TEXT", 0.0

    vector = vectorizer.transform([processed]).toarray()
    pred = model.predict(vector)[0][0]

    # Map: 0 = FAKE, 1 = REAL
    label = "REAL NEWS" if round(pred) == 1 else "FAKE NEWS"
    confidence = pred*100 if round(pred)==1 else (1-pred)*100

    return label, confidence

# ==========================
# Suspicious words detection
# ==========================
def get_suspicious_words(text):
    words = text.lower().split()
    return [w for w in words if w in SUSPICIOUS_WORDS]

# ==========================
# Flask Route
# ==========================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    suspicious_words = []
    news_text = ""

    if request.method == "POST":
        news_text = request.form.get("news_text", "")
        prediction, confidence = predict_news(news_text)
        if prediction != "INSUFFICIENT TEXT":
            suspicious_words = get_suspicious_words(news_text)

    return render_template(
        "index.html",
        news_text=news_text,
        prediction=prediction,
        confidence=confidence,
        suspicious_words=suspicious_words
    )

if __name__ == "__main__":
    app.run(debug=True)
