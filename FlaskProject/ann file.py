import pandas as pd
import numpy as np
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import pickle

# ==========================
# Load Dataset
# ==========================
dataset_path = r'/Users/sendhanumapathy/PycharmProjects/ANN PROJECT/fake (1).csv'
df = pd.read_csv(dataset_path)

# Binary label: 0 = FAKE, 1 = REAL
df['binary_label'] = df['label'].apply(lambda x: 1 if str(x).lower() == 'real' else 0)

# ==========================
# Preprocessing
# ==========================
def preprocess_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\!\?]', '', text)
    return ' '.join(text.split())

df['content'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).apply(preprocess_text)
df = df[df['content'].str.len() > 50].reset_index(drop=True)

# ==========================
# TF-IDF Vectorizer
# ==========================
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2))
X = vectorizer.fit_transform(df['content']).toarray()
y = df['binary_label'].values

# Save vectorizer
with open("tokenizer_fast.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# ==========================
# Train-Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ==========================
# Class Weights
# ==========================
weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {0: weights[0], 1: weights[1]}

# ==========================
# Build & Train Model
# ==========================
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    class_weight=class_weights,
    verbose=1
)

# Save model
model.save("fake_news_detector_ann_fast.keras")
print("\nModel trained and saved successfully!")

# ==========================
# Suspicious words
# ==========================
SUSPICIOUS_WORDS = ['shocking', 'breaking', 'exclusive', 'viral', 'unbelievable',
                    'clickbait', 'alert', 'fake', 'exposed', 'secret']

# ==========================
# Prediction function
# ==========================
def predict_news(text):
    processed = preprocess_text(text)
    if len(processed.split()) < 5:
        return "INSUFFICIENT TEXT", 0.0

    vector = vectorizer.transform([processed]).toarray()
    pred = model.predict(vector)[0][0]

    # 0 = FAKE, 1 = REAL
    label = "REAL NEWS" if round(pred) == 1 else "FAKE NEWS"
    confidence = pred*100 if round(pred) == 1 else (1-pred)*100

    return label, confidence

def get_suspicious_words(text):
    words = text.lower().split()
    return [w for w in words if w in SUSPICIOUS_WORDS]

# ==========================
# Interactive Terminal Input
# ==========================
print("\n=== FAKE NEWS DETECTION SYSTEM ===")
print("Type 'exit' to quit\n")

while True:
    user_input = input("Enter news text: ").strip()
    if user_input.lower() == 'exit':
        print("\nExiting system. Stay safe!")
        break

    label, confidence = predict_news(user_input)
    suspicious_words = get_suspicious_words(user_input)

    if label == "INSUFFICIENT TEXT":
        print("Text too short for meaningful analysis. Enter at least 5 words.\n")
        continue

    print(f"\nPrediction: {label}")
    print(f"Confidence: {confidence:.2f}%")
    if suspicious_words:
        print(f"Suspicious words detected: {', '.join(suspicious_words)}")
    print("\n" + "-"*50 + "\n")
