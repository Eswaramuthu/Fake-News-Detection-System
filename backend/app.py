import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import re
import string
import pickle

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# ============================================
# 1. LOAD DATASET
# ============================================
print("="*60)
print("FAKE NEWS DETECTION SYSTEM - ARTIFICIAL NEURAL NETWORK")
print("="*60)
print("\nLoading dataset...")

dataset_path = r'C:\Users\Dell\Desktop\ANN\Fake-News-Detection-System\dataset\fake.csv'
df = pd.read_csv(dataset_path)

print(f"Dataset loaded successfully!")
print(f"Total articles: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# ============================================
# 2. IMPROVED DATA PREPROCESSING
# ============================================
print("\nPreprocessing data...")

def preprocess_text(text):
    """Clean and preprocess text data - IMPROVED VERSION"""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# Convert label column to binary: real=0, fake=1
df['binary_label'] = df['label'].apply(lambda x: 0 if str(x).lower() == 'real' else 1)

print(f"\nLabel distribution:")
print(f"Real News (0): {(df['binary_label'] == 0).sum()}")
print(f"Fake News (1): {(df['binary_label'] == 1).sum()}")

# Combine title and text with better handling
df['content'] = (df['title'].fillna('') + ' ' + df['text'].fillna('')).apply(preprocess_text)

# Remove empty content
df = df[df['content'].str.len() > 50].reset_index(drop=True)

print(f"Articles after filtering: {len(df)}")
print(f"Preprocessing complete!")

# Print sample to verify data quality
print("\nSample article (Real News):")
sample_real = df[df['binary_label'] == 0].iloc[0]
print(f"Title: {sample_real['title'][:100]}")
print(f"Content preview: {sample_real['content'][:200]}...")

print("\nSample article (Fake News):")
sample_fake = df[df['binary_label'] == 1].iloc[0]
print(f"Title: {sample_fake['title'][:100]}")
print(f"Content preview: {sample_fake['content'][:200]}...")

# ============================================
# 3. IMPROVED TOKENIZATION
# ============================================
print("\nTokenizing text...")

MAX_WORDS = 20000  # Increased vocabulary
MAX_LEN = 300      # Increased sequence length

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(df['content'])

sequences = tokenizer.texts_to_sequences(df['content'])
X = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')
y = df['binary_label'].values

print(f"Tokenization complete!")
print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Vocabulary size: {len(tokenizer.word_index)}")

# ============================================
# 4. TRAIN-TEST SPLIT
# ============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# ============================================
# 5. BUILD IMPROVED NEURAL NETWORK
# ============================================
print("\n" + "="*60)
print("BUILDING IMPROVED ARTIFICIAL NEURAL NETWORK")
print("="*60)

model = Sequential([
    # Embedding layer
    Embedding(input_dim=MAX_WORDS, output_dim=128),
    SpatialDropout1D(0.2),
    
    # Bidirectional LSTM layers
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
    Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2)),
    
    # Dense layers
    Dense(64, activation='relu'),
    Dropout(0.5),
    
    Dense(32, activation='relu'),
    Dropout(0.3),
    
    # Output layer
    Dense(1, activation='sigmoid')
])

# Use a better optimizer with learning rate scheduling
optimizer = keras.optimizers.Adam(learning_rate=0.001)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("\nModel Architecture:")
model.summary()

# ============================================
# 6. TRAINING WITH IMPROVED CALLBACKS
# ============================================
print("\n" + "="*60)
print("TRAINING THE MODEL")
print("="*60)

EPOCHS = 10
BATCH_SIZE = 64

# Better callbacks
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

print(f"\nTraining Parameters:")
print(f"  - Epochs: {EPOCHS}")
print(f"  - Batch Size: {BATCH_SIZE}")
print(f"  - Training Samples: {len(X_train)}")
print(f"  - Validation Samples: {len(X_val)}")

print("\nTraining in progress...")
print("-"*60)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)

# ============================================
# 7. COMPREHENSIVE MODEL EVALUATION
# ============================================
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Evaluate on test set
test_results = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_results[1]*100:.2f}%")
print(f"Test Loss: {test_results[0]:.4f}")
print(f"Test Precision: {test_results[2]*100:.2f}%")
print(f"Test Recall: {test_results[3]*100:.2f}%")

# Get predictions
y_pred_prob = model.predict(X_test, verbose=0)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Real News', 'Fake News']))

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(f"True Negatives (Real as Real): {cm[0][0]}")
print(f"False Positives (Real as Fake): {cm[0][1]}")
print(f"False Negatives (Fake as Real): {cm[1][0]}")
print(f"True Positives (Fake as Fake): {cm[1][1]}")

# Test on sample data with original titles
print("\n" + "="*60)
print("SAMPLE PREDICTIONS")
print("="*60)

test_samples = 10
sample_indices = np.random.choice(len(X_test), test_samples, replace=False)

for i, idx in enumerate(sample_indices, 1):
    # Get original data index
    orig_idx = df.index[X_test.shape[0] + X_val.shape[0] + idx]
    
    # Get original title
    orig_title = df.iloc[orig_idx]['title']
    true_label = "FAKE" if y_test[idx] == 1 else "REAL"
    
    # Predict
    pred_prob = model.predict(X_test[idx:idx+1], verbose=0)[0][0]
    pred_label = "FAKE" if pred_prob > 0.5 else "REAL"
    
    match = "✓" if pred_label == true_label else "✗"
    
    print(f"\n{i}. {match} {orig_title[:80]}...")
    print(f"   True: {true_label} | Predicted: {pred_label} | Confidence: {pred_prob:.4f}")

# ============================================
# 8. SAVE MODEL AND TOKENIZER
# ============================================
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

model.save('fake_news_detector_ann.keras')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

config = {
    'MAX_WORDS': MAX_WORDS,
    'MAX_LEN': MAX_LEN
}
with open('config.pkl', 'wb') as f:
    pickle.dump(config, f)

print("\nModel saved as 'fake_news_detector_ann.keras'")
print("Tokenizer saved as 'tokenizer.pkl'")
print("Configuration saved as 'config.pkl'")

# ============================================
# 9. IMPROVED PREDICTION FUNCTION
# ============================================
def predict_news(text, model, tokenizer, max_len):
    """Predict if a news article is fake or real"""
    # Preprocess
    processed_text = preprocess_text(text)
    
    # Check if text is too short
    if len(processed_text.split()) < 5:
        return "INSUFFICIENT TEXT", 0.0
    
    # Tokenize
    sequence = tokenizer.texts_to_sequences([processed_text])
    
    # Pad
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded, verbose=0)[0][0]
    
    if prediction > 0.5:
        label = "FAKE NEWS"
        confidence = prediction * 100
    else:
        label = "REAL NEWS"
        confidence = (1 - prediction) * 100
    
    return label, confidence

# ============================================
# 10. INTERACTIVE PREDICTION SYSTEM
# ============================================
print("\n" + "="*60)
print("FAKE NEWS DETECTION SYSTEM - READY!")
print("="*60)

def run_detection():
    """Run interactive fake news detection"""
    while True:
        print("\n" + "-"*60)
        print("Enter news text to check (or type 'exit' to quit):")
        print("-"*60)
        
        user_input = input("\nNews Text: ").strip()
        
        if user_input.lower() == 'exit':
            print("\n" + "="*60)
            print("Thank you for using Fake News Detection System!")
            print("="*60)
            break
        
        if len(user_input) < 20:
            print("\nWarning: Please enter a longer text (at least 20 characters)")
            continue
        
        # Make prediction
        label, confidence = predict_news(user_input, model, tokenizer, MAX_LEN)
        
        if label == "INSUFFICIENT TEXT":
            print("\nError: Text is too short for meaningful analysis.")
            continue
        
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        
        if label == "FAKE NEWS":
            print(f"\nResult: {label}")
            print(f"Confidence: {confidence:.2f}%")
            print("\nThis news article appears to be FAKE.")
            print("Recommendation: Verify from trusted sources.")
        else:
            print(f"\nResult: {label}")
            print(f"Confidence: {confidence:.2f}%")
            print("\nThis news article appears to be REAL.")
            print("Recommendation: Cross-check with multiple sources.")
        
        print("="*60)

# Run the interactive system
run_detection()