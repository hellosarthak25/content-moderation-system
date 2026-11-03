"""
STAGE 3: Baseline Model (TF-IDF + Logistic Regression)
Content Moderation System

"""

import os
import time
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

# Import the cleaner we built in Stage 2 (or use fallback)
try:
    from text_cleaner import clean_text
except ImportError:
    def clean_text(t):
        return str(t).lower()

# The 6 target toxicity categories
LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

def load_data():
    print("=" * 65)
    print("STEP 1: LOADING CLEANED DATA")
    print("=" * 65)
    
    train_path = "train_clean.csv"
    val_path = "val_clean.csv"
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Could not find '{train_path}' or '{val_path}'. "
            "Please make sure Stage 2 (stage2_clean.py) has been run first!"
        )
        
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Fill any accidental NaN text with placeholder
    train_df["clean_text"] = train_df["clean_text"].fillna("empty_comment").astype(str)
    val_df["clean_text"] = val_df["clean_text"].fillna("empty_comment").astype(str)
    
    print(f"✓ Training samples:   {len(train_df):,}")
    print(f"✓ Validation samples: {len(val_df):,}")
    
    return train_df, val_df

def extract_features(train_texts, val_texts):
    print("\n" + "=" * 65)
    print("STEP 2: EXTRACTING TF-IDF FEATURES (WORDS + CHARACTERS)")
    print("=" * 65)
    start_time = time.time()
    
    # 1. Word-level TF-IDF (1-word and 2-word combinations)
    print("• Extracting Word n-grams (1-2 words, top 30,000 features)...")
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=30000,
        sublinear_tf=True,
        strip_accents="unicode"
    )
    X_train_word = word_vectorizer.fit_transform(train_texts)
    X_val_word = word_vectorizer.transform(val_texts)
    
    # 2. Character-level TF-IDF (2-5 characters, catches typos, obfuscated curses like f**k)
    print("• Extracting Character n-grams (2-5 chars, top 40,000 features)...")
    char_vectorizer = TfidfVectorizer(
        ngram_range=(2, 5),
        analyzer="char",
        max_features=40000,
        sublinear_tf=True,
        strip_accents="unicode"
    )
    X_train_char = char_vectorizer.fit_transform(train_texts)
    X_val_char = char_vectorizer.transform(val_texts)
    
    # Combine word and character features side-by-side
    print("• Combining word and character feature matrices...")
    X_train = hstack([X_train_word, X_train_char]).tocsr()
    X_val = hstack([X_val_word, X_val_char]).tocsr()
    
    elapsed = time.time() - start_time
    print(f"✓ Features extracted in {elapsed:.1f}s | Shape: {X_train.shape[1]:,} features per comment")
    
    vectorizers = {
        "word": word_vectorizer,
        "char": char_vectorizer
    }
    return X_train, X_val, vectorizers

def train_and_evaluate(X_train, y_train_df, X_val, y_val_df):
    print("\n" + "=" * 65)
    print("STEP 3: TRAINING 6 LOGISTIC REGRESSION MODELS")
    print("=" * 65)
    
    models = {}
    roc_scores = {}
    val_predictions = pd.DataFrame(index=y_val_df.index)
    
    print(f"{'Category':15s} | {'Training Time':>13s} | {'ROC-AUC Score':>14s}")
    print("-" * 65)
    
    total_start = time.time()
    for col in LABEL_COLUMNS:
        t0 = time.time()
        
        # We use C=2.0 for balanced regularization and class_weight='balanced'
        # to help with rare categories like 'threat' and 'identity_hate'
        model = LogisticRegression(
            C=2.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        )
        
        # Train on this specific category
        model.fit(X_train, y_train_df[col])
        t_elapsed = time.time() - t0
        
        # Predict probabilities on validation set (0.0 to 1.0)
        probs = model.predict_proba(X_val)[:, 1]
        val_predictions[col] = probs
        
        # Calculate ROC-AUC score
        score = roc_auc_score(y_val_df[col], probs)
        roc_scores[col] = score
        models[col] = model
        
        print(f"{col:15s} | {t_elapsed:11.2f}s | {score:14.4f}")
        
    mean_roc_auc = np.mean(list(roc_scores.values()))
    print("-" * 65)
    print(f"{'MEAN ROC-AUC':15s} | {time.time()-total_start:11.2f}s | {mean_roc_auc:14.4f}")
    
    return models, roc_scores, mean_roc_auc

def save_artifacts(vectorizers, models, filename="baseline_model.joblib"):
    print("\n" + "=" * 65)
    print("STEP 4: SAVING MODEL ARTIFACTS")
    print("=" * 65)
    
    artifacts = {
        "vectorizers": vectorizers,
        "models": models,
        "labels": LABEL_COLUMNS
    }
    joblib.dump(artifacts, filename)
    size_mb = os.path.getsize(filename) / (1024 * 1024)
    print(f"✓ Model package saved to '{filename}' ({size_mb:.2f} MB)")

def predict_single_comment(comment_text, artifacts):
    """Predicts toxicity probabilities for any custom raw text."""
    vectorizers = artifacts["vectorizers"]
    models = artifacts["models"]
    
    # 1. Clean the incoming text
    cleaned = clean_text(comment_text)
    
    # 2. Extract features
    feat_word = vectorizers["word"].transform([cleaned])
    feat_char = vectorizers["char"].transform([cleaned])
    X = hstack([feat_word, feat_char]).tocsr()
    
    # 3. Predict probability for each label
    results = {}
    for col in LABEL_COLUMNS:
        prob = models[col].predict_proba(X)[0, 1]
        results[col] = float(prob)
        
    return results

def run_test_suite(artifacts):
    print("\n" + "=" * 65)
    print("STEP 5: LIVE DEMONSTRATION ON TEST SENTENCES")
    print("=" * 65)
    
    test_comments = [
        "Thank you so much for helping me edit this Wikipedia article! Have a great day.",
        "You are an absolute idiot and know nothing. Stop editing my pages.",
        "I know where you live and I am going to find you and kill you.",
        "This is complete nonsense and complete garbage information.",
    ]
    
    for text in test_comments:
        scores = predict_single_comment(text, artifacts)
        print(f"\nComment: \"{text}\"")
        is_toxic = any(score >= 0.50 for score in scores.values())
        status = "🚨 [FLAGGED / TOXIC]" if is_toxic else "✅ [CLEAN / APPROVED]"
        print(f"Status:  {status}")
        print("Scores:")
        for col, score in scores.items():
            bar = "█" * int(score * 20)
            flag = "⚠️" if score >= 0.50 else " "
            print(f"  {flag} {col:15s}: {score*100:5.1f}%  |{bar:<20}|")

if __name__ == "__main__":
    # 1. Load data
    train_df, val_df = load_data()
    
    # 2. Extract TF-IDF features
    X_train, X_val, vectorizers = extract_features(
        train_df["clean_text"],
        val_df["clean_text"]
    )
    
    # 3. Train models and get benchmark scores
    models, roc_scores, mean_auc = train_and_evaluate(
        X_train, train_df,
        X_val, val_df
    )
    
    # 4. Save model package
    save_artifacts(vectorizers, models, "baseline_model.joblib")
    
    # 5. Run live predictions on examples
    saved_artifacts = {
        "vectorizers": vectorizers,
        "models": models,
        "labels": LABEL_COLUMNS
    }
    run_test_suite(saved_artifacts)
    
    print("\n" + "=" * 65)
    print("🎉 STAGE 3 COMPLETE! Baseline Model Ready & Benchmarked.")
    print("=" * 65)
