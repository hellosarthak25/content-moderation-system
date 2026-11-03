"""
moderation_service.py
Model Serving & Policy Decision Engine for Content Moderation.
Uses TF-IDF + Logistic Regression baseline model for cloud deployment.
(DistilBERT fine-tuned model available locally - see README)
"""

import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import joblib
from scipy.sparse import hstack

from text_cleaner import clean_text
import database

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

# ----------------------------------------------------------------
# Auto-download baseline model from Hugging Face Hub if not present
# ----------------------------------------------------------------
BASELINE_PATH = "./baseline_model.joblib"
HF_REPO = "hellosarthak25/content-moderation-distilbert"

if not os.path.exists(BASELINE_PATH):
    print("Downloading baseline model from Hugging Face Hub...")
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id=HF_REPO, filename="baseline_model.joblib", local_dir=".")
    print("Baseline model downloaded successfully!")

# ----------------------------------------------------------------
# Load model artifacts (TF-IDF vectorizers + Logistic Regression)
# ----------------------------------------------------------------
print(f"Loading Content Moderation baseline model...")
artifacts = joblib.load(BASELINE_PATH)
vectorizers = artifacts["vectorizers"]
models = artifacts["models"]
print("Model loaded successfully!")


def evaluate_policy(scores: dict) -> tuple[str, str]:
    """
    Applies safety rules and business policies to raw toxicity probabilities.

    Verdicts:
    - AUTO_BLOCKED: High confidence on severe harm (threat, severe toxicity, hate speech)
    - FLAGGED_FOR_REVIEW: Borderline toxicity needing human moderator inspection
    - APPROVED: Safe and civil content
    """
    # 1. Critical safety triggers (instant block)
    if scores.get("threat", 0.0) >= 0.50:
        return "AUTO_BLOCKED", "threat"
    if scores.get("severe_toxic", 0.0) >= 0.50:
        return "AUTO_BLOCKED", "severe_toxic"
    if scores.get("identity_hate", 0.0) >= 0.60:
        return "AUTO_BLOCKED", "identity_hate"
    if scores.get("toxic", 0.0) >= 0.75:
        return "AUTO_BLOCKED", "toxic"

    # 2. Borderline / Moderate triggers (queue for human review)
    for label in ["toxic", "insult", "obscene", "identity_hate"]:
        if scores.get(label, 0.0) >= 0.40:
            return "FLAGGED_FOR_REVIEW", label

    # 3. Clean content
    return "APPROVED", "clean"


def moderate_comment(text: str, log_to_db: bool = True) -> dict:
    """
    Moderates a single piece of text:
    1. Preprocesses/cleans text
    2. Extracts TF-IDF features
    3. Runs inference through Logistic Regression classifiers
    4. Evaluates moderation policy
    5. Records entry in SQLite database
    """
    t0 = time.time()

    cleaned = clean_text(text)
    if not cleaned:
        cleaned = "empty_comment"

    # Extract TF-IDF features (word + character n-grams)
    feat_word = vectorizers["word"].transform([cleaned])
    feat_char = vectorizers["char"].transform([cleaned])
    X = hstack([feat_word, feat_char]).tocsr()

    # Predict probability for each toxicity label
    scores = {}
    for col in LABEL_COLUMNS:
        prob = models[col].predict_proba(X)[0, 1]
        scores[col] = float(prob)

    # Decision Engine
    verdict, primary_category = evaluate_policy(scores)
    latency_ms = (time.time() - t0) * 1000

    result = {
        "text": text,
        "cleaned_text": cleaned,
        "verdict": verdict,
        "primary_category": primary_category,
        "scores": {k: round(v, 4) for k, v in scores.items()},
        "latency_ms": round(latency_ms, 2),
    }

    # Save to SQLite database
    if log_to_db:
        db_record = database.log_comment(
            comment_text=text,
            verdict=verdict,
            primary_category=primary_category,
            scores=scores,
            latency_ms=latency_ms
        )
        result["log_id"] = db_record["id"]
        result["timestamp"] = db_record["timestamp"]

    return result
