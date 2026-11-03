"""
moderation_service.py
Model Serving & Policy Decision Engine for Content Moderation.
Loads fine-tuned DistilBERT and applies business safety policies.
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

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

# Locate trained model directory
POSSIBLE_PATHS = [
    "models/distilbert_moderation",
    "./distilbert_moderation",
]

MODEL_PATH = None
for path in POSSIBLE_PATHS:
    if os.path.exists(path) and (os.path.exists(os.path.join(path, "model.safetensors")) or os.path.exists(os.path.join(path, "pytorch_model.bin"))):
        MODEL_PATH = path
        break

if not MODEL_PATH:
    raise FileNotFoundError(
        "Could not find trained DistilBERT model. Looked in: " + ", ".join(POSSIBLE_PATHS)
    )

print(f"Loading Content Moderation model from: '{MODEL_PATH}'...")

# Use MPS (Apple Silicon GPU) if available, otherwise CPU
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Inference Device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()
print("✓ Model and tokenizer successfully loaded into memory!")


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
    2. Runs inference through DistilBERT
    3. Evaluates moderation policy
    4. Records entry in SQLite database
    """
    t0 = time.time()
    
    cleaned = clean_text(text)
    if not cleaned:
        cleaned = "empty_comment"
        
    inputs = tokenizer(
        cleaned,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        
    scores = {col: float(probs[i]) for i, col in enumerate(LABEL_COLUMNS)}
    
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
