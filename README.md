# 🛡️ Automated Content Moderation System

An end-to-end, multi-label AI content moderation web system built with fine-tuned **DistilBERT**, **FastAPI**, and **SQLAlchemy**.

The system detects and categorizes comments across 6 distinct toxicity behaviors:
1. `toxic`
2. `severe_toxic`
3. `obscene`
4. `threat`
5. `insult`
6. `identity_hate`

---

## 🚀 Quick Start (Running the App)

### 1. Setup Environment
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Web Application
```bash
python app.py
```
Open **`http://localhost:8000`** in your browser to interact with the live moderation dashboard!

---

## 📁 Project Structure

```
├── app.py                         # Production FastAPI web server & interactive UI
├── moderation_service.py          # Model inference & 3-tier moderation policy engine
├── database.py                    # SQLAlchemy SQLite audit logging layer
├── text_cleaner.py                # Regex-based text cleaning module
├── requirements.txt               # Python package dependencies
│
├── models/
│   ├── distilbert_moderation/     # Fine-tuned DistilBERT weights & tokenizer
│   └── baseline_model.joblib      # Stage 3 Baseline (TF-IDF + Logistic Regression)
│
├── stage1_eda.py                  # Stage 1: Exploratory Data Analysis & stats
├── stage2_clean.py                # Stage 2: Text preprocessing & stratified splitting
├── stage3_baseline.py             # Stage 3: TF-IDF + Logistic Regression benchmark
├── stage4_transformer.py          # Stage 4: DistilBERT fine-tuning script
└── eda_distribution.png           # Stage 1: Label distribution & correlation chart
```

---

## 🏗️ System Architecture & Workflow

1. **Stage 1: Exploratory Data Analysis (`stage1_eda.py`)**
   * Inspects class imbalance (89.8% clean vs. 10.2% toxic) and co-occurrence patterns across all 6 labels.
2. **Stage 2: Text Preprocessing (`stage2_clean.py`)**
   * Cleans raw text, expands contractions, normalizes repeated characters, and generates stratified train/val splits.
3. **Stage 3: Classical ML Baseline (`stage3_baseline.py`)**
   * Extracts word & char n-gram TF-IDF features and trains 6 Logistic Regression classifiers to establish the benchmark score.
4. **Stage 4: Transformer Fine-Tuning (`stage4_transformer.py`)**
   * Fine-tunes `distilbert-base-uncased` with `BCEWithLogitsLoss` for contextual multi-label classification.
5. **Stage 5: Production Web Service (`app.py`)**
   * Serves an interactive dashboard with animated toxicity meters, automated moderation policy actions (`APPROVED`, `FLAGGED_FOR_REVIEW`, `AUTO_BLOCKED`), and SQLite database audit logging.
