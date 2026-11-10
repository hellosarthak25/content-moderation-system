# 🛡️ AI-Powered Content Moderation System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/DistilBERT-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" />
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" />
  <img src="https://img.shields.io/badge/Machine_Learning-FF6F00?style=for-the-badge&logo=scikitlearn&logoColor=white" />
</p>

<p align="center">
  <b>Making online spaces safer, one comment at a time.</b>
  <br/>
  An intelligent, AI-driven system that detects, classifies, and moderates toxic online comments using Natural Language Processing and Deep Learning.
</p>

<p align="center">
  <a href="https://content-moderation-system-nf9s.onrender.com/">
    <img src="https://img.shields.io/badge/🚀_EXPLORE_LIVE_PROJECT-6366F1?style=for-the-badge" />
  </a>
  <a href="https://github.com/hellosarthak25/content-moderation-system">
    <img src="https://img.shields.io/badge/VIEW_SOURCE_CODE-181717?style=for-the-badge&logo=github" />
  </a>
</p>

---

## 📌 Project Overview

The rapid growth of online communities has made it increasingly difficult to maintain safe and respectful digital environments. Millions of comments are shared across online platforms, and some contain toxic, abusive, or hateful language.

**AI-Powered Content Moderation System** addresses this challenge by using machine learning and Natural Language Processing (NLP) to automatically identify harmful language and assist in content moderation.

The system uses a fine-tuned **DistilBERT transformer model** to analyze comments and perform **multi-label toxicity classification** across six different categories.

Unlike a simple keyword-based filter, the model analyzes the context of text to identify different forms of toxic language. Based on its predictions and moderation policies, the system generates an appropriate moderation decision.

### 🎯 What Does the System Detect?

The system identifies six categories of toxic or harmful language:

<table>
  <tr>
    <td width="50%">
      <h3>🔴 01. Toxic</h3>
      General harmful, hostile, or disrespectful language that creates an unpleasant interaction.
    </td>
    <td width="50%">
      <h3>🚨 02. Severe Toxic</h3>
      Extremely aggressive, abusive, or highly harmful language.
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>🔞 03. Obscene</h3>
      Vulgar, explicit, or inappropriate expressions.
    </td>
    <td width="50%">
      <h3>⚠️ 04. Threat</h3>
      Language expressing intimidation, threats, or intentions to cause harm.
    </td>
  </tr>
  <tr>
    <td width="50%">
      <h3>💬 05. Insult</h3>
      Offensive, demeaning, or insulting statements directed towards others.
    </td>
    <td width="50%">
      <h3>🌐 06. Identity Hate</h3>
      Hateful or discriminatory language targeting individuals based on their identity.
    </td>
  </tr>
</table>

**Important:** These categories are independently evaluated. A single comment can belong to multiple categories at the same time.

---

## ⚡ Explore the Live Application

<p align="center">
  <a href="https://content-moderation-system-nf9s.onrender.com/">
    <img src="https://img.shields.io/badge/OPEN_LIVE_DASHBOARD-Visit_Application-6366F1?style=for-the-badge&logo=render&logoColor=white" />
  </a>
</p>

The application provides an interactive dashboard where users can:

- Submit comments for toxicity analysis.
- View predicted toxicity categories.
- Examine category-wise prediction scores.
- See automated moderation decisions.
- Explore moderation statistics and audit logs.

**Try it here:** https://content-moderation-system-nf9s.onrender.com/

---

## ✨ Key Features

| Feature | Description |
|:---|:---|
| 🧠 AI-Powered Detection | Uses fine-tuned DistilBERT for contextual toxicity classification. |
| 🏷️ Multi-Label Classification | Detects multiple toxicity categories in a single comment. |
| 📊 Prediction Scores | Displays category-wise toxicity scores. |
| ⚡ Automated Moderation | Generates moderation decisions based on prediction scores and policy rules. |
| 🖥️ Interactive Dashboard | Provides a clean interface for comment analysis and monitoring. |
| 📦 Batch Moderation | Supports processing multiple comments through the API. |
| 🗄️ Audit Logging | Stores moderation activity using SQLite and SQLAlchemy. |
| 📈 Moderation Analytics | Provides statistics about processed comments and moderation outcomes. |
| 🔌 REST API | Provides endpoints for moderation and retrieving system information. |

---

## 🏗️ How It Works

The system follows a complete machine learning pipeline, starting from data exploration and preprocessing to model training, prediction, and automated moderation.

```mermaid
flowchart TD
    A["💬 User Submits Comment"] --> B["Text Preprocessing"]
    B --> C["Fine-Tuned DistilBERT"]
    C --> D["Multi-Label Classification"]
    D --> E["Six Toxicity Prediction Scores"]
    E --> F["Moderation Policy Engine"]
    F --> G{"Moderation Decision"}
    G --> H["APPROVED"]
    G --> I["FLAGGED FOR REVIEW"]
    G --> J["AUTO BLOCKED"]
    H --> K["SQLite Audit Logs"]
    I --> K
    J --> K
    K --> L["Dashboard & Analytics"]

    style A fill:#334155,color:#fff
    style C fill:#4338ca,color:#fff
    style F fill:#6d28d9,color:#fff
    style H fill:#166534,color:#fff
    style I fill:#a16207,color:#fff
    style J fill:#991b1b,color:#fff
```

### Moderation Decisions

The system generates three possible moderation outcomes:

| Decision | Meaning |
|:---|:---|
| <img src="https://img.shields.io/badge/APPROVED-16A34A?style=flat-square" /> | The comment is considered acceptable under the moderation policy. |
| <img src="https://img.shields.io/badge/FLAGGED_FOR_REVIEW-D97706?style=flat-square" /> | The comment requires additional attention or review. |
| <img src="https://img.shields.io/badge/AUTO_BLOCKED-DC2626?style=flat-square" /> | The comment meets the configured criteria for automatic blocking. |

---

## 🧪 Model Development Pipeline

The project was developed through four major machine learning stages.

### Stage 1 — Exploratory Data Analysis (EDA)

**File:** `stage1_eda.py`

- Explores the dataset and its structure.
- Analyzes the distribution of toxicity labels.
- Identifies class imbalance.
- Studies the co-occurrence of different toxicity categories.

### Stage 2 — Text Preprocessing

**File:** `stage2_clean.py`

- Cleans raw text.
- Expands contractions.
- Normalizes repeated characters.
- Performs stratified training and validation splitting.

### Stage 3 — Machine Learning Baseline

**File:** `stage3_baseline.py`

- Uses TF-IDF for text feature extraction.
- Incorporates word and character n-grams.
- Trains Logistic Regression classifiers for the six toxicity labels.
- Establishes a baseline for comparison with the transformer model.

### Stage 4 — Transformer Model Training

**File:** `stage4_transformer.py`

- Uses `distilbert-base-uncased`.
- Fine-tunes the transformer for multi-label classification.
- Uses PyTorch and Hugging Face Transformers.
- Implements `BCEWithLogitsLoss` for multi-label learning.

The trained model is then integrated into the moderation service for comment analysis.

---

## 🛠️ Technology Stack

<table>
  <tr>
    <th>Category</th>
    <th>Technologies</th>
  </tr>
  <tr>
    <td><b>Programming Language</b></td>
    <td>Python</td>
  </tr>
  <tr>
    <td><b>Machine Learning</b></td>
    <
