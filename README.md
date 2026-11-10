# 🛡️ AI Content Moderation System

<p align="center">
  <b>An AI-powered content moderation platform for detecting and classifying toxic online comments using NLP and Deep Learning.</b>
</p>

<p align="center">
  <a href="https://content-moderation-system-nf9s.onrender.com/">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-Try_Now-6366F1?style=for-the-badge" alt="Live Demo"/>
  </a>
  <a href="https://github.com/hellosarthak25/content-moderation-system">
    <img src="https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
</p>

---

## 🌐 Live Demo

**Try the application here:**

👉 https://content-moderation-system-nf9s.onrender.com/

Explore the interactive dashboard, submit comments, view toxicity predictions, and inspect moderation decisions in real time.

---

## 📌 About the Project

The **AI Content Moderation System** is an end-to-end machine learning application designed to automatically identify and classify harmful or inappropriate content in online comments.

It uses a fine-tuned **DistilBERT transformer model** to perform multi-label toxicity classification, allowing a single comment to be evaluated across multiple toxicity categories simultaneously.

The system combines NLP, deep learning, a FastAPI backend, and an interactive web dashboard to provide automated moderation decisions and maintain an audit history of processed comments.

### 🎯 Problem Statement

Online platforms receive thousands of comments every day. Manually identifying toxic, abusive, or harmful content is time-consuming and difficult to scale.

This project aims to simplify that process through automated toxicity detection and a structured moderation system.

## ✨ Key Features

- **Multi-Label Toxicity Detection:** Identifies multiple toxicity categories within a single comment.
- **DistilBERT-Based Classification:** Uses a fine-tuned transformer model for contextual text understanding.
- **Real-Time Comment Analysis:** Processes user-submitted comments through an interactive dashboard.
- **Automated Moderation Decisions:** Classifies comments into three moderation outcomes:
  - `APPROVED` – Clean comments.
  - `FLAGGED_FOR_REVIEW` – Comments requiring further review.
  - `AUTO_BLOCKED` – Comments identified as high-severity violations.
- **Toxicity Probability Scores:** Displays prediction scores for different toxicity categories.
- **Interactive Dashboard:** Provides a user-friendly interface with moderation statistics and visual indicators.
- **Database Audit Logging:** Stores moderation events and decisions using SQLite.
- **REST API:** Supports individual and batch comment moderation.

## 🧠 Toxicity Categories

The model detects six different categories:

| Category | Description |
|---|---|
| Toxic | Generally harmful or abusive language |
| Severe Toxic | Highly harmful or aggressive language |
| Obscene | Vulgar or inappropriate language |
| Threat | Threatening language |
| Insult | Insulting or offensive language |
| Identity Hate | Hate directed towards identity groups |

## 🏗️ System Architecture & Workflow

The project follows a structured machine learning pipeline:

**1. Exploratory Data Analysis (EDA)**

- Analyzes the dataset and label distribution.
- Studies class imbalance and relationships between toxicity categories.

**2. Text Preprocessing**

- Cleans and normalizes text.
- Expands contractions.
- Handles repeated characters.
- Performs stratified data splitting.

**3. Baseline Model Development**

- Uses TF-IDF for text feature extraction.
- Implements Logistic Regression classifiers.
- Establishes a baseline for model comparison.

**4. Transformer Model Training**

- Fine-tunes `distilbert-base-uncased`.
- Uses `BCEWithLogitsLoss` for multi-label classification.
- Generates toxicity scores across all six categories.

**5. Backend & Moderation Engine**

- Uses FastAPI to handle moderation requests.
- Applies moderation policies to model predictions.
- Generates automated moderation decisions.

**6. Database & Dashboard**

- Stores moderation records using SQLite and SQLAlchemy.
- Displays predictions, moderation statistics, and audit logs through the web dashboard.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data manipulation and analysis |
| Scikit-learn | Baseline machine learning model |
| NLP | Text preprocessing and classification |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | DistilBERT implementation |
| DistilBERT | Multi-label toxicity classification |
| FastAPI | Backend and REST API |
| SQLAlchemy | Database operations |
| SQLite | Moderation audit logging |
| HTML, CSS & JavaScript | Interactive dashboard |
| Render | Application deployment |

## 📂 Project Structure

```text
content-moderation-system/
│
├── distilbert_moderation/
│   └── Fine-tuned DistilBERT model and tokenizer
│
├── app.py
│   └── FastAPI server and interactive dashboard
│
├── moderation_service.py
│   └── Model inference and moderation policy engine
│
├── database.py
│   └── SQLite database and audit logging
│
├── text_cleaner.py
│   └── Text preprocessing utilities
│
├── stage1_eda.py
│   └── Exploratory data analysis
│
├── stage2_clean.py
│   └── Text cleaning and data preparation
│
├── stage3_baseline.py
│   └── TF-IDF and Logistic Regression baseline
│
├── stage4_transformer.py
│   └── DistilBERT fine-tuning
│
├── requirements.txt
│   └── Project dependencies
│
├── Dockerfile
│   └── Container configuration
│
└── README.md
```

## ⚙️ Installation & Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/hellosarthak25/content-moderation-system.git
```

### 2. Navigate to the Project Directory

```bash
cd content-moderation-system
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python app.py
```

### 6. Open in Browser

Visit:

```text
http://localhost:8000
```

The content moderation dashboard will be available locally.

## 🔌 API Endpoints

The application provides the following REST API endpoints:

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/moderate` | Moderate a single comment |
| POST | `/api/moderate/batch` | Moderate multiple comments |
| GET | `/api/logs` | Retrieve moderation audit logs |
| GET | `/api/stats` | Retrieve moderation statistics |
| POST | `/api/logs/clear` | Clear moderation audit logs |

### Example API Request

**Endpoint:** `/api/moderate`

```json
{
  "text": "Have a great day!"
}
```

The API returns the moderation result, including toxicity predictions and the corresponding moderation decision.

## 🚀 Deployment

The application is deployed using **Render**.

🔗 **Live Application:** https://content-moderation-system-nf9s.onrender.com/

## 🔮 Future Improvements

- Improve model performance through further experimentation and optimization.
- Add support for additional languages.
- Introduce batch file uploads for large-scale moderation.
- Improve moderation analytics and reporting.
- Explore integration with social media platforms through APIs.

## 🤝 Contributing

Contributions, suggestions, and feedback are welcome!

If you have ideas for improving the project, feel free to fork the repository and submit a pull request.

## 📄 License

This project is intended for educational and research purposes.

---

<p align="center">
  <b>Built with Python, NLP, Machine Learning & Deep Learning ❤️</b>
</p>

<p align="center">
  ⭐ If you find this project interesting, consider giving the repository a star!
</p>
