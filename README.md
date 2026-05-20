# 🎫 AI-Powered Support Ticket Intelligence System

<div align="center">

### 🤖 Enterprise NLP & Machine Learning Solution for Smart Customer Support Automation

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge\&logo=python)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange?style=for-the-badge)
![NLP](https://img.shields.io/badge/NLP-Text%20Analytics-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-red?style=for-the-badge\&logo=streamlit)
![Scikit Learn](https://img.shields.io/badge/Scikit--Learn-ML-yellow?style=for-the-badge\&logo=scikitlearn)

</div>

---

# 🚀 Project Overview

Modern businesses receive thousands of customer support tickets every day.

These include:

* 💳 Billing Issues
* 🐛 Technical Problems
* 👤 Account Issues
* 🔒 Security Concerns
* 📞 General Queries

Manual ticket handling creates major operational challenges:

❌ Tickets are not categorized properly
❌ Urgent issues get delayed
❌ Support teams waste time sorting tickets
❌ Customer satisfaction decreases

---

# 🎯 Solution

This project uses:

✅ Natural Language Processing (NLP)
✅ Machine Learning
✅ AI-driven operational intelligence

to automatically:

* classify tickets
* prioritize issues
* analyze sentiment
* detect operational risk
* route tickets intelligently
* generate AI-powered support reports

---

# ✨ Features

# 🎯 Intelligent Ticket Classification

Automatically classifies tickets into:

* 💰 Billing
* 💻 Technical Support
* 👤 Account Management
* 🔒 Security Issues
* 📞 General Support

---

# ⚡ Priority Prediction

Predicts:

* 🔴 Critical
* 🟠 High
* 🟡 Medium
* 🟢 Low

priority tickets automatically.

---

# 😊 Sentiment & Emotion Analysis

Analyzes:

* Very Negative
* Negative
* Neutral
* Positive
* Very Positive

and emotions like:

* Angry
* Frustrated
* Happy
* Dissatisfied

---

# 🛡️ AI Risk Assessment

Detects:

* Fraud Risk
* Security Breach Risk
* Financial Impact
* Customer Churn Risk

with intelligent risk scoring.

---

# 📨 Smart Ticket Routing

Automatically routes tickets to:

| Category     | Assigned Team      |
| ------------ | ------------------ |
| 💰 Billing   | Finance Team       |
| 💻 Technical | Technical Support  |
| 👤 Account   | Account Management |
| 🔒 Security  | Security Team      |

---

# 📊 AI Decision Support System

Generates:

* AI operational reports
* urgency score
* escalation level
* SLA prediction
* business impact
* AI summaries
* suggested resolutions

---

# 📈 Interactive Analytics Dashboard

Includes:

* 📊 Ticket distribution charts
* 📉 Confusion matrix
* 📌 Support workload analytics
* ☁️ WordCloud visualization
* 📋 Model performance metrics
* 💰 ROI calculator

---

# 🛠️ Tech Stack

# 👨‍💻 Programming

* Python

# 🤖 Machine Learning

* Scikit-learn
* Logistic Regression
* TF-IDF Vectorization

# 🧠 NLP

* TextBlob
* NLTK

# 📊 Visualization

* Matplotlib
* Seaborn
* WordCloud

# 🌐 Frontend

* Streamlit

---

# 📁 Project Structure

```bash
support-ticket-classification/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_text_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_ticket_classification.ipynb
│   ├── 05_priority_prediction.ipynb
│   └── 06_model_evaluation.ipynb
│
├── models/
│   ├── ticket_classifier.pkl
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
│
├── outputs/
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 🔄 Complete Workflow

```text
Customer Support Ticket
            ↓
Text Cleaning & Preprocessing
            ↓
Tokenization & NLP Processing
            ↓
TF-IDF Feature Extraction
            ↓
Machine Learning Classification
            ↓
Priority Prediction
            ↓
Sentiment & Risk Analysis
            ↓
AI Operational Intelligence
            ↓
Smart Routing & SLA Prediction
            ↓
Comprehensive AI Support Report
```

---

# 📊 Machine Learning Pipeline

## 🔹 Step 1 — Data Collection

Load support ticket dataset.

---

## 🔹 Step 2 — Text Preprocessing

Perform:

* lowercasing
* punctuation removal
* stopword removal
* tokenization

---

## 🔹 Step 3 — Feature Engineering

Convert text into numerical vectors using:

* TF-IDF Vectorization

---

## 🔹 Step 4 — Model Training

Train:

* Logistic Regression classifier

---

## 🔹 Step 5 — Model Evaluation

Evaluate using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 🔹 Step 6 — AI Operational Analysis

Perform:

* sentiment analysis
* risk assessment
* urgency detection
* smart routing

---

# 📈 Model Evaluation Metrics

The project includes:

✅ Accuracy Score
✅ Precision Score
✅ Recall Score
✅ F1 Score
✅ Confusion Matrix
✅ Classification Report

---

# ▶️ How To Run The Project

# 1️⃣ Clone Repository

```bash
git clone <your-github-repository-link>
```

---

# 2️⃣ Open Project Folder

```bash
cd support-ticket-classification
```

---

# 3️⃣ Create Virtual Environment

## Windows

```bash
python -m venv venv
```

Activate environment:

```bash
venv\Scripts\activate
```

---

# 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 5️⃣ Install Streamlit

```bash
pip install streamlit
```

---

# 6️⃣ Run Jupyter Notebooks

Run notebooks in order:

| Notebook                       | Purpose                |
| ------------------------------ | ---------------------- |
| 01_data_understanding.ipynb    | Dataset exploration    |
| 02_text_preprocessing.ipynb    | NLP preprocessing      |
| 03_feature_engineering.ipynb   | TF-IDF vectorization   |
| 04_ticket_classification.ipynb | ML training            |
| 05_priority_prediction.ipynb   | Priority logic         |
| 06_model_evaluation.ipynb      | Performance evaluation |

This generates:

* trained models
* vectorizers
* processed data

---

# 7️⃣ Run Streamlit Application

```bash
streamlit run app.py
```

---

# 8️⃣ Open Browser

Streamlit automatically opens:

```text
http://localhost:8501
```

---

# 💡 Example Ticket

```text
My payment failed twice and money was deducted from my account.
Please resolve this urgently.
```

---

# 📋 Example AI Report

| Feature              | Output                      |
| -------------------- | --------------------------- |
| Ticket Category      | Billing Issue               |
| Priority             | High                        |
| Sentiment            | Very Negative               |
| Risk Level           | High Risk                   |
| Assigned Team        | Finance Team                |
| SLA Target           | 2 Hours                     |
| Urgency Score        | 92%                         |
| Suggested Resolution | Verify payment gateway logs |

---

# 💰 Business Impact

This AI system helps businesses:

✅ Reduce manual ticket sorting
✅ Improve customer response time
✅ Detect urgent issues faster
✅ Reduce SLA violations
✅ Improve operational efficiency
✅ Reduce support costs
✅ Improve customer satisfaction

---

# 📌 ROI Benefits

Organizations can:

* reduce support handling time
* optimize support workforce
* reduce operational costs
* improve customer retention

---

# 🚀 Future Enhancements

* Deep Learning Models
* Transformer-based NLP
* Real-time Database Integration
* Multi-language Support
* Voice-based Ticket Analysis
* Cloud Deployment
* Generative AI Auto-Reply System

---

# 👨‍💻 Author

## Subhankar Nandi

🎓 BTech CSE AIML Student
🤖 Machine Learning & AI Enthusiast
🚀 Passionate about NLP, AI Automation & Intelligent Systems

---
