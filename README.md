# FlipLens: Returns Triage & Fraud Detection System
 
## 📌 Project Overview
This project is an intelligent **Returns Triage and Fraud Detection System** designed to streamline e-commerce returns, minimize revenue leakage from return fraud, and optimize customer experience. The system leverages a dual-layered approach: a highly optimized Machine Learning (ML) model for fast risk scoring and an advanced Large Language Model (LLM) agent framework (LangGraph + Google Gemini) for nuanced policy evaluation and final verdict justification.

## 🏗️ Solution Architecture
Our architecture separates fast quantitative risk assessment from qualitative policy reasoning:

1. **Robust Data Backend (PostgreSQL)**: Handles highly normalized schemas covering users, sellers, orders, products, and returns. Includes automated temporal feature engineering.
2. **Machine Learning Pipeline (LightGBM)**: Calculates an instantaneous "Fraud Probability Score" by analyzing behavioral metadata (return-to-order ratio, velocity, past anomalies, seller integrity).
3. **GenAI Triage Agent (LangGraph + Gemini)**: Acts as the intelligent orchestrator. It consumes the ML risk score, fetches specific category return policies using **RAG (Retrieval-Augmented Generation)**, and reads customer narratives/evidence. 
4. **Decision Engine**: Automatically classifies returns into three buckets:
   - ✅ **Auto-Approve**: Low risk, trusted user, within policy.
   - ❌ **Auto-Reject**: High risk, clear policy violation, or behavioral fraud signals.
   - ⚠️ **Escalate**: Borderline cases (e.g., highly trusted user but missing required image evidence for a high-value claim) that require human manual review.
5. **Interactive Dashboard (Streamlit)**: Provides an investigator UI to test individual queries, review batch escalation logs, and interact with the adversarial probe set.

## 🚀 Key Features
* **Dynamic Policy Retrieval (RAG)**: Automatically fetches the exact return policy (e.g., Electronics, Fashion, Jewellery) using FAISS and Langchain.
* **Adversarial Robustness**: Evaluated against a rigorous dataset of borderline cases, swap fraud attempts, and conflicting signals.
* **Explainable AI (XAI)**: Generates detailed, human-readable reasoning chains for every decision, citing specific policies and user features via SHAP values and LLM logic.
* **Idempotent Batch Processing**: Ability to process thousands of backlog returns seamlessly, picking up exactly where it left off in case of interruptions.

## 💻 Tech Stack
* **Language**: Python 3.10+
* **Database**: PostgreSQL 17
* **Machine Learning**: Scikit-Learn, LightGBM, SHAP, Pandas, NumPy
* **GenAI / LLM Framework**: Langchain, LangGraph, Google Gemini 3.5 Flash, FAISS
* **Frontend UI**: Streamlit

## 🛠️ Setup & Installation

### 1. Prerequisites
* PostgreSQL installed and running on default port (5432).
* Python 3.10 or higher.
* Google Gemini API Key.

### 2. Environment Setup
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements_agents.txt
```

### 3. Configuration
Open `settings.py` and ensure the following are configured correctly:
* `DB_URI = "postgresql://postgres:YOUR_PASSWORD@localhost:5432/grid_db"`
* `GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"`

### 4. Database Initialization & Data Generation
Initialize the database schema, insert synthetic data, and create the adversarial test queries:
```bash
python db/db_config.py
```
*(Note: If you have existing dumps, you can run the provided powershell scripts).*

### 5. Train the ML Model
Train the LightGBM fraud detection model to generate the necessary artifacts (`lgbm_model.pkl` and dataset splits):
```bash
python ml/model_pipeline.py
```

### 6. Build the RAG Policy Index
Vectorize the return policies so the LLM can query them:
```bash
python rag/build_index.py
```

### 7. Launch the Dashboard
Start the Streamlit application to interact with the system:
```bash
streamlit run app.py
```
Access the dashboard at `http://localhost:8501`.

## 🧪 Testing the System
You can test the system's reasoning directly in the dashboard using the "Single Query" tab. We recommend testing the following adversarial IDs to see the LLM's nuanced decision-making in action:
* `RET_09905003` (Bad user vs. Bad Seller - Swap Claim)
* `RET_09905004` (Bad user vs. Good Seller - Swap Claim)
* `RET_09905005` (Trusted user, high value item, missing required image evidence)
