# 🧠 AmritaGPT v2 — Research Intelligence Platform

AmritaGPT v2 is a cutting-edge **Local Academic Intelligence Platform** designed for universities and research labs to map student expertise, discover collaborations, and generate strategic insights—all running locally on your hardware.

---

## 🚀 Key Features

- **📊 Advanced Analytics**: Real-time visualization of student skill distribution and department demographics.
- **🧠 Local AI Insights**: Strategic trend analysis generated locally using **Ollama (Llama 3.2)**.
- **🕸 Expertise Knowledge Graph**: A network-based model connecting Students, Skills, and Research Areas to identify expertise "Hubs".
- **🤝 AI Research Team Builder**: Autonomous formation of complementary student research teams using **K-Means Clustering**.
- **💬 Student Intelligence Chat (RAG)**: A local Retrieval-Augmented Generation system to query student profiles using natural language.
- **📑 Strategic Reports**: Automated, professional academic report generation for university administration.

---

## 🛠 Technology Stack

- **Frontend**: Streamlit
- **Local LLM**: Ollama (Llama 3.2)
- **Vector DB**: FAISS
- **Graph Engine**: NetworkX & Plotly
- **ML Framework**: Scikit-learn (K-Means), LangChain
- **Data Handling**: Pandas

---

## 🏁 Getting Started

### 1. Prerequisites
- [Ollama](https://ollama.com/) installed and running.
- Python 3.9+ installed.

### 2. Setup Models
Pull the required local models via Ollama:
```bash
ollama pull llama3.2
```

### 3. Installation
Install the project dependencies using `uv` or `pip`:
```bash
# Using uv (Recommended)
uv sync

# OR using pip
pip install -r requirements.txt
```

### 4. Run the Platform
Launch the Streamlit dashboard:
```bash
uv run streamlit run app.py
```

---

## 📂 Project Structure

- `analytics/`: Logic for LLM-powered insight generation.
- `graphs/`: Knowledge graph construction and analytics.
- `ml_models/`: Student clustering and team formation algorithms.
- `reports/`: Documentation and formal report generators.
- `data/`: Student datasets (includes synthetic `student_data.csv`).
- `utils/`: RAG engine and shared utilities.
- `app.py`: Main Streamlit application entry point.

---

## 🛡 Privacy & Security
Because AmritaGPT v2 uses **Ollama**, all data stays on your local machine. No student data is sent to external APIs or third-party servers, making it compliant with strict academic privacy standards.

---
Built for the **Amrita University Research Lab** initiative.
