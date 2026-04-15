# Vehicle Maintenance Prediction & Agentic Fleet Management System

An AI-driven fleet analytics system that predicts vehicle maintenance needs using classical machine learning and generates structured, actionable recommendations through an agentic AI workflow.

---

## Project Overview

This project is developed in two phases:

- **Milestone 1 (Mid-Sem):** A machine learning model predicts whether a vehicle requires maintenance using historical sensor and usage data.
- **Milestone 2 (End-Sem):** The system is extended into an agent-based AI application that reasons about vehicle health, retrieves domain knowledge, and generates professional maintenance reports.

---

## Key Features

- Machine learning-based maintenance prediction (Decision Tree)
- Imbalanced data handling using SMOTE
- Rule-based risk analysis engine
- Retrieval-Augmented Generation (RAG) using ChromaDB
- Agentic workflow using LangGraph
- LLM-based report generation via Groq API
- Interactive UI built with Gradio
- End-to-end pipeline from raw input to actionable output

---

## System Architecture

```
Input Data → ML Prediction → Risk Analysis → RAG Retrieval → LLM Report Generation
```

| Component | Description |
|---|---|
| **ML Model** | Decision Tree Classifier predicting maintenance requirement (0 or 1) |
| **Risk Analysis** | Detects issues like weak battery, worn tires, high mileage, accident history |
| **RAG** | Retrieves relevant maintenance guidelines from ChromaDB vector store |
| **LangGraph Agent** | Controls workflow and manages state across nodes |
| **Groq LLM** | Generates structured, professional maintenance reports |

---

## Tech Stack

| Category | Tools |
|---|---|
| ML | Scikit-learn, SMOTE (imbalanced-learn) |
| Agent Framework | LangGraph |
| LLM API | Groq (`llama3-8b-8192`) |
| Vector DB | ChromaDB + Sentence Transformers |
| UI | Gradio |
| Hosting | Hugging Face Spaces |
| Language | Python |

---

## Project Structure

```
vehicle-maintenance-ai/
│
├── app.py                          # Main application entry point
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── vehicle_dataset.csv         # Training dataset
│
├── notebooks/
│   └── milestone1_training.ipynb   # ML model training (Milestone 1)
│
├── src/
│   ├── agent.py                    # LangGraph agent workflow
│   ├── rag.py                      # ChromaDB RAG setup and retrieval
│   └── model_utils.py              # ML model loading and prediction
│
├── assets/
│   └── architecture.png
│
└── chroma_db/                      # Persisted vector store
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/vehicle-maintenance-ai.git
cd vehicle-maintenance-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Your Groq API Key

```bash
export GROQ_API_KEY=your_api_key_here
```

---

## Deployed Application

The application is publicly hosted on Hugging Face Spaces.

(hosted link ka wait chal raha hai)

To use it:
1. Open the deployed app link
2. Fill in the vehicle details in the form
3. Click **"Analyze Vehicle"**
4. View the maintenance prediction, risk factors, retrieved guidelines, and final report

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | Decision Tree Classifier |
| Output | Binary (0 = No Maintenance, 1 = Maintenance Required) |
| Imbalance Handling | SMOTE |
| Key Input Features | Mileage, Tire Condition, Brake Condition, Battery Status, Accident History, Vehicle Age |

---

---

## Agent Workflow Documentation

The system is an agent-based pipeline integrating ML, rule-based reasoning, RAG, and LLM report generation to simulate intelligent fleet management.

### State Structure

| State Key | Description |
|---|---|
| `input_data` | Vehicle attributes |
| `prediction` | ML output (0 = No Maintenance, 1 = Maintenance Required) |
| `risk_factors` | Identified issues from rule-based analysis |
| `guidelines` | Retrieved maintenance recommendations |
| `report` | Final structured output |

### Node Summary

| Node | Role |
|---|---|
| **Input Node** | Runs Decision Tree model, outputs maintenance prediction |
| **Risk Analysis Node** | Rule-based checks on battery, tires, brakes, mileage, age, history |
| **RAG Retrieval Node** | Queries ChromaDB using risk factors, retrieves guidelines |
| **Report Node** | Combines all inputs, generates structured LLM report |

### Design Highlights

- **Explainability** — Risk factors provide transparency into predictions
- **Grounded Output** — RAG reduces hallucination
- **Modularity** — Each node has a single, clear responsibility
- **Scalability** — Extendable to fleet-level systems

---

## Structured Fleet Management Report

Every analysis generates a report in the following format:

**1. Health Summary** — Maintenance prediction (Yes/No) and overall risk level (Low / Moderate / High)

**2. Risk Explanation** — Key detected issues such as weak battery, worn tires or brakes, high mileage, poor service or accident history

**3. Recommended Actions** — Component-level fixes (battery, brakes, tires) and servicing tasks

**4. Timeline** — Immediate (safety-critical), Short-term (general servicing), Long-term (preventive)

**5. Safety Disclaimer** — This report is advisory and should not replace professional inspection.

---


## Responsible AI

- All outputs are advisory and not definitive
- Safety disclaimers are included in every report
- RAG is used to ground the LLM and reduce hallucinations
- Deterministic ML logic is combined with LLM reasoning

---

## Code Quality

- Code formatted using **Black** for consistency and readability.
- Version controlled using Git and hosted on GitHub.

---

## Deliverables

- [x] Publicly deployed application (Hugging Face Spaces)
- [x] Complete GitHub repository with codebase
- [x] Demo video (functionality walkthrough)
- [x] LangGraph agent workflow documentation
- [x] Structured fleet maintenance reports

---

## Future Improvements

- Real-time IoT sensor integration
- Advanced models (XGBoost, Neural Networks)
- Feature importance and explainability (SHAP)
- Fleet-level dashboard with multiple vehicles
- Automated service scheduling system

---

## 📄 License

This project is developed for academic and educational purposes as part of a Generative AI course assignment.
