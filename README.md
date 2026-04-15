# 🚗 Vehicle Maintenance Prediction & Agentic Fleet Management System

An AI-driven fleet analytics system that predicts vehicle maintenance needs using classical machine learning and generates structured, actionable recommendations through an agentic AI workflow. The system is designed to move beyond simple binary prediction — combining rule-based diagnostics, retrieval-augmented generation, and large language model reasoning to produce reports that are explainable, grounded, and professionally structured.

---

## 📌 Project Overview

Fleet maintenance is one of the most operationally critical challenges for logistics companies, transportation services, and mobility platforms. Unplanned vehicle breakdowns lead to service disruptions, safety risks, and high repair costs. This project addresses that problem by building an intelligent system that proactively predicts maintenance needs and guides fleet operators with actionable recommendations.

The project is developed across two milestones:

- **Milestone 1 (Mid-Sem):** A classical machine learning pipeline is built to predict whether a vehicle requires maintenance. The model is trained on historical vehicle data including mileage, component conditions, service history, and accident records. Special attention is given to handling class imbalance in the dataset using SMOTE, ensuring the model does not simply predict the majority class.

- **Milestone 2 (End-Sem):** The prediction system is extended into a full agentic AI application. Rather than returning just a 0 or 1, the system now reasons about *why* a vehicle needs maintenance, retrieves domain-specific knowledge to support that reasoning, and generates a structured, human-readable report through a large language model. The entire workflow is orchestrated using LangGraph.

---

## ✨ Key Features

- **ML-Based Maintenance Prediction** using a Decision Tree Classifier trained on real vehicle telemetry features
- **Class Imbalance Handling** via SMOTE to ensure balanced and reliable predictions
- **Rule-Based Risk Analysis** that identifies specific failure indicators such as weak battery, worn brakes, high mileage, and poor accident history
- **Retrieval-Augmented Generation (RAG)** using ChromaDB to fetch relevant maintenance guidelines from a structured knowledge base
- **Agentic Workflow** built with LangGraph, defining a clear multi-step pipeline with explicit state transitions
- **LLM-Powered Report Generation** via Groq API, producing structured and professional maintenance reports
- **Interactive UI** built with Gradio for easy vehicle data input and report viewing
- **Public Deployment** on Hugging Face Spaces for accessible, hosted usage

---

## 🏗️ System Architecture

The system follows a linear agentic pipeline where each stage builds on the output of the previous one:

```
Input Data → ML Prediction → Risk Analysis → RAG Retrieval → LLM Report Generation
```

**Stage 1 — Input:** The user provides vehicle details through the Gradio interface. These attributes include mileage, vehicle age, fuel type, transmission type, engine size, tire condition, brake condition, battery status, service history, and accident history.

**Stage 2 — ML Prediction:** The input is preprocessed and passed to the trained Decision Tree model, which returns a binary prediction — 0 (no maintenance needed) or 1 (maintenance required).

**Stage 3 — Risk Analysis:** Even when the ML model predicts maintenance is needed, it does not explain *what* is wrong. The Risk Analysis node fills this gap. It applies a set of deterministic rules to the input features and identifies specific risk factors — for example, flagging a weak battery, worn-out tires, high accident count, or excessive mileage. This step makes the system interpretable and trustworthy.

**Stage 4 — RAG Retrieval:** The identified risk factors are used as a query to search a ChromaDB vector database. This database contains a curated knowledge base of maintenance guidelines. The system retrieves the most relevant guidelines, grounding the final report in domain knowledge rather than relying purely on LLM generation.

**Stage 5 — Report Generation:** All gathered context — the ML prediction, risk factors, and retrieved guidelines — is sent to the Groq LLM. The model generates a structured maintenance report with clearly defined sections, ready to be presented to the fleet operator.

| Component | Description |
|---|---|
| **ML Model** | Decision Tree Classifier predicting maintenance requirement (0 or 1) |
| **Risk Analysis** | Rule-based engine detecting weak battery, worn tires, high mileage, accident history |
| **RAG** | ChromaDB vector store with maintenance guidelines, retrieved by semantic similarity |
| **LangGraph Agent** | Orchestrates workflow and manages shared state across all nodes |
| **Groq LLM** | Generates structured, professional fleet maintenance reports |

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| ML | Scikit-learn, SMOTE (imbalanced-learn) |
| Agent Framework | LangGraph |
| LLM API | Groq (`llama-3.1-8b-instant`) |
| Vector DB | ChromaDB + Sentence Transformers |
| UI | Gradio |
| Hosting | Hugging Face Spaces |
| Language | Python |

Each tool was chosen deliberately. LangGraph was selected over simple function chaining because it provides explicit state management and makes the pipeline easy to extend. ChromaDB was chosen for its lightweight setup and compatibility with Sentence Transformers embeddings. Groq was used for its free-tier access to fast LLM inference, making it practical for academic deployment.

---

## 📁 Project Structure

```
vehicle-maintenance-ai/
│
├── app.py                          # Main application entry point
├── requirements.txt                # All dependencies
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
│   └── architecture.png            # System architecture diagram
│
└── chroma_db/                      # Persisted ChromaDB vector store
```

The `src/` directory separates concerns cleanly — the agent workflow, RAG logic, and model utilities are each in their own module. This makes the codebase modular and easy to maintain or extend.

---

## 🚀 Getting Started

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

### 4. Run the Application

```bash
python app.py
```

---

## 🌐 Deployed Application

The application is publicly hosted on Hugging Face Spaces and requires no local setup to use.

**➡️ Live Demo:** *(hosted link coming soon)*

To use the deployed app:
1. Open the application link
2. Fill in your vehicle's details using the input form
3. Click **"Analyze Vehicle"**
4. The system will display the maintenance prediction, detected risk factors, retrieved guidelines, and the full structured report

---

## 📊 Model Details

The machine learning model at the core of Milestone 1 is a **Decision Tree Classifier** trained on a labelled vehicle dataset. The dataset contains features such as mileage, vehicle age, fuel type, transmission type, engine size, tire condition, brake condition, battery status, service history count, and accident history count. The target variable is `Need_Maintenance` (0 or 1).

A key challenge in this dataset is class imbalance — vehicles that do not need maintenance may outnumber those that do, which can bias the model toward always predicting the majority class. To address this, **SMOTE (Synthetic Minority Oversampling Technique)** is applied during training to synthetically balance the classes before fitting the model.

| Property | Value |
|---|---|
| Algorithm | Decision Tree Classifier |
| Output | Binary (0 = No Maintenance, 1 = Maintenance Required) |
| Imbalance Handling | SMOTE |
| Key Input Features | Mileage, Tire Condition, Brake Condition, Battery Status, Accident History, Vehicle Age |

---

## 🤖 Agent Workflow Documentation

The Milestone 2 system is built as a **LangGraph agent pipeline**, where each node in the graph performs a specific role and passes a shared state object to the next node. This design makes the system transparent, modular, and easy to debug or extend.

### State Structure

The agent maintains a shared state dictionary that is updated and passed through all nodes:

| State Key | Description |
|---|---|
| `input_data` | Raw vehicle attributes provided by the user |
| `prediction` | ML model output (0 = No Maintenance, 1 = Maintenance Required) |
| `risk_factors` | List of specific issues identified by the rule-based analysis |
| `guidelines` | Relevant maintenance guidelines retrieved from ChromaDB |
| `report` | Final structured report generated by the LLM |

### Node Summary

| Node | Role |
|---|---|
| **Input Node** | Preprocesses input and runs the Decision Tree model to get a maintenance prediction |
| **Risk Analysis Node** | Applies rule-based checks on battery, tires, brakes, mileage, vehicle age, and history |
| **RAG Retrieval Node** | Uses risk factors to query ChromaDB and retrieve the most relevant maintenance guidelines |
| **Report Node** | Sends all context to the Groq LLM and generates the final structured report |

### Design Highlights

- **Explainability** — The Risk Analysis node makes the system interpretable by surfacing specific reasons behind a prediction, rather than returning a black-box score
- **Grounded Output** — RAG ensures the LLM report is anchored in real maintenance knowledge, significantly reducing hallucination
- **Modularity** — Each node has a single, well-defined responsibility, making it easy to swap out components such as the ML model or the LLM provider
- **Scalability** — The pipeline is designed to be extended to fleet-level analysis, where multiple vehicles can be processed in sequence

---

## 📋 Structured Fleet Management Report

Every time the system analyzes a vehicle, it generates a structured report in the following format. This report is designed to be directly useful to a fleet manager or maintenance technician.

**1. Health Summary**
A high-level overview of the vehicle's condition. Includes the binary maintenance prediction (Yes / No) and an overall risk classification (Low / Moderate / High) based on the number and severity of detected risk factors.

**2. Risk Explanation**
A plain-language breakdown of the specific issues detected by the Risk Analysis node. Examples include weak battery charge, worn-out tire condition, degraded brake condition, high accident count, excessive mileage, and poor service history. Each factor is listed with context so the reader understands why it contributes to maintenance risk.

**3. Recommended Actions**
Concrete, component-level recommendations derived from the retrieved maintenance guidelines and LLM reasoning. This may include battery replacement, tire rotation or replacement, brake pad inspection, full vehicle servicing, or immediate repair of reported issues.

**4. Timeline**
A prioritized action timeline split into three horizons — Immediate actions for safety-critical issues, Short-term actions for general servicing needs, and Long-term actions for preventive maintenance to avoid future failures.

**5. Safety Disclaimer**
Every report includes a standard disclaimer stating that the output is advisory in nature and should not replace a professional mechanical inspection. This is a responsible AI design choice to prevent over-reliance on automated outputs.

---

## ⚖️ Responsible AI

Responsible AI practices are embedded throughout the design of this system rather than treated as an afterthought. The ML model is evaluated not just on accuracy but on its behavior across both classes, with SMOTE ensuring it does not ignore the minority class. The Risk Analysis layer adds a deterministic, rule-based check that provides transparency into predictions. RAG grounds the LLM output in a curated knowledge base, reducing the risk of the model fabricating maintenance advice. Every generated report includes a safety disclaimer to ensure users understand the advisory nature of the output. Together, these design choices reflect a commitment to building AI systems that are explainable, reliable, and safe.

---

## 🧹 Code Quality

- Code formatted using **Black** for consistency and readability across all modules
- Version controlled using Git and hosted on GitHub
- Modular project structure separating ML, RAG, and agent logic into distinct files

---

## 📦 Deliverables

- [x] Publicly deployed application on Hugging Face Spaces
- [x] Complete GitHub repository with full codebase
- [x] Demo video with functionality walkthrough
- [x] LangGraph agent workflow documentation
- [x] Structured fleet management reports

---

## 🔮 Future Improvements

The current system is a strong foundation but has clear directions for growth. Real-time IoT sensor integration would allow the system to process live vehicle telemetry rather than manual input. More advanced ML models such as XGBoost or neural networks could improve prediction accuracy. Adding SHAP-based feature importance would further enhance explainability. A fleet-level dashboard would allow operators to monitor the health of multiple vehicles simultaneously. Finally, an automated scheduling system could turn recommendations directly into booked service appointments.

---

## 📄 License

This project is developed for academic and educational purposes as part of a Generative AI course assignment.
