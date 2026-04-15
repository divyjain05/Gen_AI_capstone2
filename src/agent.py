from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from typing import TypedDict, List, Dict

from src.model_utils import predict_maintenance
from src.rag import retrieve_guidelines

# LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

# STATE
class GraphState(TypedDict):
    input_data: Dict
    prediction: int
    risk_factors: List[str]
    guidelines: List[str]
    report: str

# NODE 1: ML Prediction
def input_node(state: GraphState):
    prediction = predict_maintenance(state["input_data"])
    return {
        "prediction": prediction
    }

# NODE 2: Risk Analysis
def risk_node(state: GraphState):
    d = state["input_data"]
    risks = []

    if d["Battery_Status"] == "weak":
        risks.append("weak battery")

    if d["Tire_Condition"] == "worn out":
        risks.append("worn tires")

    if d["Brake_Condition"] == "worn out":
        risks.append("brake issues")

    if d["Accident_History"] >= 3:
        risks.append("high accident history")

    if d["Mileage"] > 50000:
        risks.append("high mileage")

    if d["Vehicle_Age"] > 8:
        risks.append("old vehicle")

    return {
        "risk_factors": risks
    }

# NODE 3: RAG
def rag_node(state: GraphState):
    risks = state.get("risk_factors", [])

    query = ", ".join(risks) if risks else "general vehicle maintenance"

    guidelines = retrieve_guidelines(query)

    return {
        "guidelines": guidelines
    }

# NODE 4: REPORT
def report_node(state: GraphState):
    vehicle_data = state["input_data"]
    prediction = state.get("prediction", 0)
    risks = state.get("risk_factors", [])
    guidelines = state.get("guidelines", [])

    prompt = f"""
You are a fleet maintenance expert.

Vehicle data: {vehicle_data}
ML Risk Prediction: {"Needs Maintenance" if prediction == 1 else "No Maintenance Required"}
Risk Factors: {risks}
Relevant Guidelines: {guidelines}

Generate a structured report with:
1. Health Summary
2. Recommended Actions
3. Timeline
4. Safety Disclaimer

Keep it concise and professional.
"""

    response = llm.invoke(prompt)

    return {
        "report": response.content
    }

# BUILD GRAPH
def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("input", input_node)
    builder.add_node("risk", risk_node)
    builder.add_node("rag", rag_node)
    builder.add_node("report", report_node)

    builder.set_entry_point("input")

    builder.add_edge("input", "risk")
    builder.add_edge("risk", "rag")
    builder.add_edge("rag", "report")

    return builder.compile()