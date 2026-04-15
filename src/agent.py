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
def risk_node(state):
    d = state["input_data"]
    risks = []

    # ---------------- CORE COMPONENT CONDITIONS ---------------- #

    if d["Battery_Status"] == "weak":
        risks.append("weak battery - immediate inspection required")

    if d["Tire_Condition"] == "worn out":
        risks.append("worn tires - safety risk, replace immediately")

    if d["Brake_Condition"] == "worn out":
        risks.append("brake system degradation - urgent attention required")

    # ---------------- MILEAGE & USAGE ---------------- #

    if d["Mileage"] > 80000:
        risks.append("very high mileage - major components wear expected")
    elif d["Mileage"] > 50000:
        risks.append("high mileage - increased maintenance frequency required")

    if d["Odometer_Reading"] > 120000:
        risks.append("extreme odometer reading - full system inspection recommended")

    # ---------------- VEHICLE AGE ---------------- #

    if d["Vehicle_Age"] > 10:
        risks.append("aged vehicle - high probability of component failure")
    elif d["Vehicle_Age"] > 7:
        risks.append("old vehicle - preventive maintenance required")

    # ---------------- SERVICE HISTORY ---------------- #

    if d["Service_History"] <= 2:
        risks.append("low service history - potential hidden failures")

    if d["Maintenance_History"] == "poor":
        risks.append("poor maintenance history - high breakdown risk")
    elif d["Maintenance_History"] == "average":
        risks.append("inconsistent maintenance - moderate risk accumulation")

    # ---------------- REPORTED ISSUES ---------------- #

    if d["Reported_Issues"] >= 3:
        risks.append("multiple reported issues - system instability likely")
    elif d["Reported_Issues"] >= 1:
        risks.append("existing issues reported - further inspection needed")

    # ---------------- ACCIDENT HISTORY ---------------- #

    if d["Accident_History"] >= 3:
        risks.append("severe accident history - structural and brake risks")
    elif d["Accident_History"] >= 1:
        risks.append("accident history present - component reliability reduced")

    # ---------------- FUEL EFFICIENCY ---------------- #

    if d["Fuel_Efficiency"] < 12:
        risks.append("low fuel efficiency - possible engine degradation")
    elif d["Fuel_Efficiency"] < 15:
        risks.append("reduced fuel efficiency - maintenance recommended")

    # ---------------- ENGINE & SIZE ---------------- #

    if d["Engine_Size"] > 2500:
        risks.append("large engine - higher wear and maintenance demand")

    # ---------------- FUEL TYPE SPECIFIC ---------------- #

    if d["Fuel_Type"] == "diesel" and d["Mileage"] > 60000:
        risks.append("diesel engine wear - injector and filter inspection required")

    if d["Fuel_Type"] == "electric" and d["Battery_Status"] != "new":
        risks.append("electric vehicle battery degradation risk")

    if d["Fuel_Type"] == "petrol" and d["Vehicle_Age"] > 5:
        risks.append("petrol engine aging - spark and combustion check needed")

    # ---------------- TRANSMISSION ---------------- #

    if d["Transmission_Type"] == "manual" and d["Mileage"] > 50000:
        risks.append("manual transmission wear - clutch inspection required")

    # ---------------- OWNER TYPE ---------------- #

    if d["Owner_Type"] == "third":
        risks.append("multiple ownership history - uncertain maintenance quality")
    elif d["Owner_Type"] == "second":
        risks.append("second owner vehicle - moderate usage uncertainty")

    # ---------------- INSURANCE SIGNAL ---------------- #

    if d["Insurance_Premium"] > 20000:
        risks.append("high insurance premium - risk-prone vehicle profile")

    # ---------------- CROSS CONDITIONS (IMPORTANT) ---------------- #

    if d["Vehicle_Age"] > 7 and d["Mileage"] > 70000:
        risks.append("old + high mileage - compounded wear risk")

    if d["Maintenance_History"] == "poor" and d["Service_History"] <= 3:
        risks.append("poor maintenance + low servicing - critical neglect risk")

    if d["Battery_Status"] == "weak" and d["Fuel_Type"] == "electric":
        risks.append("critical battery condition in EV - urgent replacement needed")

    if d["Brake_Condition"] == "worn out" and d["Accident_History"] >= 1:
        risks.append("brake wear + accident history - severe safety hazard")

    if d["Tire_Condition"] == "worn out" and d["Mileage"] > 40000:
        risks.append("tire degradation with high usage - immediate replacement")

    # ---------------- FALLBACK (BIAS TOWARD MAINTENANCE) ---------------- #

    if len(risks) == 0:
        risks.append("routine maintenance recommended based on standard usage")

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
Use ONLY the provided guidelines to justify recommendations.
Do not hallucinate.

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