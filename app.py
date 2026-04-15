import streamlit as st
from src.agent import build_graph

# initializing graph
graph = build_graph()

st.set_page_config(page_title="Vehicle Maintenance AI", layout="centered")

st.title("Vehicle Maintenance AI")
st.write("AI-powered fleet maintenance analysis with ML + RAG + LLM")

#INPUT AREA

with st.form("vehicle_form"):

    Vehicle_Model = st.selectbox("Vehicle Model", ["car","truck","bus","van","motorcycle","suv"])
    Mileage = st.number_input("Mileage", min_value=0)
    Maintenance_History = st.selectbox("Maintenance History", ["good","average","poor"])
    Reported_Issues = st.number_input("Reported Issues", min_value=0)
    Vehicle_Age = st.number_input("Vehicle Age", min_value=0)
    Fuel_Type = st.selectbox("Fuel Type", ["petrol","diesel","electric"])
    Transmission_Type = st.selectbox("Transmission", ["manual","automatic"])
    Engine_Size = st.number_input("Engine Size", min_value=0)
    Odometer_Reading = st.number_input("Odometer Reading", min_value=0)
    Last_Service_Date = st.text_input("Last Service Date")
    Warranty_Expiry_Date = st.text_input("Warranty Expiry Date")
    Owner_Type = st.selectbox("Owner Type", ["first","second","third"])
    Insurance_Premium = st.number_input("Insurance Premium", min_value=0)
    Service_History = st.number_input("Service History", min_value=0)
    Accident_History = st.number_input("Accident History", min_value=0)
    Fuel_Efficiency = st.number_input("Fuel Efficiency", min_value=0.0)
    Tire_Condition = st.selectbox("Tire Condition", ["new","good","worn out"])
    Brake_Condition = st.selectbox("Brake Condition", ["new","good","worn out"])
    Battery_Status = st.selectbox("Battery Status", ["new","good","weak"])

    submitted = st.form_submit_button("Analyze Vehicle")

#PROCESS WILL HAPPEN HERE

if submitted:

    input_data = {
        "Vehicle_Model": Vehicle_Model,
        "Mileage": Mileage,
        "Maintenance_History": Maintenance_History,
        "Reported_Issues": Reported_Issues,
        "Vehicle_Age": Vehicle_Age,
        "Fuel_Type": Fuel_Type,
        "Transmission_Type": Transmission_Type,
        "Engine_Size": Engine_Size,
        "Odometer_Reading": Odometer_Reading,
        "Last_Service_Date": Last_Service_Date,
        "Warranty_Expiry_Date": Warranty_Expiry_Date,
        "Owner_Type": Owner_Type,
        "Insurance_Premium": Insurance_Premium,
        "Service_History": Service_History,
        "Accident_History": Accident_History,
        "Fuel_Efficiency": Fuel_Efficiency,
        "Tire_Condition": Tire_Condition,
        "Brake_Condition": Brake_Condition,
        "Battery_Status": Battery_Status
    }

    with st.spinner("Analyzing vehicle..."):

        result = graph.invoke({
            "input_data": input_data
        })

 #OUTPUT AREA

    st.subheader("Maintenance Report")
    st.write(result["report"])

  #DEBUGGING for better results
    with st.expander("Debug Info"):
        st.write("Prediction:", result["prediction"])
        st.write("Risk Factors:", result["risk_factors"])
        st.write("Guidelines:", result["guidelines"])