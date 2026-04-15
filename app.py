import gradio as gr
import os
from src.agent import build_graph

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
graph = build_graph()

def analyze_vehicle(
    Vehicle_Model,
    Mileage,
    Maintenance_History,
    Reported_Issues,
    Vehicle_Age,
    Fuel_Type,
    Transmission_Type,
    Engine_Size,
    Odometer_Reading,
    Last_Service_Date,
    Warranty_Expiry_Date,
    Owner_Type,
    Insurance_Premium,
    Service_History,
    Accident_History,
    Fuel_Efficiency,
    Tire_Condition,
    Brake_Condition,
    Battery_Status
):
    
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

    result = graph.invoke({
        "input_data": input_data
    })

    return result["report"]


interface = gr.Interface(
    fn=analyze_vehicle,
    
    inputs=[
        gr.Dropdown(["car","truck","bus","van","motorcycle","suv"], label="Vehicle Model"),
        gr.Number(label="Mileage (km) (e.g., 50000, 79093)"),
        gr.Dropdown(["good","average","poor"], label="Maintenance History"),
        gr.Number(label="Reported Issues (eg., 0, 1, 3)"),
        gr.Number(label="Vehicle Age (years) (eg., 3, 5, 10)"),
        gr.Dropdown(["petrol","diesel","electric"], label="Fuel Type"),
        gr.Dropdown(["manual","automatic"], label="Transmission"),
        gr.Number(label="Engine Size (cc) (eg., 1500, 2000, 3000)"),
        gr.Number(label="Odometer Reading (km) (eg., 50000, 120000)"),
        gr.Textbox(label="Last Service Date (DD-MM-YYYY)"),
        gr.Textbox(label="Warranty Expiry Date (DD-MM-YYYY)"),
        gr.Dropdown(["first","second","third"], label="Owner Type"),
        gr.Number(label="Insurance Premium (eg., 20782)"),
        gr.Number(label="Service History (eg., 0, 8, 15)"),
        gr.Number(label="Accident History (eg., 0, 1, 4)"),
        gr.Number(label="Fuel Efficiency (km/l) (eg., 13.62, 10.5, 8.0)"),
        gr.Dropdown(["new","good","worn out"], label="Tire Condition"),
        gr.Dropdown(["new","good","worn out"], label="Brake Condition"),
        gr.Dropdown(["new","good","weak"], label="Battery Status"),
    ],
    
    outputs=gr.Textbox(label="Maintenance Report"),
    
    title="Vehicle Maintenance AI",
    description="ML + Agentic AI Fleet Management System"
)

interface.launch()