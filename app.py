import gradio as gr
from src.agent import build_graph

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
        gr.Number(label="Mileage"),
        gr.Dropdown(["good","average","poor"], label="Maintenance History"),
        gr.Number(label="Reported Issues"),
        gr.Number(label="Vehicle Age"),
        gr.Dropdown(["petrol","diesel","electric"], label="Fuel Type"),
        gr.Dropdown(["manual","automatic"], label="Transmission"),
        gr.Number(label="Engine Size"),
        gr.Number(label="Odometer Reading"),
        gr.Textbox(label="Last Service Date"),
        gr.Textbox(label="Warranty Expiry Date"),
        gr.Dropdown(["first","second","third"], label="Owner Type"),
        gr.Number(label="Insurance Premium"),
        gr.Number(label="Service History"),
        gr.Number(label="Accident History"),
        gr.Number(label="Fuel Efficiency"),
        gr.Dropdown(["new","good","worn out"], label="Tire Condition"),
        gr.Dropdown(["new","good","worn out"], label="Brake Condition"),
        gr.Dropdown(["new","good","weak"], label="Battery Status"),
    ],
    
    outputs=gr.Textbox(label="Maintenance Report"),
    
    title="Vehicle Maintenance AI",
    description="ML + Agentic AI Fleet Management System"
)

interface.launch()