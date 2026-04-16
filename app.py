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

    result = graph.invoke({"input_data": input_data})
    return result["report"]


with gr.Blocks(theme=gr.themes.Soft()) as demo:

    gr.Markdown("# Vehicle Maintenance AI")
    gr.Markdown("Predict maintenance needs and generate intelligent recommendations using ML + Agentic AI")

    # ---------------- VEHICLE DETAILS ---------------- #
    gr.Markdown("## Vehicle Details")

    with gr.Row():
        with gr.Column():
            Vehicle_Model = gr.Dropdown(["car","truck","bus","van","motorcycle","suv"], label="Vehicle Model")
            Mileage = gr.Number(label="Mileage (km)(eg: 12000, 30000)")
            Vehicle_Age = gr.Number(label="Vehicle Age (years)(eg: 1, 5, 10)")
            Fuel_Type = gr.Dropdown(["petrol","diesel","electric"], label="Fuel Type")
            Transmission_Type = gr.Dropdown(["manual","automatic"], label="Transmission")

        with gr.Column():
            Engine_Size = gr.Number(label="Engine Size (cc)(eg: 1500, 2000, 3000)")
            Odometer_Reading = gr.Number(label="Odometer Reading (km) (eg: 5000, 60000, 120000)")
            Owner_Type = gr.Dropdown(["first","second","third"], label="Owner Type")
            Insurance_Premium = gr.Number(label="Insurance Premium (in Rs)(eg: 25000, 45000)")
            Fuel_Efficiency = gr.Number(label="Fuel Efficiency (km/l) (eg: 10, 15.4, 20)")

    # ---------------- HISTORY ---------------- #
    gr.Markdown("## Maintenance & Usage History")

    with gr.Row():
        with gr.Column():
            Maintenance_History = gr.Dropdown(["good","average","poor"], label="Maintenance History")
            Service_History = gr.Number(label="Service Count (eg: 0, 2, 5)")
            Reported_Issues = gr.Number(label="Reported Issues (eg: 0, 1, 3)")

        with gr.Column():
            Accident_History = gr.Number(label="Accident History (eg: 0, 1, 4)")
            Last_Service_Date = gr.Textbox(label="Last Service Date (DD-MM-YYYY)")
            Warranty_Expiry_Date = gr.Textbox(label="Warranty Expiry Date (DD-MM-YYYY)")

    # ---------------- CONDITION ---------------- #
    gr.Markdown("## Vehicle Condition")

    with gr.Row():
        Tire_Condition = gr.Dropdown(["new","good","worn out"], label="Tire Condition")
        Brake_Condition = gr.Dropdown(["new","good","worn out"], label="Brake Condition")
        Battery_Status = gr.Dropdown(["new","good","weak"], label="Battery Status")

    # ---------------- ACTION ---------------- #
    analyze_btn = gr.Button("Analyze Vehicle", variant="primary")

    output = gr.Markdown(label="Maintenance Report")

    analyze_btn.click(
        fn=analyze_vehicle,
        inputs=[
            Vehicle_Model, Mileage, Maintenance_History, Reported_Issues,
            Vehicle_Age, Fuel_Type, Transmission_Type, Engine_Size,
            Odometer_Reading, Last_Service_Date, Warranty_Expiry_Date,
            Owner_Type, Insurance_Premium, Service_History,
            Accident_History, Fuel_Efficiency, Tire_Condition,
            Brake_Condition, Battery_Status
        ],
        outputs=output,
        show_progress=True
    )

demo.launch()