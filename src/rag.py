from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

guidelines = [
    "Weak battery in electric vehicles requires inspection every 6 months",
    "Worn out tires with mileage greater than 50000 should be replaced immediately",
    "Vehicles with more than 3 accidents require brake system inspection",
    "High mileage vehicles require frequent servicing every 3 months",
    "Old vehicles above 8 years require full inspection",
    "Poor maintenance history increases engine failure risk",
    "Brake pads worn out must be replaced immediately",
    "Low fuel efficiency indicates possible engine or fuel system issues",
    "Vehicles with high insurance premium often indicate higher risk category",
    "Frequent service history suggests heavy usage and higher wear",
    "Manual transmission vehicles may require clutch inspection after heavy use",
    "Electric vehicles require periodic battery health diagnostics",
    "Diesel engines require injector cleaning at regular intervals",
    "Petrol engines may require spark plug replacement periodically",
    "High odometer reading indicates potential wear on internal components",
    "Vehicles with repeated issues need comprehensive inspection",
    "Tire pressure imbalance can lead to uneven wear",
    "Brake fluid must be replaced periodically",
    "Battery older than 3 years should be tested",
    "Low service history increases risk of hidden failures",
    "Frequent short trips can degrade battery health",
    "Heavy load vehicles wear tires faster",
    "Urban driving increases brake wear",
    "High-speed driving increases engine stress",
    "Vehicles exposed to extreme weather need frequent checks",
    "Suspension systems degrade over time and require inspection",
    "Oil change delays increase engine wear",
    "Transmission fluid must be maintained regularly",
    "Cooling system failure leads to overheating risks",
    "Engine knocking indicates serious internal issues",
    "Exhaust issues reduce efficiency and increase emissions",
    "Fuel contamination damages engine performance",
    "Electrical faults may arise from poor battery condition",
    "Brake noise indicates wear and tear",
    "Vibration at high speeds indicates alignment issues",
    "Wheel alignment should be checked periodically",
    "Air filter blockage reduces engine efficiency",
    "Driving patterns influence maintenance needs",
    "Fleet vehicles require stricter maintenance cycles",
    "Older batteries reduce startup reliability",
    "Repeated braking reduces pad life significantly",
    "High RPM driving accelerates engine wear",
    "Oil leakage indicates seal failure",
    "Coolant leakage leads to overheating",
    "Irregular servicing leads to cascading failures",
    "Multiple minor issues can indicate major upcoming failure"
]

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = Chroma.from_texts(
    texts=guidelines,
    embedding=embedding,
    persist_directory="/tmp/chroma_db"
)

def retrieve_guidelines(query):
    docs = vectorstore.similarity_search(query, k=2)
    return [doc.page_content for doc in docs]