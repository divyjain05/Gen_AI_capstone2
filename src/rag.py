from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

guidelines = [
    "Weak battery in electric vehicles requires inspection every 6 months",
    "Worn out tires with mileage greater than 50000 should be replaced immediately",
    "Vehicles with 3 or more accidents need brake system inspection",
    "Poor maintenance history increases risk of engine failure",
    "High mileage vehicles require frequent servicing",
    "Old vehicles above 8 years need complete inspection",
    "Brake condition marked as worn out requires urgent replacement",
    "Low fuel efficiency may indicate engine issues"
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