import joblib
import pandas as pd

data = joblib.load("decision_tree_balanced.pkl")
model = data["model"]
train_columns = data["columns"]

def predict_maintenance(input_dict):
    df = pd.DataFrame([input_dict])
    df = pd.get_dummies(df)
    df = df.reindex(columns=train_columns, fill_value=0)
    return int(model.predict(df)[0])