import requests
import pandas as pd
import numpy as np
import random
import json

# === Load your data ===
df = pd.read_csv("final_table.csv")  # Replace with your actual dataset path

# === Drop the target column ===
if "TARGET" in df.columns:
    df = df.drop(columns=["TARGET"])

# === Pick a random row as a sample ===
sample_row = df.sample(n=1, random_state=42).iloc[0]

# === Convert to dict format ===
sample_dict = sample_row.to_dict()

# Convert NaNs to None
sample_dict = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in sample_dict.items()}


# === Send to FastAPI ===
url = "https://bank-default-analysis-latest.onrender.com/predict"

payload = {
    "data": sample_dict
}

response = requests.post(url, json=payload)

# === Print response ===
print("Status code:", response.status_code)
print("Response:", json.dumps(response.json(), indent=2))
