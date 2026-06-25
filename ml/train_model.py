import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Dataset load karo
df = pd.read_csv("data/network_data.csv")

# Features aur target
X = df[["CPU", "Latency", "PacketLoss"]]
y = df["Status"]

# Model banao
model = RandomForestClassifier(random_state=42)

# Train karo
model.fit(X, y)

# Save karo
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model Trained Successfully!")