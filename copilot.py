import pickle
import ollama
from knowledge import get_runbook

# Model load
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Sample network metrics
cpu = 25
latency = 15
loss = 0

# Prediction
prediction = model.predict([[cpu, latency, loss]])[0]
runbook = get_runbook(cpu, latency , loss)

# Prompt for Phi-3
prompt = f"""
Network Metrics:

CPU={cpu}
Latency={latency}
Loss={loss}

Predicted Status={prediction}

Reference Runbook:

{runbook}

Return answer in format:

Issue:
Cause:
Action:

Maximum 40 words.
"""

response = ollama.chat(
    model="phi3",
    messages=[{"role": "user", "content": prompt}]
)

print("Prediction:", prediction)
print()
print(response["message"]["content"])