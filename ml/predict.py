import pickle

# Model load karo
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# New network data
cpu = 92
latency = 240
loss = 9

prediction = model.predict([[cpu, latency, loss]])

print("Prediction:", prediction[0])