import pandas as pd
import random

rows = []

# --------------------
# NORMAL (200 rows)
# --------------------
for _ in range(200):
    cpu = random.randint(15, 60)
    latency = random.randint(20, 120)
    loss = random.randint(0, 2)

    rows.append([
        cpu,
        latency,
        loss,
        "normal"
    ])

# --------------------
# RISK (150 rows)
# --------------------
for _ in range(150):
    cpu = random.randint(55, 85)
    latency = random.randint(120, 220)
    loss = random.randint(2, 5)

    rows.append([
        cpu,
        latency,
        loss,
        "risk"
    ])

# --------------------
# FAILURE (150 rows)
# --------------------
for _ in range(150):
    cpu = random.randint(85, 100)
    latency = random.randint(220, 350)
    loss = random.randint(6, 15)

    rows.append([
        cpu,
        latency,
        loss,
        "failure"
    ])

# Shuffle rows
random.shuffle(rows)

# Create DataFrame
df = pd.DataFrame(
    rows,
    columns=[
        "CPU",
        "Latency",
        "PacketLoss",
        "Status"
    ]
)

# Save dataset
df.to_csv("data/network_data.csv", index=False)

print("=" * 40)
print("Dataset generated successfully!")
print("Total rows:", len(df))
print("Saved to: ../data/network_data.csv")
print("=" * 40)