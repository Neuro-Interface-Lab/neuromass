import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

fichier = "tegrastats.log"
data = []

with open(fichier) as f:
    for line in f:
        m = re.match(r"(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})", line)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%d-%m-%Y %H:%M:%S")

        pwr  = re.search(r"VDD_GPU (\d+)mW", line)
        temp = re.search(r"gpu@([\d.]+)C", line)
        ram  = re.search(r"RAM (\d+)/\d+MB", line)

        data.append({
            "timestamp": ts,
            "gpu_power_W": int(pwr.group(1))/1000 if pwr else None,
            "gpu_temp_C": float(temp.group(1)) if temp else None,
            "ram_GB": int(ram.group(1))/1024 if ram else None,
        })

df = pd.DataFrame(data)
df["temps_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()

fig, axes = plt.subplots(3, 1, figsize=(6, 5), sharex=True)

axes[0].plot(df["temps_s"], df["gpu_power_W"], color="red")
axes[0].set_ylabel("Puissance GPU (W)")
axes[0].set_title("Puissance GPU en fonction du temps")
axes[0].grid(True, alpha=0.3)

axes[1].plot(df["temps_s"], df["gpu_temp_C"], color="orange")
axes[1].set_ylabel("Température GPU (°C)")
axes[1].set_title("Température GPU en fonction du temps")
axes[1].grid(True, alpha=0.3)

axes[2].plot(df["temps_s"], df["ram_GB"], color="blue")
axes[2].set_ylabel("RAM (GB)")
axes[2].set_xlabel("Temps (s)")
axes[2].set_title("RAM utilisée en fonction du temps")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("tegrastats_time_series.png", dpi=150, bbox_inches="tight")
plt.show()

df.to_csv("tegrastats_time_series.csv", index=False)
print("Créés : tegrastats_time_series.png et .csv")