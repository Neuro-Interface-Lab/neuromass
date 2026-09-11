import re
import csv

input_file = "tegrastats_kuramoto.log"
output_file = "tegrastats_kuramoto.csv"

data = []

with open(input_file, "r") as f:
    for line in f:
        # Timestamp
        timestamp = re.search(
            r"(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})",
            line
        )

        # RAM
        ram = re.search(r"RAM (\d+)/(\d+)MB", line)

        # GPU power
        gpu_power = re.search(
            r"VDD_GPU (\d+)mW/(\d+)mW/(\d+)mW",
            line
        )

        # CPU/SOC power
        cpu_soc_power = re.search(
            r"VDD_CPU_SOC_MSS (\d+)mW/(\d+)mW/(\d+)mW",
            line
        )

        # System power
        system_power = re.search(
            r"VIN_SYS_5V0 (\d+)mW/(\d+)mW/(\d+)mW",
            line
        )

        # Total input power
        vin_power = re.search(
            r"VIN (\d+)mW/(\d+)mW/(\d+)mW",
            line
        )

        # Temperatures
        gpu_temp = re.search(r"gpu@([\d.]+)C", line)
        cpu_temp = re.search(r"cpu@([\d.]+)C", line)
        tj_temp = re.search(r"tj@([\d.]+)C", line)

        if not timestamp or not ram:
            continue

        row = {
            "timestamp": timestamp.group(1),
            "ram_used_MB": int(ram.group(1)),
            "ram_total_MB": int(ram.group(2)),
            "gpu_power_mW": int(gpu_power.group(1)) if gpu_power else None,
            "cpu_soc_power_mW": int(cpu_soc_power.group(1)) if cpu_soc_power else None,
            "system_power_mW": int(system_power.group(1)) if system_power else None,
            "vin_power_mW": int(vin_power.group(1)) if vin_power else None,
            "gpu_temp_C": float(gpu_temp.group(1)) if gpu_temp else None,
            "cpu_temp_C": float(cpu_temp.group(1)) if cpu_temp else None,
            "tj_temp_C": float(tj_temp.group(1)) if tj_temp else None,
        }

        data.append(row)

fieldnames = data[0].keys()

with open(output_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print(f"{len(data)} mesures extraites")
print(f"Fichier créé : {output_file}")