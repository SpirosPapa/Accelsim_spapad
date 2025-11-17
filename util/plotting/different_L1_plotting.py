import os
import re
import matplotlib.pyplot as plt

# Set base directory
base_dir = "."

# Pattern to identify relevant folders
folder_prefix = "correl-html_V100_"
filename = "gpc_cycles.V100-SASS-LINEAR-RR-256B-FRFCFS.kernel.raw.csv"

# Data holders
l1_sizes = []
errors = []
correlations = []

for folder in os.listdir(base_dir):
    if folder.startswith(folder_prefix):
        l1_str = folder.replace(folder_prefix, "")
        try:
            l1_size = int(l1_str)
        except ValueError:
            continue  # Skip folders with invalid suffix
        
        filepath = os.path.join(base_dir, folder, filename)
        if not os.path.isfile(filepath):
            print(f"Missing: {filepath}")
            continue

        with open(filepath, "r") as f:
            header_line = f.readline().strip()

            # Example format: V100-SASS-LINEAR... [Correl=0.749 Err=427.96%]
            match = re.search(r"Correl=([0-9.]+)\s+Err=([0-9.]+)%", header_line)
            if match:
                corr = float(match.group(1))
                err = float(match.group(2))
                l1_sizes.append(l1_size)
                errors.append(err)
                correlations.append(corr)
            else:
                print(f"Could not parse correlation/error from: {filepath}")

# Sort all by L1 size
sorted_data = sorted(zip(l1_sizes, errors, correlations))
l1_sizes, errors, correlations = zip(*sorted_data)

# Plot 1: Error vs L1 size
plt.figure(figsize=(10, 6))
plt.bar([str(x) for x in l1_sizes], errors, color="salmon")
plt.xlabel("L1 Cache Size (bytes)")
plt.ylabel("Total Error (%)")
plt.title("Total Error vs L1 Cache Size")
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("error_vs_l1.png")

# Plot 2: Correlation vs L1 size
plt.figure(figsize=(10, 6))
plt.bar([str(x) for x in l1_sizes], correlations, color="skyblue")
plt.xlabel("L1 Cache Size (bytes)")
plt.ylabel("Correlation")
plt.title("Correlation vs L1 Cache Size")
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("correlation_vs_l1.png")

print("Plots saved as error_vs_l1.png and correlation_vs_l1.png")
