# plot the bandwidth based on the data saved from alice and bob

import matplotlib.pyplot as plt
import sys

GOLDEN_RATIO = (1 + 5 ** 0.5) / 2

# import data from csv file
def import_data(file_name):
    data = []
    with open(file_name, 'r') as file:
        for line in file:
            data.append(line.strip().split(','))
    return data

DEFAULT_UPLINK_FILENAME = 'data/bandwidth_uplink_vs_key_length.csv'
DEFAULT_DOWNLINK_FILENAME = 'data/bandwidth_downlink_vs_key_length.csv'
uplink_filename = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_UPLINK_FILENAME
downlink_filename = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DOWNLINK_FILENAME

uplink_data = import_data(uplink_filename)
downlink_data = import_data(downlink_filename)

# Separate out the data
uplink_key_lengths = [int(res[0]) for res in uplink_data[1:]]
uplink_bandwidth_byps = [float(res[1]) for res in uplink_data[1:]]
uplink_bandwidth_kbps = [float(res[2]) for res in uplink_data[1:]]
downlink_key_lengths = [int(res[0]) for res in downlink_data[1:]]
downlink_bandwidth_byps = [float(res[1]) for res in downlink_data[1:]]
downlink_bandwidth_kbps = [float(res[2]) for res in downlink_data[1:]]

# Plot results
def plot_bandwidth_vs_key_length(key_lengths, bandwidth_kbps, title, filename):
    plt.figure(figsize=(GOLDEN_RATIO * 6, 6))
    plt.plot(key_lengths, bandwidth_kbps, marker='o', label="Bandwidth")
    plt.xlabel("Key Length (bits)")
    plt.ylabel("Bandwidth (Kbits/s)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

plot_bandwidth_vs_key_length(uplink_key_lengths, uplink_bandwidth_kbps, "Bandwidth Uplink vs. Key Length", "images/bandwidth_uplink_vs_key_length.png")
plot_bandwidth_vs_key_length(downlink_key_lengths, downlink_bandwidth_kbps, "Bandwidth Downlink vs. Key Length", "images/bandwidth_downlink_vs_key_length.png")
