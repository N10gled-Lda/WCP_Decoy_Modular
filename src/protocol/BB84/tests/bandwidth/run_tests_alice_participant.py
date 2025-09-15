import subprocess
import re
import matplotlib.pyplot as plt

key_lengths = [1000]
key_lengths = [100, 1000, 2000, 10000]
bytes_test_values = [5, 10, 20, 50, 100]
bytes_test_values = [10,100,200,500,1000]
bytes_test_values = [10]
runs_per_test = 5
test_results = []

for kl in key_lengths:
    for bpf in bytes_test_values:
        total_bases = total_test_res = 0
        for _ in range(runs_per_test):
            print(f"Running test with key_length={kl}, bytes_per_frame={bpf}")
            try:
                output = subprocess.check_output(
                    [
                        "python", 
                        "participant_1_alice.py",
                        "--key_length", str(kl),
                        "--bytes_per_frame", str(bpf)
                    ],
                    stderr=subprocess.STDOUT
                ).decode("utf-8")
            except subprocess.CalledProcessError as e:
                print("Error:", e)
                output = e.output.decode("utf-8")
                exit(1)
            
            # Look for lines like: "Alice detected bases size: 80 bytes"
            # and "Alice test result size: 40 bytes"
            bases_match = re.search(r"Alice detected bases size: (\d+)", output)
            test_result_match = re.search(r"Alice test result size: (\d+)", output)

            if bases_match and test_result_match:
                total_bases += int(bases_match.group(1))
                total_test_res += int(test_result_match.group(1))

        avg_bases = total_bases / runs_per_test
        avg_test_res = total_test_res / runs_per_test
        total_size = avg_bases + avg_test_res
        test_results.append((kl, bpf, avg_bases, avg_test_res, total_size))
        print("Last test results (kl, bpf, bases, test_res):", test_results[-1])


print("Key Length, BPF, Detected Bases Size, Test Result Size")
print(test_results)
# Separate out the data
key_lengths = [res[0] for res in test_results]
bpf_values = [res[1] for res in test_results]
bases_sizes = [res[2] for res in test_results]
testres_sizes = [res[3] for res in test_results]
total_sizes = [res[4] for res in test_results]

# Calculate bandwidth
distance = 2000  # in meters
c_light = 299_792_458  # speed of light in meters per second
propagation_delay_d = 2*distance/c_light
propagation_delay_t = 0.1 # in seconds
total_time = 1 # in seconds
nb_messages = 2
bandwidth = [size / (total_time - nb_messages * propagation_delay_t) for size in total_sizes]  # in bytes per second
bandwidth_kbps = [bw * 8 / 1_000 for bw in bandwidth]  # in kbps

print("Bandwidth (bps):", bandwidth)
print("Bandwidth (kbps):", bandwidth_kbps)

GOLDEN_RATIO = (1 + 5 ** 0.5) / 2
# plt.plot(bpf_values, bases_sizes, label="Detected Bases Size")
# plt.plot(bpf_values, testres_sizes, label="Test Result Size")
# plt.xlabel("Bytes per Frame")
# plt.ylabel("Size (bytes)")
# plt.title("Alice Payload Sizes vs. Bytes per Frame")
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.savefig("tests/bandwidth/images/alice_payload_sizes_vs_bytes_per_frame.png")

plt.figure()
plt.plot(key_lengths, [bw * 10**-3 for bw in bandwidth], marker='o', label="Bandwidth Downlink")
plt.xlabel("Key Length (bits)")
plt.ylabel("Bandwidth (Kbytes/s)")
plt.title("Bandwidth Downlink vs. Key Length")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("tests/bandwidth/images/bandwidth_downlink_vs_key_length.png")
plt.show()

# Save data to file
with open("tests/bandwidth/data/alice_payload_sizes_vs_bytes_per_frame.csv", "w") as f:
    f.write("Key Length, Bytes per Frame, Detected Bases Size, Test Result Size, Total Size\n")
    for res in test_results:
        f.write(f"{res[0]}, {res[1]}, {res[2]}, {res[3]}, {res[4]}\n")

with open("tests/bandwidth/data/bandwidth_downlink_vs_key_length.csv", "w") as f:
    f.write("Key Length, Bandwidth (bps), Bandwidth (kbps)\n")
    for i, kl in enumerate(key_lengths):
        f.write(f"{kl}, {bandwidth[i]}, {bandwidth_kbps[i]}\n")
        