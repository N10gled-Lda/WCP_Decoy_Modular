import subprocess
import re
import matplotlib.pyplot as plt

import sys
sys.path.append('../..')
key_lengths = [1000]
key_lengths = [100, 1000, 2000, 10000]
bytes_test_values = [5, 10, 20, 50, 100]
bytes_test_values = [10,100,200,500,1000]
bytes_test_values = [10]
runs_per_test = 5
test_results = []

for kl in key_lengths:
    for bpf in bytes_test_values:
        total_detected = total_common = total_testbits = 0
        for _ in range(runs_per_test):
            print(f"Running test with key_length={kl}, bytes_per_frame={bpf}")
            try:
                output = subprocess.check_output(
                    [
                        "python", 
                        "participant_2_bob.py", 
                        "--bytes_per_frame", str(bpf), 
                        "--key_length", str(kl)
                    ],
                    stderr=subprocess.STDOUT
                ).decode("utf-8")
            except subprocess.CalledProcessError as e:
                output = e.output.decode("utf-8")
                exit(1)

            # Look for lines like: "Bob detected indices size: 80 bytes" etc.
            detected_match = re.search(r"Bob detected indices size: (\d+)", output)
            common_match = re.search(r"Bob common indices size: (\d+)", output)
            testbits_match = re.search(r"Bob test bits size: (\d+)", output)

            # Extract sizes (if nb_threads=1 helps produce these lines)
            if detected_match and common_match and testbits_match:
                total_detected += int(detected_match.group(1))
                total_common += int(common_match.group(1))
                total_testbits += int(testbits_match.group(1))

        # Average sizes across runs
        detected_size_avg = total_detected / runs_per_test
        common_size_avg = total_common / runs_per_test
        testbits_size_avg = total_testbits / runs_per_test
        total_size_avg = detected_size_avg + common_size_avg + testbits_size_avg

        test_results.append((kl, bpf, detected_size_avg, common_size_avg, testbits_size_avg, total_size_avg))
        print("Last test results (kl, bpf, detected, common, testbits):", test_results[-1])

        # # Import the participant_2_bob module directly
        # import importlib.util
        
        # # Save original argv
        # original_argv = sys.argv.copy()
        
        # # Set command line arguments for participant_2_bob
        # sys.argv = ["participant_2_bob.py", "--bytes_per_frame", str(bpf), "--key_length", str(kl)]
        
        # try:
        #     # Import the module dynamically
        #     spec = importlib.util.spec_from_file_location("participant_2_bob", "participant_2_bob.py")
        #     participant_2_bob = importlib.util.module_from_spec(spec)
        #     spec.loader.exec_module(participant_2_bob)
            
        #     # Get the values directly from the module
        #     # You'll need to modify participant_2_bob.py to expose these values
        #     detected_size = participant_2_bob.detected_indices_size
        #     common_size = participant_2_bob.common_indices_size
        #     testbits_size = participant_2_bob.test_bits_size
            
        # except Exception as e:
        #     print(f"Error accessing values directly: {e}")
        #     detected_size = common_size = testbits_size = 0
        # finally:
        #     # Restore original argv
        #     sys.argv = original_argv
        
        # test_results.append((bpf, detected_size, common_size, testbits_size))

print("Key Length, BpF, Detec Idx, Common Idx, TestBits")
print(test_results)

# Separate out the data
key_lengths = [res[0] for res in test_results]
bpf_values = [res[1] for res in test_results]
detected_sizes = [res[2] for res in test_results]
common_sizes = [res[3] for res in test_results]
testbits_sizes = [res[4] for res in test_results]
total_sizes = [res[5] for res in test_results]

# Calculate bandwidth
distance = 2000  # in meters
c_light = 299_792_458  # speed of light in meters per second
propagation_delay_d = 2 * distance / c_light
propagation_delay_t = 0.1  # in seconds
total_time = 1  # in seconds
nb_messages = 3
bandwidth = [size / (total_time - nb_messages * propagation_delay_t) for size in total_sizes]  # in bytes per second
bandwidth_kbps = [bw * 8 / 1_000 for bw in bandwidth]  # in kbps

print("Bandwidth (bps):", bandwidth)
print("Bandwidth (kbps):", bandwidth_kbps)

# Plot results
# plt.plot(bpf_values, detected_sizes, label="Detected Indices Size")
# plt.plot(bpf_values, common_sizes, label="Common Indices Size")
# plt.plot(bpf_values, testbits_sizes, label="Test Bits Size")
# plt.xlabel("Bytes per Frame")
# plt.ylabel("Size (bytes)")
# plt.title("Payload Sizes vs. Bytes per Frame")
# plt.legend()
# plt.grid(True)
# plt.show()
# plt.savefig("tests/bandwidth/images/payload_sizes_vs_bytes_per_frame.png")


plt.figure()
plt.plot(key_lengths, [bw * 10**-3 for bw in bandwidth], marker='o', label="Bandwidth Uplink")
plt.xlabel("Key Length (bits)")
plt.ylabel("Bandwidth (Kbytes/s)")
plt.title("Bandwidth Uplink vs. Key Length")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("tests/bandwidth/images/bandwidth_uplink_vs_key_length.png")
plt.show()

# Save data to file
with open("tests/bandwidth/data/bob_payload_sizes_vs_bytes_per_frame.csv", "w") as f:
    f.write("Key Length, Bytes per Frame, Detected Indices Size, Common Indices Size, Test Bits Size, Total Size\n")
    for res in test_results:
        f.write(f"{res[0]}, {res[1]}, {res[2]}, {res[3]}, {res[4]}, {res[5]}\n")

with open("tests/bandwidth/data/bandwidth_uplink_vs_key_length.csv", "w") as f:
    f.write("Key Length, Bandwidth (bps), Bandwidth (kbps)\n")
    for i, kl in enumerate(key_lengths):
        f.write(f"{kl}, {bandwidth[i]}, {bandwidth_kbps[i]}\n")