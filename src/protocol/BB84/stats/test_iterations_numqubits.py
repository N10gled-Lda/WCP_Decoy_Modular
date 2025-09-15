import argparse 
import time
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bb84_protocol.old_version.alice_participant import Alice
from bb84_protocol.old_version.bob_participant import Bob

GOLDEN_RATIO = 1.61803398875

# Parameters Constant
host = 'localhost'
port = 12345

loss_rate = 1 - 0.0
do_test = True
test_fraction = 0.9999
error_threshold = 0.1

# Parameters to test default values
bytes_per_frame = 50
qubit_freq_us = 200
sync_frames = 50
sync_bytes = 20


key_length = [100, 1000, 5000, 10000, 50000, 100000]
num_frames = [key // bytes_per_frame for key in key_length]

argparser = argparse.ArgumentParser(description='Test various qubit frequencies each with multiple iterations')
argparser.add_argument('--role', '-r',  type=str, default='bob', help='Role of participant: alice or bob')
argparser.add_argument('--num_iterations', '-nt', type=int, default=10, help='Number of iterations to run')
argparser.add_argument('--run_test', '-rt', default=True, help='Run test or just graph')
argparser.add_argument('--do_graph', '-dg', default=True, help='Show the graph')

args = argparser.parse_args()
role = args.role.lower()
run_test = False if args.run_test == "False" else True
num_iterations = args.num_iterations
do_graph = False if args.do_graph == "False" else True


if role not in ['alice', 'bob']:
    print("Role should be either alice or bob")
    exit()


if run_test:
    if role == "alice":
        for key, nf in zip(key_length, num_frames):
            print(f"Running test for key length: {key} and num_frames: {nf}")
            for i in range(num_iterations):
                time.sleep(0.5)
                alice = Alice(num_qubits=key,
                    num_frames=nf,
                    bytes_per_frame=bytes_per_frame,
                    qubit_delay_us= qubit_freq_us,
                    sync_frames=sync_frames,
                    sync_bytes_per_frame=sync_bytes,
                    loss_rate=loss_rate,
                    test_bool=do_test,
                    test_fraction=test_fraction,
                    error_threshold=error_threshold
                    )
                alice.run_alice()





    if role == "bob":
        bob_failed_percentages_key_iter = []
        bob_failed_percentages_key_mean = []
        bob_times_m_delay_key_iter = []
        bob_times_bins_key_iter = []
        bob_times_bins_key_mean = []
        bob_times_bins_key_sigma = []
        bob_times_bins_key_mean_mean = []
        bob_times_bins_key_mean_sigma = []
        bob_times_bins_key_sigma_mean = []
        bob_times_bins_key_sigma_sigma = []
        
        for key, nf in zip(key_length, num_frames):
            print(f"Running test for key_length: {key}, num_frames: {nf}")
            bob_failed_percentages_iter = []
            bob_times_bins_iter = []
            times_m_delay_iter = []
            bob_times_bins_iter_mean = []
            bob_times_bins_iter_sigma = []
            for i in range(num_iterations):
                time.sleep(0.7)
                bob = Bob(num_qubits=key,
                    num_frames=nf,
                    bytes_per_frame=bytes_per_frame,
                    qubit_delay_us= qubit_freq_us,
                    sync_frames=sync_frames,
                    sync_bytes_per_frame=sync_bytes,
                    test_bool=do_test,
                    test_fraction=test_fraction,
                    )
                bob.run_bob()
                bob_failed_percentages_iter.append(bob.failed_percentage)
                print("Iteration %d/%d for key - %d " % (i + 1, num_iterations, key))
                print("Failed Percentage: %f" % bob.failed_percentage)
                times = bob.detected_timestamps
                bob_times_bins_temp = []
                times_m_delay = []
                # Calculate the time bins
                for frame in range(nf):
                    for j in range(len(times) // nf - 1):
                        times_m_delay.append(times[j + frame * bytes_per_frame] - (j + 1)*bob.time_bin)
                        time_bins_val = times[j + 1 + frame * bytes_per_frame] - times[j + frame * bytes_per_frame]
                        bob_times_bins_temp.append(time_bins_val)
                bob_times_bins_iter.append(bob_times_bins_temp)
                times_m_delay_iter.append(times_m_delay)
                bob_times_bins_iter_mean.append(float(np.mean(bob_times_bins_temp)))
                bob_times_bins_iter_sigma.append(float(np.std(bob_times_bins_temp)))

            # Save in list to all in one file - NOT DONE CURRENTLY
            bob_failed_percentages_key_iter.append(bob_failed_percentages_iter)
            bob_failed_percentages_key_mean.append(float(np.mean(bob_failed_percentages_iter)))
            
            bob_times_bins_key_iter.append(bob_times_bins_iter)
            bob_times_m_delay_key_iter.append(times_m_delay_iter)
            bob_times_bins_key_mean.append(bob_times_bins_iter_mean)
            bob_times_bins_key_sigma.append(bob_times_bins_iter_sigma)

            bob_times_bins_key_mean_mean.append(float(np.mean(bob_times_bins_iter_mean)))
            bob_times_bins_key_mean_sigma.append(float(np.std(bob_times_bins_iter_mean)))
            bob_times_bins_key_sigma_mean.append(float(np.mean(bob_times_bins_iter_sigma)))
            bob_times_bins_key_sigma_sigma.append(float(np.std(bob_times_bins_iter_sigma)))

            # save the results to a file
            path = f"results/key_100_100000/bob_fails_key_{key}_iter_{num_iterations}.txt"
            path2 = f"results/key_100_100000/bob_times_key_{key}_iter_{num_iterations}.txt"
            with open(path, "w") as f:
                f.write("Key_Length\tFailed_Percentage\tMean_Failed_Percentage\n")
                f.write(f"{key}\t{bob_failed_percentages_iter}\t{bob_failed_percentages_key_mean[-1]}\n")
            with open(path2, "w") as f:
                f.write("Key_Length\tMean_Time_Bins\tSigma_Time_Bins\tMean_Mean_Time_Bins\tSigma_Mean_Time_Bins\tMean_Sigma_Time_Bins\tSigma_Sigma_Time_Bins\n")
                f.write(f"{key}\t{bob_times_bins_iter_mean}\t{bob_times_bins_iter_sigma}\t{bob_times_bins_key_mean_mean[-1]}\t{bob_times_bins_key_mean_sigma[-1]}\t{bob_times_bins_key_sigma_mean[-1]}\t{bob_times_bins_key_sigma_sigma[-1]}\n")
                f.write("Key_Length\tMean_Time_Delay\n")
                f.write(f"{key}\t{times_m_delay_iter}\n")
                f.write("Key_Length\tTime_Bins_All\n")
                f.write(f"{key}\t{bob_times_bins_iter}\n")




if do_graph and role == "bob":

    # Reaf files if not run test
    if not run_test:
        bob_failed_percentages_key_iter = []
        bob_failed_percentages_key_mean = []
        bob_times_m_delay_key_iter = []
        bob_times_bins_key_iter = []
        bob_times_bins_key_mean = []
        bob_times_bins_key_sigma = []
        bob_times_bins_key_mean_mean = []
        bob_times_bins_key_mean_sigma = []
        bob_times_bins_key_sigma_mean = []
        bob_times_bins_key_sigma_sigma = []

        for key in key_length:
            path = f"results/key_100_100000/bob_fails_key_{key}_iter_{num_iterations}.txt"
            path2 = f"results/key_100_100000/bob_times_key_{key}_iter_{num_iterations}.txt"
            print(f"Reading from file: {path} and {path2}")
            with open(path, "r") as f:
                f.readline()
                line = f.readline()
                key, failed_percentages, mean_failed_percentage = map(eval, line.split("\t"))
                bob_failed_percentages_key_iter.append(failed_percentages)
                bob_failed_percentages_key_mean.append(mean_failed_percentage)

            with open(path2, "r") as f:
                f.readline()
                line = f.readline()
                key, mean_time_bins, sigma_time_bins, mean_mean_time_bins, sigma_mean_time_bins, mean_sigma_time_bins, sigma_sigma_time_bins = map(eval, line.split("\t"))
                bob_times_bins_key_mean.append(mean_time_bins)
                bob_times_bins_key_sigma.append(sigma_time_bins)
                bob_times_bins_key_mean_mean.append(mean_mean_time_bins)
                bob_times_bins_key_sigma_mean.append(sigma_mean_time_bins)
                bob_times_bins_key_mean_sigma.append(mean_sigma_time_bins)
                bob_times_bins_key_sigma_sigma.append(sigma_sigma_time_bins)
                f.readline()
                line = f.readline()
                # key, times_m_delay = map(eval, line.split("\t"))
                # bob_times_m_delay_key_iter.append(times_m_delay)
                f.readline()
                line = f.readline()
                # key, time_bins_all = map(eval, line.split("\t"))
                # bob_times_bins_key_iter.append(time_bins_all)

    # Plotting
    import matplotlib.pyplot as plt
    import matplotlib.widgets as widgets

    # Colors
    colors = ['red', 'blue', 'green', 'orange', 'plum', 'cyan', 'silver']
    # dark colors
    colors_darker = ['darkred', 'darkblue', 'darkgreen', 'darkorange', 'purple', 'teal', 'dimgray']
    colors = ['red', 'blue', 'lightgreen', 'orange', 'plum', 'cyan', 'silver', 'crimson', 'cornflowerblue', 'lime', 'yellow','pink']
    colors_darker = ['darkred', 'darkblue', 'green', 'darkorange', 'purple', 'teal', 'dimgray', 'maroon', 'midnightblue', 'seagreen', 'olive','hotpink']

    # Plotting the failed percentages
    fig, ax = plt.subplots(figsize=(10, 10/ GOLDEN_RATIO))
    for i, key in enumerate(key_length):
        # Data
        data_points = bob_failed_percentages_key_iter[i]
        data_points_mean = bob_failed_percentages_key_mean[i]
    
        # Plot individual points
        ax.scatter(
            np.full(num_iterations, key),
            data_points,
            color=colors[i],
            alpha=0.5,
            edgecolor='none',
            s=15,
            label=f"Point for key: {key}"
        )
        # Plot boxplot
        ax.boxplot(
            data_points,
            positions=[key],
            widths=key/7,  # Width scales with the x-value
            patch_artist=True,
            meanline=True,
            boxprops=dict(facecolor=colors_darker[i], color=colors[i], alpha=0.5),
            whiskerprops=dict(color='black'),
            flierprops=dict(color=colors[i], markeredgecolor=colors[i]),
            medianprops=dict(color='black'),
            showfliers=False
        )
        # Plot mean
        ax.plot(
            key,
            data_points_mean,
            marker='o',
            color=colors_darker[i],
            markersize=10,
            label=f"Mean Failed % for key: {key}"
        )
        
        # Add a standard deviation line
        ax.errorbar(
            key,
            np.mean(data_points),
            yerr=np.std(data_points),
            fmt='none',
            elinewidth=2,
            alpha=0.35,
            capsize=12,
            ecolor=colors_darker[i],
        )

    ax.set_title("Failed Percentages for Bob")
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Failed Percentage")
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())    
    ax.legend(loc='upper right', prop={'size': 8})
    ax.set_xlim(min(key_length) - 20, max(key_length) + 10000)
    plt.tight_layout()

    # Plotting the time bins
    fig2, ax2 = plt.subplots(figsize=(10, 10/ GOLDEN_RATIO))

    for i, key in enumerate(key_length):
        # Data
        # data_points = bob_times_bins_key_iter[i]
        data_points_mean = bob_times_bins_key_mean[i]
        data_points_sigma = bob_times_bins_key_sigma[i]

        # scatter
        ax2.scatter(
            np.full(num_iterations, key),
            data_points_mean,
            color=colors[i],
            alpha=0.5,
            edgecolor='none',
            s=15,
            label=f"Mean Time Bins for key: {key}"
        )

        # boxplot
        ax2.boxplot(
            data_points_mean,
            positions=[key],
            widths=key/7,
            patch_artist=True,
            meanline=True,
            boxprops=dict(facecolor=colors_darker[i], color=colors[i], alpha=0.5),
            whiskerprops=dict(color='black'),
            flierprops=dict(color=colors[i], markeredgecolor=colors[i]),
            medianprops=dict(color='black'),
            showfliers=False
        )

        # Plot mean
        ax2.plot(
            key,
            np.mean(data_points_mean),
            marker='o',
            color=colors_darker[i],
            markersize=10,
            label=f"Mean of Mean Time Bins for key: {key}"
        )

    ax2.set_title("Time Bins for Bob")
    ax2.set_xlabel("Number of Qubits")
    ax2.set_ylabel("Time Bins")
    ax2.legend()
    ax2.set_xlim(min(key_length) - 20, max(key_length) + 10000)
    ax2.set_xscale('log')
    ax2.xaxis.set_major_formatter(plt.ScalarFormatter())

    # UNCOMMENT TO PLOT THE HISTOGRAM AND THE bob_times_m_delay_key_iter FILE ABOVE
    # fig3, ax3 = plt.subplots(figsize=(10, 10/ GOLDEN_RATIO))

    # ax3_slider = plt.axes([0.15, 0.02, 0.75, 0.03])
    # slider = widgets.Slider(
    #     ax=ax3_slider,
    #     label='Number of Qubits',
    #     valmin=0,
    #     valmax=len(key_length) - 1,
    #     valinit=0,
    #     valstep=1,
    # )

    # def update(val):
    #     idx = int(slider.val)
    #     ax3.clear()
    #     # Flatten the list of lists into a single array
    #     flattened_times = np.concatenate(bob_times_m_delay_key_iter[idx])
    #     nbins = 50
    #     hmin = np.mean(flattened_times) - 3 * np.std(flattened_times)
    #     hmax = np.mean(flattened_times) + 3 * np.std(flattened_times)
    #     ax3.hist(flattened_times, bins=np.linspace(hmin,hmax,nbins), color=colors[idx], alpha=0.5, edgecolor='black', label=f"Time -TB for nq: {key_length[idx]}")
    #     ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e3:.0f}'))
    #     ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    #     ax3.set_title(f"Time Bins Histogram for Bob - Number of Qubits: {key_length[idx]}")
    #     ax3.set_xlabel("Times - TB (us)")
    #     ax3.set_ylabel("Frequency")
    #     # ax3.set_yscale('log')
    #     ax3.axvline(x=qubit_freq_us*1000/2, color='red', linestyle='--', label='Qubit Frequency')
    #     ax3.axvline(x=-qubit_freq_us*1000/2, color='red', linestyle='--')
    #     failure_text = f"Failed Percentage: {bob_failed_percentages_key_mean[idx]:.2f}"
    #     ax3.text(0.02, 0.95, failure_text, transform=ax3.transAxes, fontsize=12,
    #             verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))
    #     ax3.legend()
    #     plt.draw()

    # update(0)
    # slider.on_changed(update)

    plt.tight_layout()
    plt.show()