import argparse 
import time
import numpy as np
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bb84_protocol.old_version.alice_participant import Alice
from bb84_protocol.old_version.bob_participant import Bob

GOLDEN_RATIO = 1.618

# Parameters Constant
host = 'localhost'
port = 12345

loss_rate = 1 - 0.0
do_test = True
test_fraction = 0.9999
error_threshold = 0.1

# Parameters to test default values
key_length = 5000
qubit_freq_us = 200
sync_frames = 50
sync_bytes = 20


bytes_per_frame = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150]
# bytes_per_frame = [80,90]
num_frames = [key_length // bpf for bpf in bytes_per_frame]

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
        for bpf, nf in zip(bytes_per_frame, num_frames):
            print(f"Running test for bytes_per_frame: {bpf}, num_frames: {nf}")
            for i in range(num_iterations):
                time.sleep(0.5)
                alice = Alice(num_qubits=key_length,
                    num_frames=nf,
                    bytes_per_frame=bpf,
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
        bob_failed_percentages_bpf_iter = []
        bob_failed_percentages_bpf_mean = []
        bob_times_m_delay_bpf_iter = []
        bob_times_bins_bpf_iter = []
        bob_times_bins_bpf_mean = []
        bob_times_bins_bpf_sigma = []
        bob_times_bins_bpf_mean_mean = []
        bob_times_bins_bpf_mean_sigma = []
        bob_times_bins_bpf_sigma_mean = []
        bob_times_bins_bpf_sigma_sigma = []
        
        for bpf, nf in zip(bytes_per_frame, num_frames):
            print(f"Running test for bytes_per_frame: {bpf}, num_frames: {nf}")
            bob_failed_percentages_iter = []
            bob_times_bins_iter = []
            times_m_delay_iter = []
            bob_times_bins_iter_mean = []
            bob_times_bins_iter_sigma = []
            for i in range(num_iterations):
                time.sleep(0.5)
                bob = Bob(num_qubits=key_length,
                    num_frames=nf,
                    bytes_per_frame=bpf,
                    qubit_delay_us= qubit_freq_us,
                    sync_frames=sync_frames,
                    sync_bytes_per_frame=sync_bytes,
                    test_bool=do_test,
                    test_fraction=test_fraction,
                    )
                bob.run_bob()
                bob_failed_percentages_iter.append(bob.failed_percentage)
                print("Iteration %d/%d for bpf - %d " % (i + 1, num_iterations, bpf))
                print("Failed Percentage: %f" % bob.failed_percentage)
                times = bob.detected_timestamps
                bob_times_bins_temp = []
                times_m_delay = []
                # Calculate the time bins
                for frame in range(nf):
                    for j in range(len(times) // nf - 1):
                        times_m_delay.append(times[j + frame * bpf] - (j + 1)*bob.time_bin)
                        time_bins_val = times[j + 1 + frame * bpf] - times[j + frame * bpf]
                        bob_times_bins_temp.append(time_bins_val)
                bob_times_bins_iter.append(bob_times_bins_temp)
                times_m_delay_iter.append(times_m_delay)
                bob_times_bins_iter_mean.append(float(np.mean(bob_times_bins_temp)))
                bob_times_bins_iter_sigma.append(float(np.std(bob_times_bins_temp)))

            # Save in list to all in one file - NOT DONE CURRENTLY
            bob_failed_percentages_bpf_iter.append(bob_failed_percentages_iter)
            bob_failed_percentages_bpf_mean.append(float(np.mean(bob_failed_percentages_iter)))
            
            bob_times_bins_bpf_iter.append(bob_times_bins_iter)
            bob_times_m_delay_bpf_iter.append(times_m_delay_iter)
            bob_times_bins_bpf_mean.append(bob_times_bins_iter_mean)
            bob_times_bins_bpf_sigma.append(bob_times_bins_iter_sigma)

            bob_times_bins_bpf_mean_mean.append(float(np.mean(bob_times_bins_iter_mean)))
            bob_times_bins_bpf_mean_sigma.append(float(np.std(bob_times_bins_iter_mean)))
            bob_times_bins_bpf_sigma_mean.append(float(np.mean(bob_times_bins_iter_sigma)))
            bob_times_bins_bpf_sigma_sigma.append(float(np.std(bob_times_bins_iter_sigma)))
            
            # save the results to a file
            path = f"results/bpf_10_100/bob_fails_bpf_{bpf}_iter_{num_iterations}.txt"
            path2 = f"results/bpf_10_100/bob_times_bpf_{bpf}_iter_{num_iterations}.txt"
            with open(path, "w") as f:
                f.write("Bytes_per_frame\tFailed_Percentage\tMean_Failed_Percentage\n")
                f.write(f"{bpf}\t{bob_failed_percentages_iter}\t{bob_failed_percentages_bpf_mean[-1]}\n")
            with open(path2, "w") as f:
                f.write("Bytes_per_frame\tMean_Time_Bins\tSigma_Time_Bins\tMean_Mean_Time_Bins\tSigma_Mean_Time_Bins\tMean_Sigma_Time_Bins\tSigma_Sigma_Time_Bins\n")
                f.write(f"{bpf}\t{bob_times_bins_iter_mean}\t{bob_times_bins_iter_sigma}\t{bob_times_bins_bpf_mean_mean[-1]}\t{bob_times_bins_bpf_mean_sigma[-1]}\t{bob_times_bins_bpf_sigma_mean[-1]}\t{bob_times_bins_bpf_sigma_sigma[-1]}\n")
                f.write("Bytes_per_frame\tMean_Time_Delay\n")
                f.write(f"{bpf}\t{times_m_delay_iter}\n")
                f.write("Bytes_per_frame\tTime_Bins_All\n")
                f.write(f"{bpf}\t{bob_times_bins_iter}\n")

if do_graph and role == "bob":

    # Reaf files if not run test
    if not run_test:
        bob_failed_percentages_bpf_iter = []
        bob_failed_percentages_bpf_mean = []
        bob_times_m_delay_bpf_iter = []
        bob_times_bins_bpf_iter = []
        bob_times_bins_bpf_mean = []
        bob_times_bins_bpf_sigma = []
        bob_times_bins_bpf_mean_mean = []
        bob_times_bins_bpf_mean_sigma = []
        bob_times_bins_bpf_sigma_mean = []
        bob_times_bins_bpf_sigma_sigma = []

        for bpf in bytes_per_frame:
            path = f"results/bpf_10_100/bob_fails_bpf_{bpf}_iter_{num_iterations}.txt"
            path2 = f"results/bpf_10_100/bob_times_bpf_{bpf}_iter_{num_iterations}.txt"
            print(f"Reading from file: {path} and {path2}")
            with open(path, "r") as f:
                f.readline()
                line = f.readline()
                bpf, failed_percentages, mean_failed_percentage = map(eval, line.split("\t"))
                bob_failed_percentages_bpf_iter.append(failed_percentages)
                bob_failed_percentages_bpf_mean.append(mean_failed_percentage)

            with open(path2, "r") as f:
                f.readline()
                line = f.readline()
                bpf, mean_time_bins, sigma_time_bins, mean_mean_time_bins, sigma_mean_time_bins, mean_sigma_time_bins, sigma_sigma_time_bins = map(eval, line.split("\t"))
                bob_times_bins_bpf_mean.append(mean_time_bins)
                bob_times_bins_bpf_sigma.append(sigma_time_bins)
                bob_times_bins_bpf_mean_mean.append(mean_mean_time_bins)
                bob_times_bins_bpf_sigma_mean.append(sigma_mean_time_bins)
                bob_times_bins_bpf_mean_sigma.append(mean_sigma_time_bins)
                bob_times_bins_bpf_sigma_sigma.append(sigma_sigma_time_bins)
                f.readline()
                line = f.readline()
                bpf, times_m_delay = map(eval, line.split("\t"))
                bob_times_m_delay_bpf_iter.append(times_m_delay)
                f.readline()
                line = f.readline()
                # bpf, time_bins_all = map(eval, line.split("\t"))
                # bob_times_bins_bpf_iter.append(time_bins_all)

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
    for i, bpf in enumerate(bytes_per_frame):
        # Data
        data_points = bob_failed_percentages_bpf_iter[i]
        data_points_mean = bob_failed_percentages_bpf_mean[i]

        # Plot individual points
        ax.scatter(
            np.full(num_iterations, bpf),
            data_points,
            color=colors[i],
            alpha=0.5,
            edgecolor='none',
            s=15,
            label=f"Point for bpf: {bpf}"
        )
        # Plot boxplot
        ax.boxplot(
            data_points,
            positions=[bpf],
            widths=4,
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
            bpf,
            data_points_mean,
            marker='o',
            color=colors_darker[i],
            markersize=10,
            label=f"Mean Failed % for bpf: {bpf}"
        )
        # Add a standard deviation line
        ax.errorbar(
            bpf,
            np.mean(data_points),
            yerr=np.std(data_points),
            fmt='none',
            elinewidth=2,
            alpha=0.35,
            capsize=12,
            ecolor=colors_darker[i],
        )

    ax.set_title("Failed Percentages for Bob")
    ax.set_xlabel("Bytes per Frame")
    ax.set_ylabel("Failed Percentage")
    ax.legend(prop={'size': 8})
    ax.set_xlim(min(bytes_per_frame)-5, max(bytes_per_frame) + 10)
    plt.tight_layout()

    # Plotting the time bins means and sigmas
    fig2, ax2 = plt.subplots(figsize=(10, 10/ GOLDEN_RATIO))

    for i, bpf in enumerate(bytes_per_frame):
        # Data
        data_points_mean = bob_times_bins_bpf_mean[i]
        data_points_sigma = bob_times_bins_bpf_sigma[i]
        
        # scatter plot
        ax2.scatter(
            np.full(num_iterations, bpf),
            data_points_mean,
            color=colors[i],
            alpha=0.5,
            edgecolor='none',
            s=15,
            label=f"Mean Time Bins for bpf: {bpf}"
        )
        # box plot
        ax2.boxplot(
            data_points_mean,
            positions=[bpf],
            widths=10,
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
            bpf,
            np.mean(data_points_mean),
            marker='o',
            color=colors_darker[i],
            markersize=10,
            label=f"Mean Mean Time Bins for bpf: {bpf}"
        )
        # plot sigma error
        # ax2.errorbar(
        #     bpf,
        #     np.mean(data_points_mean),
        #     yerr=np.mean(data_points_sigma),
        #     fmt='none',
        #     color=colors_darker[i],
        #     markersize=10,
        #     label=f"Mean Sigma Time Bins for bpf: {bpf}"
        # )

    ax2.set_title("Time Bins Means for Bob")
    ax2.set_xlabel("Bytes per Frame")
    ax2.set_ylabel("Time Bins Mean")
    ax2.legend(prop={'size': 8})
    ax2.set_xlim(min(bytes_per_frame)-5, max(bytes_per_frame) + 10)
    plt.tight_layout()
    
    fig3, ax3 = plt.subplots(figsize=(10, 10/ GOLDEN_RATIO))

    ax3_slider = plt.axes([0.15, 0.02, 0.75, 0.03])
    slider = widgets.Slider(
        ax=ax3_slider,
        label='Bytes per Frame',
        valmin=0,
        valmax=len(bytes_per_frame) - 1,
        valinit=0,
        valstep=1,
    )
    # Vertical red lines at half the qubit frequency for both sides
    print(qubit_freq_us*1000/2)
    def update(val):
        idx = int(slider.val)
        ax3.clear()
        # Flatten the list of lists into a single array
        flattened_times = np.concatenate(bob_times_m_delay_bpf_iter[idx])
        nbins = 50
        hmin = np.mean(flattened_times) - 3 * np.std(flattened_times)
        hmax = np.mean(flattened_times) + 3 * np.std(flattened_times)
        ax3.hist(flattened_times, bins=np.linspace(hmin,hmax,nbins), color=colors[idx], alpha=0.5, edgecolor='black', label=f"Time -TB for bpf: {bytes_per_frame[idx]}")
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e3:.0f}'))
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
        ax3.set_title(f"Time Bins Histogram for Bob - Bytes per Frame: {bytes_per_frame[idx]}")
        ax3.set_xlabel("Times - TB (us)")
        ax3.set_ylabel("Frequency")
        # ax3.set_yscale('log')
        ax3.axvline(x=qubit_freq_us*1000/2, color='red', linestyle='--', label='Qubit Frequency')
        ax3.axvline(x=-qubit_freq_us*1000/2, color='red', linestyle='--')
        failure_text = f"Failed Percentage: {bob_failed_percentages_bpf_mean[idx]:.2f}"
        ax3.text(0.02, 0.95, failure_text, transform=ax3.transAxes, fontsize=12,
                verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))
        ax3.legend()
        plt.draw()

    update(0)
    slider.on_changed(update)
    plt.tight_layout()
    plt.show()
