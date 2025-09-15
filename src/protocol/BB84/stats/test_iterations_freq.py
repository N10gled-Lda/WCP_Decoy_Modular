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
bytes_per_frame = 50
num_frames = key_length // bytes_per_frame
sync_frames = 50
sync_bytes = 20

qubit_freq_us = [100,200,300, 400,500,700,1000]
qubit_freq_us = [600,800,900]
qubit_freq_us = [100,200,300,400,500,600,700,800,900,1000]

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
        for qubit_delay in qubit_freq_us:
            for i in range(num_iterations):
                time.sleep(0.5)
                alice = Alice(num_qubits=key_length,
                    num_frames=num_frames,
                    qubit_delay_us= qubit_delay,
                    bytes_per_frame=bytes_per_frame,
                    sync_frames=sync_frames,
                    sync_bytes_per_frame=sync_bytes,
                    loss_rate=loss_rate,
                    test_bool=do_test,
                    test_fraction=test_fraction,
                    error_threshold=error_threshold
                    )
                alice.run_alice()

    if role == "bob":
        bob_failed_percentages_freq_mean = []
        bob_failed_percentages_freq_iter = []
        bob_times_bins_freq_mean = []
        bob_times_bins_freq_sigma = []
        bob_times_bins_freq_iter = []
        for qubit_delay in qubit_freq_us:
            bob_failed_percentages_iter = []
            bob_times_bins_iter_mean = []
            bob_times_bins_iter_sigma = []
            bob_times_bins_iter = []
            for i in range(num_iterations):
                bob_times_bins_temp = []
                time.sleep(0.5)
                bob = Bob(num_qubits=key_length,
                    num_frames=num_frames,
                    qubit_delay_us= qubit_delay,
                    bytes_per_frame=bytes_per_frame,
                    sync_frames=sync_frames,
                    sync_bytes_per_frame=sync_bytes,
                    test_bool=do_test,
                    test_fraction=test_fraction,
                    )
                bob.run_bob()
                bob_failed_percentages_iter.append(bob.failed_percentage)
                print("Iteration %d/%d for qubit frequency %d us" % (i + 1, num_iterations, qubit_delay))
                print("Failed Percentage: %f" % bob.failed_percentage)
                times = bob.detected_timestamps
                # Calculate the time bins
                for frame in range(num_frames):
                    for j in range(len(times) // num_frames - 1):
                        time_bins_val = times[j + 1 + frame * bytes_per_frame] - times[j + frame * bytes_per_frame]
                        bob_times_bins_temp.append(time_bins_val)
                bob_times_bins_iter.append(bob_times_bins_temp)
                bob_times_bins_iter_mean.append(float(np.mean(bob_times_bins_temp)))
                bob_times_bins_iter_sigma.append(float(np.std(bob_times_bins_temp)))

            # Make mean/save of all iterations
            bob_failed_percentages_freq_mean.append(sum(bob_failed_percentages_iter) / num_iterations)
            bob_failed_percentages_freq_iter.append(bob_failed_percentages_iter)
            bob_times_bins_freq_mean.append(sum(bob_times_bins_iter_mean) / num_iterations)
            bob_times_bins_freq_iter.append(bob_times_bins_iter)
            # Save the results in separate files for each frequency
            path = f"results/bob_fails_freq_{qubit_delay}us_iter{num_iterations}.txt"
            path2 = f"results/bob_times_freq_{qubit_delay}us_iter{num_iterations}.txt"
            with open(path, "w") as file:
                file.write("Frequency\tFailed Percentages\tMean Failed Percentage\n")
                file.write(f"{qubit_delay}\t{bob_failed_percentages_iter}\t{bob_failed_percentages_freq_mean[-1]}\n")
            with open(path2, "w") as file:
                file.write("Frequency\tTime Bins\tMean Time Bins\tSigma Time Bins\n")
                file.write(f"{qubit_delay}\t{bob_times_bins_iter}\t{bob_times_bins_iter_mean}\t{bob_times_bins_iter_sigma}\n")
                # file.write("Frequency\tTime Bins Mean\tMean Time Bins Sigma\n")
                # file.write(f"{qubit_delay}\t{bob_times_bins_iter_mean}\t{bob_times_bins_iter_sigma}\n")

if do_graph and role == "bob":
    print("Making graphs for the results")
    # Read files if not run the test
    if not run_test:
        bob_failed_percentages_freq_mean = []
        bob_failed_percentages_freq_iter = []
        bob_times_bins_freq_iter = []
        bob_times_bins_freq_mean = []
        bob_times_bins_freq_sigma = []
        # qubit_freq_us = []
        def convert_to_float(x):
            # Handle np.float64 string format
            if 'np.float64' in x:
                return float(x.split('(')[1].split(')')[0])
            return float(x)
        for qubit_delay in qubit_freq_us:
            path = f"results/bob_fails_freq_{qubit_delay}us_iter{num_iterations}.txt"
            print("Reading results from file: %s" % path)
            with open(path, "r") as file:
                lines = file.readlines()
                bob_failed_percentages_freq_iter.append([float(x) for x in lines[1].split("\t")[1].strip("[]").split(", ")])
                bob_failed_percentages_freq_mean.append(float(lines[1].split("\t")[2]))
            path2 = f"results/bob_times_freq_{qubit_delay}us_iter{num_iterations}.txt"
            print("Reading results from file: %s" % path2)
            with open(path2, "r") as file:
                lines = file.readlines()
                from ast import literal_eval
                # bob_times_bins_freq_iter.append([float(x) for x in lines[1].split("\t")[1].strip("[]").split(", ")])

                inner_lists = literal_eval(lines[1].split("\t")[1])
                bob_times_bins_freq_iter.append([float(x) for sublist in inner_lists for x in sublist])
                # bob_times_bins_freq_iter.append([float(x) for x in lines[3].split("\t")[1].strip("[]").split(", ")])
                bob_times_bins_freq_mean.append([convert_to_float(x) for x in lines[1].split("\t")[2].strip("[]").split(", ")])
                bob_times_bins_freq_sigma.append([convert_to_float(x) for x in lines[1].split("\t")[3].strip("[]").split(", ")])
                # bob_times_bins_freq_mean.append(float(x) for x in lines[1].split("\t")[2])
                # bob_times_bins_freq_sigma.append(float(x) for x in lines[1].split("\t")[3])
        

    # Plot packages
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.widgets as widgets

    # Plot the failed percentage
    
    fig, ax = plt.subplots(figsize=(10, 10/GOLDEN_RATIO))
    ax.set_title("Bob Failed Percentage vs Qubit Frequency")
    ax.set_xlabel("Qubit Frequency (us)")
    ax.set_ylabel("Failed Percentage")

    # Color for the diferent qubit frequencies
    colors = ['red', 'blue', 'green', 'orange', 'plum', 'cyan', 'silver']
    colors_darker = ['darkred', 'darkblue', 'darkgreen', 'darkorange', 'purple', 'teal', 'dimgray']

    colors = ['red', 'blue', 'green', 'orange', 'plum', 'cyan', 'silver', 'crimson', 'cornflowerblue', 'yellow']
    colors_darker = ['darkred', 'darkblue', 'darkgreen', 'darkorange', 'purple', 'teal', 'dimgray', 'maroon', 'midnightblue', 'olive']


    for i, freq in enumerate(qubit_freq_us):
        # Extract data for the current qubit frequency
        data_points = bob_failed_percentages_freq_iter[i]
        
        # Plot individual points in the background
        ax.scatter(
            np.full(len(data_points), freq),
            data_points,
            color=colors[i],
            alpha=0.5,
            s=20,
            edgecolor='none',
            label=f"Raw Points (Freq {freq})"
        )
        
        # Add a boxplot for the current frequency
        ax.boxplot(
            data_points,
            positions=[freq],
            widths=25,
            patch_artist=True,
            meanline=True,
            boxprops=dict(facecolor=colors[i], alpha=0.4),
            medianprops=dict(color="black"),
            showfliers=False,
            # remove lines until extreme values
            whiskerprops=dict(color="black", linewidth=1.5),
        )
        
        # Add a standard deviation line
        ax.errorbar(
            freq,
            np.mean(data_points),
            yerr=np.std(data_points),
            fmt='none',
            elinewidth=2,
            alpha=0.35,
            capsize=12,
            ecolor=colors_darker[i],
        )

        # Plot the mean failed percentage
        ax.plot(qubit_freq_us[i], bob_failed_percentages_freq_mean[i], 'o', markersize=10,color=colors_darker[i], label="Mean Failed %% - freq %d us" % qubit_freq_us[i])

    # Add legend
    ax.legend()
    # Limits for the x-axis to make the boxplot more visible
    ax.set_xlim(min(qubit_freq_us) - 100, max(qubit_freq_us) + 100)
    # make legend smaller
    plt.legend(prop={'size': 8})

    # # Plot the time bins
    # fig2, ax2 = plt.subplots(figsize=(10, 10/ GOLDEN_RATIO))

    # for i, freq in enumerate(qubit_freq_us):
    #     # Data
    #     # data_points = bob_times_bins_freq_iter[i]
    #     data_points_mean = bob_times_bins_freq_mean[i]
    #     data_points_sigma = bob_times_bins_freq_sigma[i]

    #     # scatter
    #     ax2.scatter(
    #         np.full(num_iterations, freq),
    #         data_points_mean,
    #         color=colors[i],
    #         alpha=0.5,
    #         edgecolor='none',
    #         s=15,
    #         label=f"Mean Time Bins for freq: {freq}"
    #     )

    #     # boxplot
    #     ax2.boxplot(
    #         data_points_mean,
    #         positions=[freq],
    #         widths=freq/7,
    #         patch_artist=True,
    #         meanline=True,
    #         boxprops=dict(facecolor=colors_darker[i], color=colors[i], alpha=0.5),
    #         whiskerprops=dict(color='black'),
    #         flierprops=dict(color=colors[i], markeredgecolor=colors[i]),
    #         medianprops=dict(color='black'),
    #         showfliers=False
    #     )

    #     # Plot mean
    #     ax2.plot(
    #         freq,
    #         np.mean(data_points_mean),
    #         marker='o',
    #         color=colors_darker[i],
    #         markersize=10,
    #         label=f"Mean of Mean Time Bins for freq: {freq}"
    #     )

    #     # # Add a standard deviation line
    #     # ax2.errorbar(
    #     #     freq,
    #     #     np.mean(data_points),
    #     #     yerr=np.std(data_points),
    #     #     fmt='none',
    #     #     elinewidth=10,
    #     #     alpha=0.35,
    #     #     capsize=10,
    #     #     ecolor=colors[i],
    #     # )

    # ax2.set_title("Time Bins for Bob")
    # ax2.set_xlabel("Frequency")
    # ax2.set_ylabel("Time Bins")
    # ax2.legend()
    # ax2.set_xlim(min(qubit_freq_us) - 20, max(qubit_freq_us) + 10000)
    # ax2.set_xscale('log')
    # ax2.xaxis.set_major_formatter(plt.ScalarFormatter())


    # # Plot the time bins histogram
    # fig3, ax3 = plt.subplots(figsize=(10, 10/ GOLDEN_RATIO))

    # ax3_slider = plt.axes([0.10, 0.02, 0.8, 0.03])
    # slider = widgets.Slider(
    #     ax=ax3_slider,
    #     label='Frequency")',
    #     valmin=0,
    #     valmax=len(qubit_freq_us) - 1,
    #     valinit=0,
    #     valstep=1,
    # )

    # def update(val):
    #     idx = int(slider.val)
    #     ax3.clear()
    #     # Flatten the list of lists into a single array
    #     flattened_times = np.concatenate(bob_times_bins_freq_iter[idx])
    #     nbins = 50
    #     hmin = np.mean(flattened_times) - 3 * np.std(flattened_times)
    #     hmax = np.mean(flattened_times) + 3 * np.std(flattened_times)
    #     # mean = np.mean(bob_times_bins_freq_iter[freq_idx])
    #     # sigma = mean/2
    #     # hmin = mean - 1.75*sigma
    #     # hmax = mean + 6*sigma
    #     ax3.hist(flattened_times, bins=np.linspace(hmin,hmax,nbins+1), color=colors[idx], alpha=0.5, edgecolor='black', label=f"Time -TB for freq: {qubit_freq_us[idx]}")
    #     ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e3:.0f}'))
    #     ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
    #     ax3.set_title(f"Time Bins Histogram for Bob - Qubit Frequency: {qubit_freq_us[idx]}")
    #     ax3.set_xlabel("Qunit Frequency (us)")
    #     ax3.set_ylabel("Times Bins")
    #     # ax3.set_yscale('log')
    #     ax3.axvline(x=qubit_freq_us*1000/2, color='red', linestyle='--', label='Qubit Frequency')
    #     ax3.axvline(x=-qubit_freq_us*1000/2, color='red', linestyle='--')
    #     failure_text = f"Failed Percentage: {bob_failed_percentages_freq_mean[idx]:.2f}"
    #     ax3.text(0.02, 0.95, failure_text, transform=ax3.transAxes, fontsize=12,
    #             verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.5))
    #     ax3.legend()
    #     plt.draw()

    # update(0)
    # slider.on_changed(update)
        


    # Show plots
    plt.tight_layout()
    plt.show()
    
