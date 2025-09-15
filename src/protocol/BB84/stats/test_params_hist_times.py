# Description: Make graph/histogram to test the parameters and observe both the time taken and the number of qubits sent and how similar they are to the expected time bin, and to see the failure rate of the test.
import argparse
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import time
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from bb84_protocol.old_version.alice_participant import Alice
from bb84_protocol.old_version.bob_participant import Bob

# Parameters Constant
host = 'localhost'
port = 12345

loss_rate = 1 - 0.0
do_test = True
test_fraction = 0.9999
error_threshold = 0.1

# Parameters to test default values
key_length = 1000
qubit_freq_us = 500
bytes_per_frame = 20
num_frames = key_length // bytes_per_frame
sync_frames = 100
sync_bytes = 50


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Run BB84 Protocol Simulation Tests')
    argparser.add_argument('-role', '-r', type=str, required=False, default='Bob',
                            help='Role of the participant (Alice or Bob)')
    argparser.add_argument('-run_bool', '-rb', type=bool, default=False,
                            help='Whether to run the simulation or to use file data')
    argparser.add_argument('-test_number', '-tn', type=int, default=1,
                            help='Identifier for the test')
    argparser.add_argument('-file_name', '-fn', type=str, default='bob_times_bins_key.txt',
                            help='Name of the file to store the results')
    
    args = argparser.parse_args()
    role = args.role.lower()
    run_bool = args.run_bool
    test_number = args.test_number
    # Parameters to test
    if test_number == 1:
        # List from 100 to 1000 with step 100 then from 1000 to 10000 with step 500
        key_length = list(range(100, 1100, 100)) + list(range(1500, 10500, 500))
        num_frames  = [key // bytes_per_frame for key in key_length]
        label = 'Key Lengths'
        param_var = key_length
        file = 'bob_times_bins_key.txt'
        loc = './results/'
    
    if test_number == 2:
        # List from 100 to 1000 with step 100 then from 1000 to 10000 with step 500 
        qubit_freq_us = list(range(100, 1000, 100)) + list(range(1000, 11000, 1000))
        # qubit_freq_us = list(range(100, 1000, 100)) + list(range(1000, 2000, 500))
        label = 'Qubit Frequencies'
        param_var = qubit_freq_us
        file = 'bob_times_bins_qubit_freq.txt'
        loc = './results/'
    
    if test_number == 3:
        # List from 10 to 100 with step 10 then from 100 to 500 with step 50
        key_length = 10000
        bytes_per_frame = list(range(10, 110, 10)) + list(range(150, 550, 50))
        num_frames = [key_length // byte for byte in bytes_per_frame]
        label = 'Bytes Per Frame'
        param_var = bytes_per_frame
        file = 'bob_times_bins_bytes_per_frame.txt'
        loc = './results/'

    # bytes_per_frame = [10, 20, 50]
    # sync_frames = [100, 1000, 10000]
    # sync_bytes = [10, 50, 100]

    # Variables to store the results
    bob_times_bins_key = []
    bob_failed_percentage = []
    # Test different values of key_length
    if run_bool:
        if test_number == 1:
            for key, num_frame in zip(key_length, num_frames):
                if role == 'alice':
                    alice = Alice(num_qubits=key,
                        qubit_delay_us=qubit_freq_us,
                        num_frames=num_frame,
                        bytes_per_frame=bytes_per_frame,
                        sync_frames=sync_frames,
                        sync_bytes_per_frame=sync_bytes,
                        loss_rate=loss_rate,
                        test_bool=do_test,
                        test_fraction=test_fraction,
                        error_threshold=error_threshold
                        )
                    alice.run_alice(host, port)

                if role == 'bob':
                    bob_times_bins = []
                    bob = Bob(num_qubits=key,
                        num_frames=num_frame,
                        bytes_per_frame=bytes_per_frame,
                        sync_frames=sync_frames,
                        sync_bytes_per_frame=sync_bytes,
                        test_bool=do_test,
                        test_fraction=test_fraction,
                        )
                    bob.run_bob(host, port)
                    bob_failed_percentage.append(bob.failed_percentage)
                    times = bob.detected_timestamps
                    # Calculate the time bins
                    for frame in range(num_frame):
                        for i in range(len(times) // num_frame - 1):
                            time_bins_val = times[i + 1 + frame * bytes_per_frame] - times[i + frame * bytes_per_frame]
                            bob_times_bins.append(time_bins_val)
                    bob_times_bins_key.append(bob_times_bins)

                time.sleep(1)

            with open(loc + file, 'w') as f:
                f.write("Key Lengths: %s\n" % key_length)
                f.write("Failed Percentages: %s\n" % bob_failed_percentage)
                for item in bob_times_bins_key:
                    f.write("%s\n" % item)

        if test_number == 2:
            for qubit in qubit_freq_us:
                if role == 'alice':
                    alice = Alice(num_qubits=key_length,
                        qubit_delay_us=qubit,
                        num_frames=num_frames,
                        bytes_per_frame=bytes_per_frame,
                        sync_frames=sync_frames,
                        sync_bytes_per_frame=sync_bytes,
                        loss_rate=loss_rate,
                        test_bool=do_test,
                        test_fraction=test_fraction,
                        error_threshold=error_threshold
                        )
                    alice.run_alice(host, port)

                if role == 'bob':
                    bob_times_bins = []
                    bob = Bob(num_qubits=key_length,
                        qubit_delay_us=qubit,
                        num_frames=num_frames,
                        bytes_per_frame=bytes_per_frame,
                        sync_frames=sync_frames,
                        sync_bytes_per_frame=sync_bytes,
                        test_bool=do_test,
                        test_fraction=test_fraction,
                        )
                    bob.run_bob(host, port)
                    bob_failed_percentage.append(bob.failed_percentage)
                    times = bob.detected_timestamps
                    # Calculate the time bins
                    for frame in range(num_frames):
                        for i in range(len(times) // num_frames - 1):
                            time_bins_val = times[i + 1 + frame * bytes_per_frame] - times[i + frame * bytes_per_frame]
                            bob_times_bins.append(time_bins_val)
                    bob_times_bins_key.append(bob_times_bins)

                time.sleep(1)

            with open(loc + file, 'w') as f:
                f.write("Qubit Frequencies: %s\n" % qubit_freq_us)
                f.write("Failed Percentages: %s\n" % bob_failed_percentage)
                for item in bob_times_bins_key:
                    f.write("%s\n" % item)
        
        if test_number == 3:
            for byte, num_frame in zip(bytes_per_frame, num_frames):
                if role == 'alice':
                    alice = Alice(num_qubits=key_length,
                        qubit_delay_us=qubit_freq_us,
                        num_frames=num_frame,
                        bytes_per_frame=byte,
                        sync_frames=sync_frames,
                        sync_bytes_per_frame=sync_bytes,
                        loss_rate=loss_rate,
                        test_bool=do_test,
                        test_fraction=test_fraction,
                        error_threshold=error_threshold
                        )
                    alice.run_alice(host, port)

                if role == 'bob':
                    bob_times_bins = []
                    bob = Bob(num_qubits=key_length,
                        num_frames=num_frame,
                        bytes_per_frame=byte,
                        sync_frames=sync_frames,
                        sync_bytes_per_frame=sync_bytes,
                        test_bool=do_test,
                        test_fraction=test_fraction,
                        )
                    bob.run_bob(host, port)
                    bob_failed_percentage.append(bob.failed_percentage)
                    times = bob.detected_timestamps
                    # Calculate the time bins
                    print(len(times))
                    print(num_frame)
                    print(byte)
                    for frame in range(num_frame):
                        for i in range(len(times) // num_frame - 1):
                            time_bins_val = times[i + 1 + frame * byte] - times[i + frame * byte]
                            bob_times_bins.append(time_bins_val)
                    bob_times_bins_key.append(bob_times_bins)

                time.sleep(1)

            path = './results/' + file
            with open(path, 'w') as f:
                f.write("Bytes Per Frame: %s\n" % bytes_per_frame)
                f.write("Failed Percentages: %s\n" % bob_failed_percentage)
                for item in bob_times_bins_key:
                    f.write("%s\n" % item)

    if role == 'bob':
        if not run_bool:
            with open(loc + file, 'r') as f:
                lines = f.readlines()
                param_var = eval(lines[0].split(': ')[1])
                bob_failed_percentage = eval(lines[1].split(': ')[1])
                bob_times_bins_key = [eval(line) for line in lines[2:]]
                
        
        # Print failure results
        print(f"Bob Failed Percentage: {bob_failed_percentage}")

        # Plot the results
        # fig, axs = plt.subplots(1, len(param_var), figsize=(15, 5))
        # fig.suptitle('Time Bins for different key lengths')
        # for i, key in enumerate(param_var):
        #     axs[i].hist(bob_times_bins_key[i], bins=np.linspace(0, 1e6, 51))
        #     axs[i].set_title(f'Key Length: {key}')
        #     axs[i].set_xlabel('Time Bins (us)')
        #     axs[i].set_ylabel('Frequency')
        #     # axs[i].set_xlim(0, 1e6)
        #     axs[i].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}'))
        #     axs[i].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
        # plt.tight_layout()
        # plt.show()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.subplots_adjust(bottom=0.25)  # Make room for slider

        # Add slider
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        slider = Slider(
            ax=ax_slider,
            label=label,
            valmin=0,
            valmax=len(param_var)-1,
            valinit=0,
            valstep=1
        )

        def update(val):
            ax.clear()
            key_idx = int(slider.val)
            time_bins = bob_times_bins_key[key_idx]
            mean = np.mean(time_bins)
            sigma = param_var[key_idx]*1000/2 if test_number == 2 else mean/8
            nbins = 75
            ax.hist(time_bins, bins=np.linspace(mean - 2 * sigma, mean + 2 * sigma, nbins+1))
            if test_number == 1:
                ax.set_title(f'Time Bins - Key Length: {param_var[key_idx]}')
            if test_number == 2:
                ax.set_title(f'Time Bins - Qubit Frequency (us): {param_var[key_idx]}')
            if test_number == 3:
                ax.set_title(f'Time Bins - Bytes Per Frame: {param_var[key_idx]}')
            ax.set_xlabel('Time Bins (us)')
            ax.set_ylabel('Frequency')
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e3:.0f}'))
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}'))
            failure_text = f'Failure Rate: {bob_failed_percentage[key_idx]:.2f}%'
            ax.text(0.95, 0.95, failure_text, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))
            # Add a red line for the mean
            ax.axvline(mean, color='r', linestyle='dashed', linewidth=1)
            fig.canvas.draw_idle()

        # Initial plot with first key length
        update(0)


        slider.on_changed(update)
        plt.show()