import numpy as np
import random
import time
import socket
import pickle
from typing import List, Tuple

# Logger
import logging
from logging_setup import setup_logger
# Setup logger
logger = setup_logger("BB84 Protocol Simulation Logger", logging.INFO)


class SynchronizationChannel:
    """
    Synchronization channel to establish time bins for clock alignment using single-byte markers.
    """
    def __init__(self, send_rate, sync_bytes_per_frame):
        self.send_rate = send_rate  # Time interval between photon transmissions (in seconds)
        self.sync_bytes_per_frame = sync_bytes_per_frame
        self.start_marker = 100  # Start of synchronization
        self.end_marker = 50    # End of synchronization

    def send_frames(self, connection: socket.socket, num_frames: int):
        """
        Send synchronization frames with single-byte markers.
        """
        try:
            for frame in range(num_frames):
                print(f"Sending synchronization frame {frame + 1}/{num_frames}")

                # Send start marker
                connection.sendall(self.start_marker.to_bytes(1, 'big'))
                time.sleep(self.send_rate)

                # Simulate sending sync bytes
                for _ in range(self.sync_bytes_per_frame):
                    connection.sendall(random.randint(1, 255).to_bytes(1, 'big'))
                    time.sleep(self.send_rate)

                # Send end marker
                connection.sendall(self.end_marker.to_bytes(1, 'big'))
        except Exception as e:
            print(f"Error in sending frames: {e}")

    def receive_frames(self, connection: socket.socket, num_frames: int):
        """
        Receive synchronization frames with single-byte markers.
        """
        try:
            time_bins = []
            for frame in range(num_frames):
                print(f"Receiving synchronization frame {frame + 1}/{num_frames}")
                start_time = None
                end_time = None
                times_elapsed = []

                while True:
                    data = connection.recv(1)
                    if not data:
                        break

                    byte = int.from_bytes(data, 'big')

                    if byte == self.start_marker:
                        start_time = time.time_ns()
                        times_elapsed = []
                    elif byte == self.end_marker:
                        end_time = time.time_ns()
                        break
                    else:
                        if start_time is not None:
                            elapsed_time = time.time_ns() - start_time
                            times_elapsed.append(elapsed_time)

                if start_time and end_time and len(times_elapsed) > 1:
                    time_from_last = [times_elapsed[i] - times_elapsed[i - 1] for i in range(1, len(times_elapsed))]
                    time_bin = np.mean(time_from_last)
                    time_bins.append(time_bin)
                    print(f"Calculated time bin for frame {frame + 1}: {time_bin:.6f} seconds")
                else:
                    print(f"Not enough data to calculate time bin for frame {frame + 1}.")

            if time_bins:
                average_time_bin = np.mean(time_bins)
                print(f"Average time bin for all frames: {average_time_bin:.6f} seconds")
                return average_time_bin
            else:
                print("No time bins calculated.")
                return None
        except Exception as e:
            print(f"Error in receiving frames: {e}")
