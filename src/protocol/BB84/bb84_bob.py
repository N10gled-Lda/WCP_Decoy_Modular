import sys
import os
# Add parent directory to path if the module is there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Configure logging
import logging
from logging_setup import setup_logger
# Setup logger
# logger = setup_logger("BB84 Simulation Log", logging.DEBUG)
logger = setup_logger("BB84 Simulation Log", logging.INFO)

from classical_communication_channel.communication_channel.connection_info import ConnectionInfo
from classical_communication_channel.communication_channel.mac_config import MAC_Config, MAC_Algorithms 
from classical_communication_channel.communication_channel.role import Role

from BB84.bb84_protocol.bob_side_thread_ccc import BobQubits, BobThread
# from bb84_protocol.bob_side_thread_ccc import read

import queue
import threading
import time
import argparse
import pickle

def read(role: Role, queues: list[queue.Queue]):
    try:
        while True:
            payload = role.get_from_inbox()
            (thread_id, data) = pickle.loads(payload)
            queues[thread_id].put(data)
    except KeyboardInterrupt:
        role.clean()

### IP, PORTs, CONNECTIONs ###
IP_ADDRESS = "127.0.0.1"
PORT_NUMBER_ALICE = 65432
PORT_NUMBER_BOB = 65433
PORT_NUMBER_QUANTIC_CHANNEL = 12345
SHARED_SECRET_KEY = b'IzetXlgAnY4oye56'

### INPUTS ###
KEY_LENGTH_DEFAULT = 3500
BYTES_PER_FRAME_DEFAULT = 10
SYNC_FRAMES_DEFAULT = 10
SYNC_BYTES_DEFAULT = 10
TEST_FRACTION = 0.1

NUM_THREADS_DEFAULT = 1

if __name__ == "__main__":

    # Configure the arguments for the script
    parser = argparse.ArgumentParser(description="Bob's side of the BB84 protocol")
    parser.add_argument("-k", "--key_length", type=int, default=KEY_LENGTH_DEFAULT, help="Length of the key to be generated")
    parser.add_argument("-bpf", "--bytes_per_frame", type=int, default=BYTES_PER_FRAME_DEFAULT, help="Number of bytes per frame")
    parser.add_argument("-sf", "--sync_frames", type=int, default=SYNC_FRAMES_DEFAULT, help="Number of sync frames")
    parser.add_argument("-sb", "--sync_bytes", type=int, default=SYNC_BYTES_DEFAULT, help="Number of sync bytes per frame")
    parser.add_argument("-tf", "--test_fraction", type=float, default=TEST_FRACTION, help="Fraction of the key to be used for testing")
    parser.add_argument("-nth", "--num_threads", type=int, default=NUM_THREADS_DEFAULT, help="Number of threads to be used")
    args = parser.parse_args()

    # Set the default values for the arguments
    key_length = args.key_length
    bytes_per_frame = args.bytes_per_frame
    num_frames = key_length // bytes_per_frame

    sync_frames = args.sync_frames
    sync_bytes = args.sync_bytes

    do_test = True
    test_fraction = args.test_fraction

    # Number of threads
    nb_threads = args.num_threads
    # id for each thread
    thread_ids = [i for i in range(nb_threads)]
    # Divide the key into parts for each thread
    key_parts = [key_length // nb_threads for _ in range(nb_threads)]
    # for i in range(key_length % nb_threads):
    #     key_parts[i] += 1
    num_frames_parts = [key_parts[i] // bytes_per_frame for i in range(nb_threads)]

    # Setup the connection for the classical communication channel
    mac_configuration = MAC_Config(MAC_Algorithms.CMAC, SHARED_SECRET_KEY)
    bob_info = ConnectionInfo(IP_ADDRESS, PORT_NUMBER_BOB)
    role_bob = Role.get_instance(bob_info)
    alice_info = ConnectionInfo(IP_ADDRESS, PORT_NUMBER_ALICE)
    print(f"CCC: Role Bob - {bob_info.ip} : {bob_info.port} - is connected to the Role Bob - {alice_info.ip} : {alice_info.port}")

    role_bob.peer_connection_info = alice_info

    # Set up a simple socket connection for the qubits exchange temporarily
    while True:
        try:
            connection = BobQubits.setup_client(IP_ADDRESS, PORT_NUMBER_QUANTIC_CHANNEL)
            print(f"Connection established at {IP_ADDRESS}:{PORT_NUMBER_QUANTIC_CHANNEL}")

            bob_queues = []
            bob_qubits = []
            bob_cccs = []
            # Create the instances of BobQubits and BobThread for each thread
            for i in range(nb_threads):  
                # bob_qubit = BobQubits(
                #     num_qubits=key_parts[i],
                #     num_frames=num_frames_parts[i],
                #     bytes_per_frame=bytes_per_frame,
                #     sync_frames=sync_frames,
                #     sync_bytes_per_frame=sync_bytes,
                # )
                # bob_qubits.append(bob_qubit)

                bob_queue = queue.Queue()
                # bob_ccc = BobThread(
                #     test_bool=do_test,
                #     test_fraction=test_fraction,
                #     thread_id=i,
                #     role=role_bob,
                #     receive_queue=bob_queue
                # )
                bob_queues.append(bob_queue)
                # bob_cccs.append(bob_ccc)

            # Start the reader thread
            reader_thread = threading.Thread(target=read, args=[role_bob, bob_queues], daemon=True)
            reader_thread.start()

            def run_all_bs_thread(start_event: threading.Event, thread_id: int):
                # start_event.wait() # Wait for the event to be set
                bob_qubit = BobQubits(
                    num_qubits=key_parts[thread_id],
                    num_frames=num_frames_parts[thread_id],
                    bytes_per_frame=bytes_per_frame,
                    sync_frames=sync_frames,
                    sync_bytes_per_frame=sync_bytes,
                )
                bob_qubit.run_bob_qubits(connection)
                bob_qubits.append(bob_qubit)
                
                print(f"Bob thread {thread_id} - Qubits received.")
                bob_ccc = BobThread(
                    test_bool=do_test,
                    test_fraction=test_fraction,
                    thread_id=thread_id,
                    role=role_bob,
                    receive_queue=bob_queues[thread_id]
                )
                bob_ccc.set_bits_bases_qubits_idxs(bob_qubit.detected_bits, bob_qubit.detected_bases, bob_qubit.detected_qubits_bytes, bob_qubit.detected_idxs, bob_qubit.average_time_bin)
                start_event.set() # Set the event for the next thread after receiving the qubits
                bob_ccc.run_bob_thread()
                bob_cccs.append(bob_ccc)
                print(f"Bob thread {thread_id} -  finished.")

            bob_threads = []
            final_results = []
            final_failed_percentage = []
            start_event = threading.Event()
            start_event.set() # Set the event for the first thread
            # Start the threads for Bob
            for i in range(nb_threads):
                bob_thread = threading.Thread(target=run_all_bs_thread, args=(start_event, i))
                bob_threads.append(bob_thread)
                bob_thread.start()
                if i < nb_threads - 1:  # Don't wait after starting the last thread
                    start_event.clear()  # Clear the event for the next thread
                    start_event.wait()  # Wait for the thread to process and set the event

            for bob_thread in bob_threads:
                bob_thread.join()

            print("All Bob threads have finished.")

            for i in range(nb_threads):
                final_results.append(bob_cccs[i].final_key)
                final_failed_percentage.append(bob_cccs[i].failed_percentage)

            # print("Final results: ", final_results)
            print("Final failed percentage: ", final_failed_percentage)

            if nb_threads == 1:
                print(f"Bob detected indices size: {bob_cccs[0].bob_detected_idx_size} bytes")
                print(f"Bob common indices size: {bob_cccs[0].bob_common_idx_size} bytes")
                print(f"Bob test bits size: {bob_cccs[0].bob_test_bits_size} bytes")

            break


        except KeyboardInterrupt:
            print("Stopped by user.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)
        finally:
            print("Closing the initial server...")
            connection.close()
            
    # Close role
    role_bob.clean()