import sys
import os
# Add parent directory to path if the module is there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
import logging
from ..logging_setup import setup_logger
# Setup logger
# logger = setup_logger("BB84 Simulation Log", logging.DEBUG)
logger = setup_logger("BB84 Simulation Log", logging.INFO)



from ..classical_communication_channel.communication_channel.connection_info import ConnectionInfo
from ..classical_communication_channel.communication_channel.mac_config import MAC_Config, MAC_Algorithms 
from ..classical_communication_channel.communication_channel.role import Role

from .bb84_protocol.alice_side_thread_ccc import AliceQubits, AliceThread
# from bb84_protocol.alice_side_thread_ccc import read

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
COM_PORT_DEFAULT = None
COM_PORT_DEFAULT = "COM5"  # Default COM port for the step motor

### INPUTS ###
KEY_LENGTH_DEFAULT = 5
BYTES_PER_FRAME_DEFAULT = 5
SYNC_FRAMES_DEFAULT = 10
SYNC_BYTES_DEFAULT = 10

LOSS_RATE_DEFAULT = 0.0
# KEY_LENGTH_DEFAULT = LOSS_RATE_DEFAULT * 1000
QUBIT_FREQ_US_DEFAULT = 1_000_00
# QUBIT_FREQ_US_DEFAULT = 1 / KEY_LENGTH_DEFAULT * 10**6
TEST_FRACTION = 0.1
NUM_THREADS_DEFAULT = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alice's side of the BB84 protocol")
    parser.add_argument("-k", "--key_length", type=int, default=KEY_LENGTH_DEFAULT, help="Length of the key to be generated")
    parser.add_argument("-bpf", "--bytes_per_frame", type=int, default=BYTES_PER_FRAME_DEFAULT, help="Number of bytes per frame")
    parser.add_argument("-sf", "--sync_frames", type=int, default=SYNC_FRAMES_DEFAULT, help="Number of sync frames")
    parser.add_argument("-sb", "--sync_bytes", type=int, default=SYNC_BYTES_DEFAULT, help="Number of sync bytes per frame")
    parser.add_argument("-lr", "--loss_rate", type=float, default=LOSS_RATE_DEFAULT, help="Loss rate for the qubits")
    parser.add_argument("-qf", "--qubit_freq_us", type=float, default=QUBIT_FREQ_US_DEFAULT, help="Qubit frequency in microseconds")
    parser.add_argument("-tf", "--test_fraction", type=float, default=TEST_FRACTION, help="Fraction of the key to be used for testing")
    parser.add_argument("-nth", "--num_threads", type=int, default=NUM_THREADS_DEFAULT, help="Number of threads to be used")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-nv", "--no-verbose", action="store_false", help="Decrease output verbosity")
    parser.add_argument("-smc", "--step_motor_comport", type=str, default=COM_PORT_DEFAULT, help="Com Port for the step motor, e.g. COM5")
    args = parser.parse_args()

    key_length = args.key_length
    bytes_per_frame = args.bytes_per_frame
    num_frames = key_length // bytes_per_frame

    loss_rate = args.loss_rate
    qubit_freq_us = args.qubit_freq_us

    sync_interval = 0.1
    sync_frames = args.sync_frames
    sync_bytes = args.sync_bytes

    do_test = True
    test_fraction = args.test_fraction
    error_threshold = 0.1
    step_motor_comport = args.step_motor_comport

    if step_motor_comport:
        from ..interface_stepperMotor.imports.stm32_interface import *
        stm = STM32Interface(step_motor_comport)
        stm.start()
        stm.connect()
    else:
        stm = None

    verbose = args.verbose
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    no_verbose = args.no_verbose
    if no_verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)       
    

    # Number of threads to be used
    nb_threads = args.num_threads
    # id for each thread
    thread_ids = [i for i in range(nb_threads)]
    # Divide the key into parts for each thread
    key_length_parts = [key_length // nb_threads for _ in range(nb_threads)]
    num_frames_parts = [key_length_parts[i] // bytes_per_frame for i in range(nb_threads)]

    
    # Setup the connection for the classical communication channel
    bandwidth_limit_MBps = 100  # Note: in MB/s; multiply by 8 to get Mbps
    mac_configuration = MAC_Config(MAC_Algorithms.CMAC, SHARED_SECRET_KEY)
    alice_info = ConnectionInfo(IP_ADDRESS, PORT_NUMBER_ALICE)
    role_alice = Role.get_instance(alice_info, mac_config=mac_configuration)
    bob_info = ConnectionInfo(IP_ADDRESS, PORT_NUMBER_BOB)
    print(f"CCC: Role Alice - {alice_info.ip} : {alice_info.port} - is connected to the Role Bob - {bob_info.ip} : {bob_info.port}")
    # role_alice.peer_connection_info = bob_info

    try:
        # Set up a simple socket connection for the qubits exchange 
        server = AliceQubits.setup_server(IP_ADDRESS, PORT_NUMBER_QUANTIC_CHANNEL)
        connection, address = server.accept()
        with connection:
            alice_queues = []
            alice_qubits = []
            alice_cccs = []

            # Create the instances of AliceQubits and AliceThread for each thread
            for i in range(nb_threads):

                alice_queue = queue.Queue()
                alice_queues.append(alice_queue)

            # Start the reader thread
            reader_thread = threading.Thread(target=read, args=[role_alice, alice_queues], daemon=True)
            reader_thread.start()

            # Function to run both part of the protocol - CHANGE THIS TO ONLY ONCE QUBIT IS RECEIVED START NEXT THREAD
            def run_all_bs_thread( start_event: threading.Event, thread: int):
                print(f"Alice Thread {thread} - thread started")
                alice_qubit = AliceQubits(num_qubits=key_length_parts[thread],
                                    qubit_delay_us=qubit_freq_us,
                                    num_frames=num_frames_parts[thread],
                                    bytes_per_frame=bytes_per_frame,
                                    sync_frames=sync_frames,
                                    sync_bytes_per_frame=sync_bytes,
                                    loss_rate=loss_rate,
                                    com_interface=stm,
                    )
                if not stm:
                    # If no step motor is used, use the mock qubit
                    alice_qubit.run_mock_alice_qubits(connection, wait_time=True)
                else:
                    # If a step motor is used, use the stepmotor interface
                    alice_qubit.run_alice_qubits_motor(connection, wait_time=True)

                alice_qubits.append(alice_qubit)
                print(f"Alice Thread {thread} - qubits sent")
                
                # alice_ccc = AliceThread(
                #         test_bool=do_test,
                #         test_fraction=test_fraction,
                #         error_threshold=error_threshold,
                #         thread_id=thread,
                #         start_idx=0,
                #         role=role_alice,
                #         receive_queue=alice_queues[thread]
                # )
                # alice_ccc.set_bits_bases_qubits(alice_qubit.bits, alice_qubit.bases, alice_qubit.qubits_bytes)
                # start_event.set() # Set the event for the next thread after sending the qubits
                # alice_ccc.run_alice_thread()
                # alice_cccs.append(alice_ccc)
                # print(f"Alice Thread {thread} - thread finished")

                # alice_key = alice_ccc.final_key

            alice_threads = []
            final_results = []
            final_failed_percentage = []
            start_event = threading.Event()
            start_event.set()  # Set the event for the first thread
            # Start the threads
            for i in range(nb_threads):
                alice_thread = threading.Thread(target=run_all_bs_thread, args=[start_event, i])
                alice_threads.append(alice_thread)
                alice_thread.start()
                if i < nb_threads - 1:  # Don't wait after starting the last thread
                    start_event.clear()  # Clear the event for the next thread
                    start_event.wait()  # Wait for the thread to process and set the event


            for thread in alice_threads:
                thread.join()

            print("All Alice threads have finished.")

            # for i in range(nb_threads):
            #     final_results.append(alice_cccs[i].final_key)
            #     final_failed_percentage.append(alice_cccs[i].failed_percentage)

            # # print("Final results: ", final_results)
            # print("Final failed percentage: ", final_failed_percentage)
            # if nb_threads == 1:
            #     print("Size of payload")
            #     print(f"Alice detected bases size: {alice_cccs[0].alice_detected_bases_size} bytes")
            #     print(f"Alice test result size: {alice_cccs[0].alice_test_result_size} bytes")
            
    except KeyboardInterrupt:
        print("Stopped by user.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Closing the initial server...")
        connection.close()
        server.close()
        if stm:
            print("Closing the step motor...")
            stm.stop()


    # Close role
    role_alice.clean()