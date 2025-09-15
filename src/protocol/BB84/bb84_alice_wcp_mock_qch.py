import sys
import os
import argparse
import threading
import queue
import pickle
import time

# Add parent directory to path if the module is there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
import logging
from examples.logging_setup import setup_logger
logger = setup_logger("WCP BB84 Alice Log", logging.INFO)

from classical_communication_channel.communication_channel.connection_info import ConnectionInfo
from classical_communication_channel.communication_channel.mac_config import MAC_Config, MAC_Algorithms 
from classical_communication_channel.communication_channel.role import Role

from BB84.bb84_protocol.alice_wcp_bb84_ccc import AliceWCPQubits, AliceWCPThread

def read(role: Role, queues: list[queue.Queue]):
    """Read data from classical communication channel"""
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
KEY_LENGTH_DEFAULT = 1000
BYTES_PER_FRAME_DEFAULT = 10
SYNC_FRAMES_DEFAULT = 10
SYNC_BYTES_DEFAULT = 10

# WCP-specific defaults
SIGNAL_INTENSITY_DEFAULT = 0.7
DECOY_INTENSITY_DEFAULT = 0.1
VACUUM_INTENSITY_DEFAULT = 0.0
SIGNAL_PROB_DEFAULT = 0.7
DECOY_PROB_DEFAULT = 0.2
VACUUM_PROB_DEFAULT = 0.1
TRANSMISSION_EFFICIENCY_DEFAULT = 0.1 # Loss of 90%

QUBIT_FREQ_US_DEFAULT = 1000 # Frequency in microseconds
TEST_FRACTION = 0.1
ERROR_THRESHOLD = 0.1 # 10% error threshold for testing
NUM_THREADS_DEFAULT = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Alice's side of the WCP BB84 protocol")
    parser.add_argument("-k", "--key_length", type=int, default=KEY_LENGTH_DEFAULT, 
                       help="Length of the key to be generated")
    parser.add_argument("-bpf", "--bytes_per_frame", type=int, default=BYTES_PER_FRAME_DEFAULT, 
                       help="Number of bytes per frame")
    parser.add_argument("-sf", "--sync_frames", type=int, default=SYNC_FRAMES_DEFAULT, 
                       help="Number of sync frames")
    parser.add_argument("-sb", "--sync_bytes", type=int, default=SYNC_BYTES_DEFAULT, 
                       help="Number of sync bytes per frame")
    
    # WCP-specific arguments
    parser.add_argument("-si", "--signal_intensity", type=float, default=SIGNAL_INTENSITY_DEFAULT,
                       help="Mean photon number for signal pulses")
    parser.add_argument("-di", "--decoy_intensity", type=float, default=DECOY_INTENSITY_DEFAULT,
                       help="Mean photon number for decoy pulses")
    parser.add_argument("-vi", "--vacuum_intensity", type=float, default=VACUUM_INTENSITY_DEFAULT,
                       help="Mean photon number for vacuum pulses")
    parser.add_argument("-sp", "--signal_prob", type=float, default=SIGNAL_PROB_DEFAULT,
                       help="Probability of sending signal pulse")
    parser.add_argument("-dp", "--decoy_prob", type=float, default=DECOY_PROB_DEFAULT,
                       help="Probability of sending decoy pulse")
    parser.add_argument("-vp", "--vacuum_prob", type=float, default=VACUUM_PROB_DEFAULT,
                       help="Probability of sending vacuum pulse")
    parser.add_argument("-te", "--transmission_efficiency", type=float, default=TRANSMISSION_EFFICIENCY_DEFAULT,
                       help="Channel transmission efficiency")
    
    parser.add_argument("-qf", "--qubit_freq_us", type=float, default=QUBIT_FREQ_US_DEFAULT, 
                       help="Qubit frequency in microseconds")
    parser.add_argument("-tf", "--test_fraction", type=float, default=TEST_FRACTION, 
                       help="Fraction of the key to be used for testing")
    parser.add_argument("-et", "--error_threshold", type=float, default=ERROR_THRESHOLD,
                       help="Error threshold for testing (as a fraction of the key length)")
    parser.add_argument("-nth", "--num_threads", type=int, default=NUM_THREADS_DEFAULT, 
                       help="Number of threads to be used")
    parser.add_argument("-v", "--verbose", action="store_true", help="Increase output verbosity")
    parser.add_argument("-nv", "--no-verbose", action="store_false", help="Decrease output verbosity")
    args = parser.parse_args()

    # Parse arguments
    key_length = args.key_length
    bytes_per_frame = args.bytes_per_frame
    num_frames = key_length // bytes_per_frame

    sync_frames = args.sync_frames
    sync_bytes = args.sync_bytes
    
    # WCP parameters
    signal_intensity = args.signal_intensity
    decoy_intensity = args.decoy_intensity
    vacuum_intensity = args.vacuum_intensity
    signal_prob = args.signal_prob
    decoy_prob = args.decoy_prob
    vacuum_prob = args.vacuum_prob
    transmission_efficiency = args.transmission_efficiency
    
    qubit_freq_us = args.qubit_freq_us
    test_fraction = args.test_fraction
    error_threshold = args.error_threshold

    # Logging setup
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

    # Threading setup
    nb_threads = args.num_threads
    thread_ids = [i for i in range(nb_threads)]
    key_length_parts = [key_length // nb_threads for _ in range(nb_threads)]
    if key_length_parts[0] < bytes_per_frame:
        logger.error(f"!!! Key length {key_length} is too short for nb threads {nb_threads} and the bytes per frame {bytes_per_frame}!!!")
        sys.exit(1)
    num_frames_parts = [key_length_parts[i] // bytes_per_frame for i in range(nb_threads)]

    # Validate WCP probabilities
    total_prob = signal_prob + decoy_prob + vacuum_prob
    if abs(total_prob - 1.0) > 0.01:
        logger.warning(f"Pulse probabilities sum to {total_prob}, normalizing...")
        signal_prob /= total_prob
        decoy_prob /= total_prob
        vacuum_prob /= total_prob

    logger.info("=== WCP BB84 Alice Configuration ===")
    logger.info(f"Key length: {key_length}")
    logger.info(f"Signal intensity (μs): {signal_intensity}")
    logger.info(f"Decoy intensity (μd): {decoy_intensity}")
    logger.info(f"Vacuum intensity (μd): {vacuum_intensity}")
    logger.info(f"Signal probability: {signal_prob:.2f}")
    logger.info(f"Decoy probability: {decoy_prob:.2f}")
    logger.info(f"Vacuum probability: {vacuum_prob:.2f}")
    logger.info(f"Transmission efficiency: {transmission_efficiency}")
    logger.info(f"Number of threads: {nb_threads}")
    logger.info(f"Test fraction: {test_fraction}")
    logger.info(f"Qubit frequency: {qubit_freq_us} μs")
    logger.info(f"Sync frames: {sync_frames}, Sync bytes per frame: {sync_bytes}")
    logger.info(f"Bytes per frame: {bytes_per_frame}")
    logger.info(f"Key length parts per thread: {key_length_parts}")
    logger.info(f"Total frames per thread: {num_frames_parts}")
    logger.info("=====================================")
    
    # Setup classical communication channel
    mac_configuration = MAC_Config(MAC_Algorithms.CMAC, SHARED_SECRET_KEY)
    alice_info = ConnectionInfo(IP_ADDRESS, PORT_NUMBER_ALICE)
    role_alice = Role.get_instance(alice_info, mac_config=mac_configuration)
    bob_info = ConnectionInfo(IP_ADDRESS, PORT_NUMBER_BOB)
    
    logger.info(f"CCC: Role Alice - {alice_info.ip}:{alice_info.port} - connected to Bob - {bob_info.ip}:{bob_info.port}")

    try:
        # Set up quantum channel server
        server = AliceWCPQubits.setup_server(IP_ADDRESS, PORT_NUMBER_QUANTIC_CHANNEL)
        connection, address = server.accept()
        
        with connection:
            logger.info(f"Quantum channel connection established with {address}")
            
            # Initialize Alice components
            alice_queues = []
            alice_wcp_qubits = []
            alice_wcp_threads = []

            # Create instances for each thread
            for i in range(nb_threads):
                # Create queue for this thread
                alice_queue = queue.Queue()
                alice_queues.append(alice_queue)
                
                # Create WCP qubits handler
                # alice_wcp = AliceWCPQubits(
                #     num_qubits=key_length_parts[i],
                #     qubit_delay_us=qubit_freq_us,
                #     num_frames=num_frames_parts[i],
                #     bytes_per_frame=bytes_per_frame,
                #     sync_frames=sync_frames,
                #     sync_bytes_per_frame=sync_bytes,
                #     mu_signal=signal_intensity,
                #     mu_decoy=decoy_intensity,
                #     mu_vacuum=vacuum_intensity,
                #     prob_signal=signal_prob,
                #     prob_decoy=decoy_prob,
                #     prob_vacuum=vacuum_prob,
                #     transmission_efficiency=transmission_efficiency
                # )
                # alice_wcp_qubits.append(alice_wcp)
                
                # # Create WCP thread handler
                # alice_wcp_thread = AliceWCPThread(
                #     role=role_alice,
                #     receive_queue=alice_queue,
                #     thread_id=i,
                #     test_bool=True,
                #     test_fraction=test_fraction,
                #     error_threshold=error_threshold,
                #     start_idx=i * key_length_parts[i]
                # )
                # alice_wcp_threads.append(alice_wcp_thread)

            # Start the classical communication reader thread
            reader_thread = threading.Thread(target=read, args=[role_alice, alice_queues], daemon=True)
            reader_thread.start()

            def run_alice_wcp_thread(start_event: threading.Event, thread_id: int):
                """Run Alice WCP protocol for a specific thread"""
                try:
                    logger.info(f"Thread {thread_id}: Starting WCP protocol...")
                    alice_wcp = AliceWCPQubits(
                        num_qubits=key_length_parts[thread_id],
                        qubit_delay_us=qubit_freq_us,
                        num_frames=num_frames_parts[thread_id],
                        bytes_per_frame=bytes_per_frame,
                        sync_frames=sync_frames,
                        sync_bytes_per_frame=sync_bytes,
                        mu_signal=signal_intensity,
                        mu_decoy=decoy_intensity,
                        mu_vacuum=vacuum_intensity,
                        prob_signal=signal_prob,
                        prob_decoy=decoy_prob,
                        prob_vacuum=vacuum_prob,
                        transmission_efficiency=transmission_efficiency
                    )
                    # alice_wcp.run_alice_wcp_protocol(connection)
                    alice_wcp.run_mock_alice_qubits(connection, True)
                    
                    # Get pulse statistics
                    stats = alice_wcp.get_pulse_statistics()
                    logger.info(f"Thread {thread_id}: Pulse statistics: {stats}")

                    alice_wcp_qubits.append(alice_wcp)

                    # Set WCP data for thread
                    alice_wcp_thread = AliceWCPThread(
                        role=role_alice,
                        receive_queue=alice_queue,
                        thread_id=i,
                        test_bool=True,
                        test_fraction=test_fraction,
                        error_threshold=error_threshold,
                        start_idx=i * key_length_parts[i]
                    )
                    alice_wcp_thread.set_wcp_data(
                        bits=alice_wcp.bits,
                        bases=alice_wcp.bases,
                        pulse_types=alice_wcp.pulse_types,
                        pulse_intensities=alice_wcp.pulse_intensities
                    )
                    
                    logger.info(f"Thread {thread_id}: WCP data set for thread")
                    # Run the WCP thread
                    start_event.set()  # Set the event to signal the thread to start

                    # Run classical communication protocol
                    final_key, success = alice_wcp_thread.run_alice_wcp_thread()
                    alice_wcp_threads.append(alice_wcp_thread)
                    print(f"Thread {thread_id}: COMPLETED")

                    # Log final key and success status
                    if success:
                        logger.info(f"Thread {thread_id}: WCP protocol completed successfully")
                        logger.info(f"Thread {thread_id}: Final key length: {len(final_key) if final_key else 0}")
                        return final_key, True
                    else:
                        logger.error(f"Thread {thread_id}: WCP protocol failed")
                        return None, False
                        
                except Exception as e:
                    logger.error(f"Thread {thread_id}: Exception in WCP protocol: {e}")
                    return None, False

            # Run all threads
            alice_threads = []
            final_results = []
            
            start_event = threading.Event()
            start_event.set()  # Set the event for the first thread

            # Start threads (staggered to avoid conflicts)
            for i in range(nb_threads):
                # thread = threading.Thread(target=lambda tid=[start_event, i]: final_results.append(run_alice_wcp_thread(tid)))
                thread = threading.Thread(target=run_alice_wcp_thread, args=[start_event, i])
                alice_threads.append(thread)
                thread.start()
                
                # Small delay between thread starts
                if i < nb_threads - 1:
                    start_event.clear()
                    start_event.wait()  # Wait for the thread to process and set the event

            # Wait for all threads to complete
            for thread in alice_threads:
                thread.join()

            logger.info("All Alice WCP threads completed")

            # Analyze results
            successful_keys = []
            failed_count = 0
            
            for i, result in enumerate(final_results):
                if result and len(result) == 2:
                    key, success = result
                    if success and key is not None:
                        successful_keys.append(key)
                        logger.info(f"Thread {i}: Generated key of length {len(key)}")
                    else:
                        failed_count += 1
                        logger.warning(f"Thread {i}: Failed to generate key")
                else:
                    failed_count += 1
                    logger.warning(f"Thread {i}: Invalid result format")

            # # Combine successful keys
            # if successful_keys:
            #     combined_key = []
            #     for key in successful_keys:
            #         combined_key.extend(key)
                
            #     logger.info(f"=== WCP BB84 Alice Results ===")
            #     logger.info(f"Successful threads: {len(successful_keys)}/{nb_threads}")
            #     logger.info(f"Failed threads: {failed_count}")
            #     logger.info(f"Total final key length: {len(combined_key)}")
            #     logger.info(f"Key efficiency: {len(combined_key)/key_length*100:.1f}%")
                
            #     # Save results to file
            #     results_file = f"alice_wcp_results_{int(time.time())}.txt"
            #     with open(results_file, 'w') as f:
            #         f.write("=== WCP BB84 Alice Results ===\n")
            #         f.write(f"Configuration:\n")
            #         f.write(f"  Signal intensity: {signal_intensity}\n")
            #         f.write(f"  Decoy intensity: {decoy_intensity}\n")
            #         f.write(f"  Signal probability: {signal_prob}\n")
            #         f.write(f"  Decoy probability: {decoy_prob}\n")
            #         f.write(f"  Vacuum probability: {vacuum_prob}\n")
            #         f.write(f"  Transmission efficiency: {transmission_efficiency}\n")
            #         f.write(f"\nResults:\n")
            #         f.write(f"  Successful threads: {len(successful_keys)}/{nb_threads}\n")
            #         f.write(f"  Total key length: {len(combined_key)}\n")
            #         f.write(f"  Key efficiency: {len(combined_key)/key_length*100:.1f}%\n")
            #         f.write(f"\nFinal key: {''.join(map(str, combined_key))}\n")
                
            #     logger.info(f"Results saved to {results_file}")
            # else:
            #     logger.error("No successful key generation!")

    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    except Exception as e:
        logger.error(f"Error in Alice WCP protocol: {e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("Closing Alice WCP connections...")
        try:
            connection.close()
            server.close()
        except:
            pass

    # Clean up role
    role_alice.clean()
    logger.info("Alice WCP protocol terminated")

import logging  # Add this import at the top
