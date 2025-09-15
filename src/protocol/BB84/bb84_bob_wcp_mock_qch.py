import sys
import os
import argparse
import threading
import queue
import pickle
import time
import logging

# Add parent directory to path if the module is there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
import logging
from examples.logging_setup import setup_logger
logger = setup_logger("WCP BB84 Bob Log", logging.INFO)

from classical_communication_channel.communication_channel.connection_info import ConnectionInfo
from classical_communication_channel.communication_channel.mac_config import MAC_Config, MAC_Algorithms 
from classical_communication_channel.communication_channel.role import Role

from BB84.bb84_protocol.bob_wcp_bb84_ccc import BobWCPQubits, BobWCPThread

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
# DETECTOR_EFFICIENCY_DEFAULT = 0.1
DETECTOR_EFFICIENCY_DEFAULT = 1
DARK_COUNT_RATE_DEFAULT = 1e-6

TEST_FRACTION = 0.1
ERROR_THRESHOLD = 0.1
NUM_THREADS_DEFAULT = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bob's side of the WCP BB84 protocol")
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
                       help="Expected signal intensity")
    parser.add_argument("-di", "--decoy_intensity", type=float, default=DECOY_INTENSITY_DEFAULT,
                       help="Expected decoy intensity")
    parser.add_argument("-vi", "--vacuum_intensity", type=float, default=VACUUM_INTENSITY_DEFAULT,
                       help="Expected vacuum intensity")
    parser.add_argument("-sp", "--signal_prob", type=float, default=SIGNAL_PROB_DEFAULT,
                       help="Probability of signal intensity")
    parser.add_argument("-dp", "--decoy_prob", type=float, default=DECOY_PROB_DEFAULT,
                       help="Probability of decoy intensity")
    parser.add_argument("-vp", "--vacuum_prob", type=float, default=VACUUM_PROB_DEFAULT,
                       help="Probability of vacuum intensity")
    parser.add_argument("-de", "--detector_efficiency", type=float, default=DETECTOR_EFFICIENCY_DEFAULT,
                       help="Detector efficiency")
    parser.add_argument("-dc", "--dark_count_rate", type=float, default=DARK_COUNT_RATE_DEFAULT,
                       help="Dark count rate")
    
    parser.add_argument("-tf", "--test_fraction", type=float, default=TEST_FRACTION, 
                       help="Fraction of the key to be used for testing")
    parser.add_argument("-et", "--error_threshold", type=float, default=ERROR_THRESHOLD,
                       help="Error threshold for parameter estimation")
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
    detector_efficiency = args.detector_efficiency
    dark_count_rate = args.dark_count_rate
    signal_intensity = args.signal_intensity
    decoy_intensity = args.decoy_intensity
    
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
    key_parts = [key_length // nb_threads for _ in range(nb_threads)]
    if key_parts[0] < bytes_per_frame:
        logger.error(f"!!! Key length {key_length} is too short for nb threads {nb_threads} and the bytes per frame {bytes_per_frame}!!!")
        sys.exit(1)
    num_frames_parts = [key_parts[i] // bytes_per_frame for i in range(nb_threads)]

    logger.info("=== WCP BB84 Bob Configuration ===")
    logger.info(f"Key length: {key_length}")
    logger.info(f"Detector efficiency: {detector_efficiency}")
    logger.info(f"Dark count rate: {dark_count_rate}")
    logger.info(f"Expected signal intensity: {signal_intensity}")
    logger.info(f"Expected decoy intensity: {decoy_intensity}")
    logger.info(f"Number of threads: {nb_threads}")

    # Setup classical communication channel
    mac_configuration = MAC_Config(MAC_Algorithms.CMAC, SHARED_SECRET_KEY)
    bob_info = ConnectionInfo(IP_ADDRESS, PORT_NUMBER_BOB)
    role_bob = Role.get_instance(bob_info, mac_config=mac_configuration)
    alice_info = ConnectionInfo(IP_ADDRESS, PORT_NUMBER_ALICE)
    
    logger.info(f"CCC: Role Bob - {bob_info.ip}:{bob_info.port} - connected to Alice - {alice_info.ip}:{alice_info.port}")
    
    role_bob.peer_connection_info = alice_info

    # Main protocol loop
    while True:
        try:
            # Set up quantum channel client
            connection = BobWCPQubits.setup_client(IP_ADDRESS, PORT_NUMBER_QUANTIC_CHANNEL)
            logger.info(f"Quantum channel connection established at {IP_ADDRESS}:{PORT_NUMBER_QUANTIC_CHANNEL}")

            # Initialize Bob components
            bob_queues = []
            bob_wcp_qubits = []
            bob_wcp_threads = []

            # Create instances for each thread
            for i in range(nb_threads):
                # Create queue for this thread
                bob_queue = queue.Queue()
                bob_queues.append(bob_queue)
                
                # Create WCP qubits handler
                # bob_wcp = BobWCPQubits(
                #     num_qubits=key_parts[i],
                #     num_frames=num_frames_parts[i],
                #     bytes_per_frame=bytes_per_frame,
                #     sync_frames=sync_frames,
                #     sync_bytes_per_frame=sync_bytes,
                #     detector_efficiency=detector_efficiency,
                #     dark_count_rate=dark_count_rate,
                #     signal_intensity=signal_intensity,
                #     decoy_intensity=decoy_intensity
                # )
                # bob_wcp_qubits.append(bob_wcp)
                
                # # Create WCP thread handler
                # bob_wcp_thread = BobWCPThread(
                #     role=role_bob,
                #     receive_queue=bob_queue,
                #     thread_id=i,
                #     test_bool=True,
                #     test_fraction=test_fraction,
                #     error_threshold=error_threshold,
                #     start_idx=i * key_parts[i],
                #     detector_efficiency=detector_efficiency,
                #     dark_count_rate=dark_count_rate
                # )
                # bob_wcp_threads.append(bob_wcp_thread)

            # Start the classical communication reader thread
            reader_thread = threading.Thread(target=read, args=[role_bob, bob_queues], daemon=True)
            reader_thread.start()

            def run_bob_wcp_thread(start_event: threading.Event, thread_id: int):
                """Run Bob WCP protocol for a specific thread"""
                try:
                    
                    logger.info(f"Thread {thread_id}: Starting WCP protocol...")
                    
                    bob_wcp = BobWCPQubits(
                        num_qubits=key_parts[i],
                        num_frames=num_frames_parts[i],
                        bytes_per_frame=bytes_per_frame,
                        sync_frames=sync_frames,
                        sync_bytes_per_frame=sync_bytes,
                        detector_efficiency=detector_efficiency,
                        dark_count_rate=dark_count_rate,
                        signal_intensity=signal_intensity,
                        decoy_intensity=decoy_intensity
                    )
                    # bob_wcp.run_bob_wcp_protocol(connection)
                    bob_wcp.run_mock_bob_qubits(connection, wait_time=True)
                    # Get detection statistics                    
                    stats = bob_wcp.get_detection_statistics()
                    logger.info(f"Thread {thread_id}: Detection statistics: {stats}")
                    bob_wcp_qubits.append(bob_wcp)

                    # Set WCP detection data for thread
                    bob_wcp_thread = BobWCPThread(
                        role=role_bob,
                        receive_queue=bob_queue,
                        thread_id=i,
                        test_bool=True,
                        test_fraction=test_fraction,
                        error_threshold=error_threshold,
                        start_idx=i * key_parts[i],
                        detector_efficiency=detector_efficiency,
                        dark_count_rate=dark_count_rate
                    )

                    bob_wcp_thread.set_wcp_detection_data(
                        detected_bits=bob_wcp.detected_bits,
                        measurement_bases=bob_wcp.detected_bases,
                        detected_pulse_bytes=bob_wcp.detected_pulse_bytes,
                        detected_indices=bob_wcp.detected_indices
                    )
                    # Set start signal
                    start_event.set()  # Signal the thread to start processing
                    
                    # Run classical communication protocol
                    final_key, success = bob_wcp_thread.run_bob_wcp_thread()
                    bob_wcp_threads.append(bob_wcp_thread)
                    print(f"Thread {thread_id}: COMPLETED")
                    
                    if success:
                        logger.info(f"Thread {thread_id}: WCP protocol completed successfully")
                        logger.info(f"Thread {thread_id}: Final key length: {len(final_key) if final_key else 0}")
                        return final_key, True
                    else:
                        logger.error(f"Thread {thread_id}: WCP protocol failed")
                        return None, False
                        
                except Exception as e:
                    logger.error(f"Thread {thread_id}: Exception in WCP protocol: {e}")
                    import traceback
                    traceback.print_exc()
                    return None, False

            # Run all threads
            bob_threads = []
            final_results = []
            start_event = threading.Event()
            
            # Start the threads for Bob
            for i in range(nb_threads):
                bob_thread = threading.Thread(target=run_bob_wcp_thread, args=(start_event, i))
                bob_threads.append(bob_thread)
                bob_thread.start()
                if i < nb_threads - 1:  # Don't wait after starting the last thread
                    start_event.clear()  # Clear the event for the next thread
                    start_event.wait()  # Wait for the thread to process and set the event

            # Wait for all threads to complete
            for thread in bob_threads:
                thread.join()

            logger.info("All Bob WCP threads completed")

            # Analyze results
            successful_keys = []
            failed_count = 0
            
            for i, bob_wcp_thread in enumerate(bob_wcp_threads):
                if bob_wcp_thread.test_success_bool:
                    successful_keys.append(bob_wcp_thread.final_key)
                    logger.info(f"Thread {i}: Key length: {len(bob_wcp_thread.final_key)}")
                else:
                    failed_count += 1
                    logger.error(f"Thread {i}: Failed to generate key")

            # # Combine successful keys and generate reports
            # if successful_keys:
            #     combined_key = []
            #     for key in successful_keys:
            #         combined_key.extend(key)
                
            #     logger.info(f"=== WCP BB84 Bob Results ===")
            #     logger.info(f"Successful threads: {len(successful_keys)}/{nb_threads}")
            #     logger.info(f"Failed threads: {failed_count}")
            #     logger.info(f"Total final key length: {len(combined_key)}")
            #     logger.info(f"Key efficiency: {len(combined_key)/key_length*100:.1f}%")
                
            #     # Generate parameter estimation report
            #     if len(bob_wcp_threads) > 0 and hasattr(bob_wcp_threads[0], 'parameter_estimator'):
            #         estimator = bob_wcp_threads[0].parameter_estimator
            #         report = estimator.get_summary_report()
            #         logger.info("\n" + report)
                
            #     # Save results to file
            #     results_file = f"bob_wcp_results_{int(time.time())}.txt"
            #     with open(results_file, 'w') as f:
            #         f.write("=== WCP BB84 Bob Results ===\n")
            #         f.write(f"Configuration:\n")
            #         f.write(f"  Detector efficiency: {detector_efficiency}\n")
            #         f.write(f"  Dark count rate: {dark_count_rate}\n")
            #         f.write(f"  Expected signal intensity: {signal_intensity}\n")
            #         f.write(f"  Expected decoy intensity: {decoy_intensity}\n")
            #         f.write(f"\nResults:\n")
            #         f.write(f"  Successful threads: {len(successful_keys)}/{nb_threads}\n")
            #         f.write(f"  Total key length: {len(combined_key)}\n")
            #         f.write(f"  Key efficiency: {len(combined_key)/key_length*100:.1f}%\n")
            #         f.write(f"\nFinal key: {''.join(map(str, combined_key))}\n")
                    
            #         # Add parameter estimation report
            #         if len(bob_wcp_threads) > 0 and hasattr(bob_wcp_threads[0], 'parameter_estimator'):
            #             f.write(f"\n{report}\n")
                
            #     logger.info(f"Results saved to {results_file}")
            # else:
            #     logger.error("No successful key generation!")

            # Break the loop on successful completion
            break

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break
        except Exception as e:
            logger.error(f"Error in Bob WCP protocol: {e}")
            import traceback
            traceback.print_exc()
            break
        finally:
            logger.info("Closing Bob WCP connections...")
            try:
                connection.close()
            except:
                pass

    # Clean up role
    role_bob.clean()
    logger.info("Bob WCP protocol terminated")
