from classical_communication_channel.communication_channel.connection_info import ConnectionInfo
from classical_communication_channel.communication_channel.mac_config import MAC_Config, MAC_Algorithms
from classical_communication_channel.communication_channel.role import Role
from ErrorReconciliation.alice_sim import AliceChannel
from ErrorReconciliation.cascade.key import Key
from PrivacyAmplification.privacy_amplification import PrivacyAmplification

from BB84.bb84_protocol.alice_side_thread_ccc import AliceQubits, AliceThread
# from bb84_protocol.alice_side_thread_ccc import read

import queue
import threading
import time
import argparse
import pickle
import math

class QKDAliceImplementation:
    """QKDAliceImplementation class establishes a connection with BobParticipant class to implement realistic QKD system on two different
    computers on the same network, responsible for Alice part.
    This class supports Single-Thread & Multi-Thread.
    """

    def __init__(self, ip_address_alice="0.0.0.0", port_number_alice=5000, ip_address_bob="127.0.0.3",
                 port_number_bob=65437, shared_secret_key=b'IzetXlgAnY4oye56'):
        """Instantiate important parameters for connection establishment"""

        self._ip_address_alice = ip_address_alice
        self._port_number_alice = port_number_alice
        self._ip_address_bob = ip_address_bob
        self._port_number_bob = port_number_bob
        self._shared_secret_key = shared_secret_key

        #Bits and Bases used in Quantum channel - Used for separation between real implementation and simulation
        self._alice_bits = None
        self._alice_bases = None
        self._alice_qubytes = None
        #Used for multi-threading
        self._queues_list = []
        self._role_alice = None
        self._alice_threads = []
        self._threads_counter = 0

        # Statistical Variables
        #   Adds the BS Object for each Thread: Idx -> Thread
        self._alice_thread_bs_objs = {}
        #   Adds the ER Object for each Thread: Idx -> Thread
        self._alice_thread_er_objs = {}




    @staticmethod
    def read(role: Role, queues: list[queue.Queue]):
        """
        Reads the messages from channel with Bob and send it to the respective Thread
        TODO: Remove the requirement for this method by structuring the connection between classes using Role.
        """
        try:
            while True:
                payload = role.get_from_inbox()
                #print(f"Received Message")
                (thread_id, data) = pickle.loads(payload)
                queues[thread_id].put(data)
        except KeyboardInterrupt:
            role.clean()

    def set_alice_bits_and_bases(self, alice_bits, alice_bases, alice_qubytes):
        """
        Set up the Alice Bits and Bobs.
        Used to make classical process easily achievable changing from real implementation and a simulation of
        quantum channel.
        """
        self._alice_bits = alice_bits
        self._alice_bases = alice_bases
        self._alice_qubytes = alice_qubytes

    def start_read_communication_channel(self):
        """Method Starts listening to the channel to process messages - Used for multi-threads
        Should be used after setup_role_alice
        """
        reader_thread = threading.Thread(target=self.read, args=[self._role_alice, self._queues_list], daemon=True)
        reader_thread.start()

    def alice_quantum_step(self, num_qubits, qubit_delay_us, num_frames, bytes_per_frame, sync_frames,
                           sync_bytes_per_frame, loss_rate, port_number_quantic_channel=13122):
        """
        Performs the simulation of a Quantum Channel through a Classical Channel.
        """

        # Set up a simple socket connection for the qubits exchange
        server = AliceQubits.setup_server(self._ip_address_alice, port_number_quantic_channel, time=None)
        connection, address = server.accept()

        alice_qubit = AliceQubits(num_qubits=num_qubits,
                                  qubit_delay_us=qubit_delay_us,
                                  num_frames=num_frames,
                                  bytes_per_frame=bytes_per_frame,
                                  sync_frames=sync_frames,
                                  sync_bytes_per_frame=sync_bytes_per_frame,
                                  loss_rate=loss_rate,
                                  )

        #alice_qubit.run_alice_qubits(connection)
        alice_qubit.run_mock_alice_qubits(connection)
        print(f"Alice qubits sent")
        connection.close()
        return alice_qubit

    def setup_role_alice(self, bandwidth_limit_megabytes_per_second=None, inbox_capacity_megabytes=None,
                         outbox_capacity_megabytes=None):
        """
        Creates the Alice Connection Object used to establish the classical communication channel with Bob.
        """

        # Setup the connection for the classical communication channel
        # bandwidth_limit_MBps = 100  # Note: in MB/s; multiply by 8 to get Mbps
        mac_configuration = MAC_Config(MAC_Algorithms.CMAC, self._shared_secret_key)
        #mac_configuration = None
        alice_info = ConnectionInfo(self._ip_address_alice, self._port_number_alice)
        role_alice = Role.get_instance(alice_info, mac_config=mac_configuration,
                                       bandwidth_limit_megabytes_per_second=bandwidth_limit_megabytes_per_second,
                                       inbox_capacity_megabytes=inbox_capacity_megabytes,
                                       outbox_capacity_megabytes=outbox_capacity_megabytes)

        # bob_info = ConnectionInfo(self._ip_address_bob, self._port_number_bob)
        # print(
        #    f"CCC: Role Alice - {alice_info.ip} : {alice_info.port} - is connected to the Role Bob - {bob_info.ip} : {bob_info.port}")
        # role_alice.peer_connection_info = bob_info
        self._role_alice = role_alice

        return role_alice


    def alice_process_base_sifting_classical_steps(self, role_alice, do_test, test_fraction, error_threshold,
                                                   receive_queue, thread_id=0, alice_bits=None, alice_bases=None, alice_qubytes=None):
        """
        Perform the Base-Sifting Step using the Classical Communication Channel.

        return an object that is then given to the ER process.
        """

        alice_queue = receive_queue
        alice_ccc = AliceThread(
            test_bool=do_test,
            test_fraction=test_fraction,
            error_threshold=error_threshold,
            thread_id=thread_id,
            start_idx=0,
            role=role_alice,
            receive_queue=alice_queue
        )

        # This if is required to separate single to multi-threading respectively
        var_alice_bits = self._alice_bits if alice_bits is None else alice_bits
        var_alice_bases = self._alice_bases if alice_bases is None else alice_bases
        #   TODO: (deprecation) alice qubytes equal to alice bits since alice qubytes is not used and soon to be deprecated
        var_alice_qubytes = self._alice_qubytes if alice_qubytes is None else alice_bits


        alice_ccc.set_bits_bases_qubits(var_alice_bits, var_alice_bases, var_alice_qubytes)

        alice_ccc.run_alice_thread()

        self._alice_thread_bs_objs[thread_id] = alice_ccc
        print(f"Alice BS finished.")

        return alice_ccc


    def alice_process_error_correction_classical_steps(self, alice_ccc, role_alice, receive_queue, thread_id=0):
        """
        Performs the Error Correction process of Alice side using Communication Channel.

        return Reconciliation Object.
        """


        alice_key = alice_ccc.final_key

        correct_key = Key.create_key_from_list(alice_key)

        if __debug__:
            print(f"Alice Correct Key Len({correct_key.get_size()}):\n{correct_key}")

        alice_er = AliceChannel(thread_id_alice=thread_id, correct_key_alice=correct_key, participant=role_alice,
                                compression_rate=0, message_queue=receive_queue)

        alice_er.process_compute_parities_threading()

        self._alice_thread_er_objs[thread_id] = alice_er
        return alice_er

    def alice_run_simulation_classical_process(self, do_test, test_fraction, error_threshold, privacy_amplification_compression_rate):
        """
        Performs all the steps using the classical channel, the Base Sifting, Error Correction.

        It is assumed that alice has bits and bases set up, as if they are none, the post-processing cannot be performed.
        """

        role_alice = self.setup_role_alice()

        # Create the list of the queue for receiving for each thread
        alice_queues = [queue.Queue() for _ in range(1)]
        reader_thread = threading.Thread(target=self.read, args=[role_alice, alice_queues], daemon=True)
        reader_thread.start()

        #Single Thread we only need the first Queue
        receive_queue = alice_queues[0]

        start_bs_time_tick = time.perf_counter()
        alice_ccc = self.alice_process_base_sifting_classical_steps(role_alice, do_test, test_fraction, error_threshold, receive_queue)
        end_bs_time_tick = time.perf_counter()

        start_er_time_tick = time.perf_counter()
        alice_er = self.alice_process_error_correction_classical_steps(alice_ccc, role_alice, receive_queue)
        end_er_time_tick = time.perf_counter()
        correct_key = alice_er.correct_key

        if __debug__:
            print(f"Alice Correct Key Len ({correct_key.get_size()}:\n{correct_key}")

        # TODO: Perform Privacy Amplification
        correct_key_list = correct_key.generate_array()
        privacy_amplification_obj = PrivacyAmplification(correct_key_list)

        initial_key_length = correct_key.get_size()
        final_key_length = int(initial_key_length * privacy_amplification_compression_rate)

        start_pa_time_tick = time.perf_counter()
        _, _, secured_key = privacy_amplification_obj.do_privacy_amplification(initial_key_length, final_key_length)
        end_pa_time_tick = time.perf_counter()

        start_print_time = time.perf_counter()

        if __debug__:
            print(f"PA Alice Key Len ({len(secured_key)}):\n{secured_key}")

        # TODO: Calculate Execution Times
        print(f"BS Time: {end_bs_time_tick - start_bs_time_tick}")
        print(f"ER Time: {end_er_time_tick - start_er_time_tick}")
        print(f"Total Classical Time: {end_er_time_tick - start_bs_time_tick}")
        print(f"PA Time: {end_pa_time_tick - start_pa_time_tick}")
        print(f"Total: {end_pa_time_tick - start_bs_time_tick}")
        end_print_time = time.perf_counter()
        print(f"Print Time: {end_print_time - start_print_time}")


        # TODO: Perform Multi-threading
        # TODO: Retrieve Data (Redundant if we use Simulator to obtain metrics about data sizes).
        # TODO: Maybe later perform the junction between this and QKD Simulator.

    def alice_run_qkd_classical_process(self, alice_bits, alice_bases, do_test, test_fraction, error_threshold,
                                                  privacy_amplification_compression_rate, receive_queue, thread_id):

        start_bs_time_tick = time.perf_counter()
        alice_ccc = self.alice_process_base_sifting_classical_steps(self._role_alice, do_test, test_fraction, error_threshold,
                                                                    receive_queue, thread_id=thread_id, alice_bits=alice_bits, alice_bases=alice_bases, alice_qubytes=alice_bits)
        end_bs_time_tick = time.perf_counter()

        start_er_time_tick = time.perf_counter()
        alice_er = self.alice_process_error_correction_classical_steps(alice_ccc, self._role_alice, receive_queue, thread_id=thread_id)
        end_er_time_tick = time.perf_counter()
        correct_key = alice_er.correct_key

        if __debug__:
            print(f"Alice Correct Key Len ({correct_key.get_size()}):\n{correct_key}")

        # TODO: Perform Privacy Amplification
        correct_key_list = correct_key.generate_array()
        privacy_amplification_obj = PrivacyAmplification(correct_key_list)

        initial_key_length = correct_key.get_size()
        final_key_length = int(initial_key_length * privacy_amplification_compression_rate)

        start_pa_time_tick = time.perf_counter()
        _, _, secured_key = privacy_amplification_obj.do_privacy_amplification(initial_key_length, final_key_length)
        end_pa_time_tick = time.perf_counter()

        if __debug__:
            print(f"PA Alice Key Len ({len(secured_key)}):\n{secured_key}")
        # TODO: Calculate Execution Times
        print(f"BS Time: {end_bs_time_tick - start_bs_time_tick} | {thread_id}")
        print(f"ER Time: {end_er_time_tick - start_er_time_tick} | {thread_id}")
        print(f"Total Classical Time: {end_er_time_tick - start_bs_time_tick}")
        print(f"PA Time: {end_pa_time_tick - start_pa_time_tick}")
        print(f"Total: {end_pa_time_tick - start_bs_time_tick}")

    def alice_join_threads(self):

        for thread in self._alice_threads:
            thread.join()


    def alice_run_qkd_classical_process_threading(self, alice_bits, alice_bases, do_test, test_fraction, error_threshold,
                                                  privacy_amplification_compression_rate):

        #Run Thread that processes the classical channel process using threads
        #TODO: Define Privacy Amplification Compression Rate Asynchronously (Exchanging Messages)?
        thread_id = self._threads_counter
        thread_receive_queue = queue.Queue()
        self._queues_list.append(thread_receive_queue)
        alice_thread = threading.Thread(target=self.alice_run_qkd_classical_process,
                                        args=(alice_bits, alice_bases, do_test, test_fraction, error_threshold,
                                                  privacy_amplification_compression_rate, thread_receive_queue, thread_id))
        self._alice_threads.append(alice_thread)
        self._threads_counter += 1
        alice_thread.start()


    def alice_produce_statistical_data(self, use_mac=True):
        """This method simply comprises in printing and showing data"""

        frame_size_header_size_in_bytes = 0

        for key, value in self._alice_thread_bs_objs.items():
            bs_messages_sent = value.messages_sent
            bs_messages_received = value.messages_received

            # 4 + 16 due to MAC
            bs_total_size_messages_sent = sum(map(lambda x: x[4] + 4 + 16, bs_messages_sent))
            bs_total_size_messages_received = sum(map(lambda x: x[4] + 4 + 16, bs_messages_received))
            bs_number_messages_sent = value.number_messages_sent
            bs_number_messages_received = value.number_messages_received

            print(f"BS Thread: {key}\n Number Messages Sent:{bs_number_messages_sent} | Total Size: {bs_total_size_messages_sent} (Bytes)\n"
                  f"Number Messages Received: {bs_number_messages_received} | Total Size: {bs_total_size_messages_received} (Bytes)")

        for key, value in self._alice_thread_er_objs.items():

            er_messages_sent = value.messages_sent
            er_messages_received = value.messages_received

            # 4 + 16 due to MAC
            er_total_size_messages_sent = sum(map(lambda x: x[4] + 4 + 16, er_messages_sent))
            er_total_size_messages_received = sum(map(lambda x: x[4] + 4 + 16, er_messages_received))
            er_number_messages_sent = value.number_messages_sent
            er_number_messages_received = value.number_messages_received

            print(
                f"ER Thread: {key}\nNumber Messages Sent: {er_number_messages_sent} | Total Size: {er_total_size_messages_sent} (Bytes)\n"
                f"Number Messages Received: {er_number_messages_received} | Total Size: {er_total_size_messages_received} (Bytes)")




