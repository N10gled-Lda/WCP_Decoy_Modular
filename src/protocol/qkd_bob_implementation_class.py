from numpy.ma.extras import average

from .classical_communication_channel.communication_channel.connection_info import ConnectionInfo
from .classical_communication_channel.communication_channel.mac_config import MAC_Config, MAC_Algorithms
from .classical_communication_channel.communication_channel.role import Role
from .ErrorReconciliation.cascade.reconciliation import Reconciliation
from .PrivacyAmplification.privacy_amplification import PrivacyAmplification

from .BB84.bb84_protocol.bob_side_thread_ccc import BobQubits, BobThread

from .ErrorReconciliation.cascade.key import Key

import queue
import threading
import time
import argparse
import pickle



class QKDBobImplementation:
    """BobParticipant class establishes a connection with BobParticipant class to implement realistic QKD system on two different
    computers on the same network, responsible for Bob part.
    This class supports Single-Thread & Multi-Thread.
    """

    def __init__(self, ip_address_alice="0.0.0.0", port_number_alice=5000, ip_address_bob="127.0.0.3",
                 port_number_bob=65437, shared_secret_key=b'IzetXlgAnY4oye56'):
        """Instantiate important parameters for connection establishment."""

        self._ip_address_alice = ip_address_alice
        self._port_number_alice = port_number_alice
        self._ip_address_bob = ip_address_bob
        self._port_number_bob = port_number_bob
        self._shared_secret_key = shared_secret_key

        self._detected_bits = None
        self._detected_bases = None
        self._detected_qubits_bytes = None
        self._detected_idxs = None
        self._average_time_bin = None

        self._queues_list = []
        self._role_bob = None
        self._bob_threads = []
        self._threads_counter = 0

        # Statistical Variables
        #   Adds the BS Object for each Thread: Idx -> Thread
        self._bob_thread_bs_objs = {}
        #   Adds the ER Object for each Thread: Idx -> Thread
        self._bob_thread_er_objs = {}

    @staticmethod
    def read(role: Role, queues: list[queue.Queue]):
        """
        Reads the messages from channel with Bob and send it to the respective Thread.
        """
        try:
            while True:
                payload = role.get_from_inbox()
                #print(f"Received Message")
                (thread_id, data) = pickle.loads(payload)
                queues[thread_id].put(data)
        except KeyboardInterrupt:
            role.clean()

    def set_bob_detected_quantum_variables(self, detected_bits, detected_bases, detected_qubits_bytes, detected_idxs
                                           , average_time_bin):
        """
        Set up the Bob detected bits; bases; detected qubits bytes; indexes and time bin.
        Used to make classical process easily achievable changing from real implementation and a simulation of
        quantum channel.
        """

        self._detected_bits = detected_bits
        self._detected_bases = detected_bases
        self._detected_qubits_bytes = detected_qubits_bytes
        self._detected_idxs = detected_idxs
        self._average_time_bin = average_time_bin

    def start_read_communication_channel(self):
        reader_thread = threading.Thread(target=self.read, args=[self._role_bob, self._queues_list], daemon=True)
        reader_thread.start()

    def setup_role_bob(self, bandwidth_limit_megabytes_per_second=None, inbox_capacity_megabytes=None,
                       outbox_capacity_megabytes=None):
        """
          Creates the Bob Connection Object used to establish the classical communication channel with Alice.
        """
        # Setup the connection for the classical communication channel
        mac_configuration = MAC_Config(MAC_Algorithms.CMAC, self._shared_secret_key)
        #mac_configuration = None
        bob_info = ConnectionInfo(self._ip_address_bob, self._port_number_bob)
        role_bob = Role.get_instance(bob_info, mac_config=mac_configuration,
                                     bandwidth_limit_megabytes_per_second=bandwidth_limit_megabytes_per_second,
                                     inbox_capacity_megabytes=inbox_capacity_megabytes,
                                     outbox_capacity_megabytes=outbox_capacity_megabytes)

        alice_info = ConnectionInfo(self._ip_address_alice, self._port_number_alice)
        print(
            f"CCC: Role Bob - {bob_info.ip} : {bob_info.port} - is connected to the Role Alice - {alice_info.ip} : {alice_info.port}")
        role_bob.peer_connection_info = alice_info

        self._role_bob = role_bob

        return role_bob

    def bob_quantum_step(self, num_qubits, num_frames, bytes_per_frame, sync_frames,
                           sync_bytes_per_frame, fixed_error_rate=0.0, port_number_quantic_channel=13122):
        """
        Performs the simulation of a Quantum Channel through a Classical Channel.
        """
        connection = BobQubits.setup_client(self._ip_address_alice, port_number_quantic_channel)
        bob_qubit = BobQubits(
            num_qubits=num_qubits,
            num_frames=num_frames,
            bytes_per_frame=bytes_per_frame,
            sync_frames=sync_frames,
            sync_bytes_per_frame=sync_bytes_per_frame,
        )

        bool_fixed_error_rate = False if fixed_error_rate <= 0.0 else True
        

        #bob_qubit.run_bob_qubits(connection)
        bob_qubit.run_mock_bob_qubits(connection, fixed_fake_error_bool=bool_fixed_error_rate, fixed_fake_error_rate=fixed_error_rate)
        connection.close()
        print(f"Bob Qubits Received.")

        return bob_qubit

    def bob_process_base_sifting_classical_steps(self, role_bob, do_test, test_fraction, receive_queue, thread_id=0,
                                                 detected_bits=None, detected_bases=None, detected_qubits_bytes=None,
                                                 detected_idxs=None, average_time_bin=None):
        """
        Perform the Base-Sifting Step using the Classical Communication Channel.

        return an object that is then given to the ER process.
        """
        bob_queue = receive_queue

        bob_ccc = BobThread(
            test_bool=do_test,
            test_fraction=test_fraction,
            thread_id=thread_id,
            role=role_bob,
            receive_queue=bob_queue
        )
        # This if is required to separate single to multi-threading respectively
        var_detected_bits = self._detected_bits if detected_bits is None else detected_bits
        var_detected_bases = self._detected_bases if detected_bases is None else detected_bases
        #   TODO: (deprecation) alice qubytes equal to alice bits since alice qubytes is not used and soon to be deprecated
        var_detected_qubits_bytes = self._detected_qubits_bytes if detected_qubits_bytes is None \
            else detected_bits
        var_detected_idxs = self._detected_idxs if detected_idxs is None else detected_idxs
        var_average_time_bin = self._average_time_bin if average_time_bin is None else average_time_bin

        bob_ccc.set_bits_bases_qubits_idxs(var_detected_bits, var_detected_bases,
                                           var_detected_qubits_bytes, var_detected_idxs,
                                           var_average_time_bin)
        bob_ccc.run_bob_thread()

        # This is performed to stop execution in case key is rejected due to High QBER
        if len(bob_ccc.final_key) == 0:
            return

        self._bob_thread_bs_objs[thread_id] = bob_ccc
        print(f"Bob BS finished.")

        return bob_ccc


    def bob_process_error_correction_classical_steps(self, bob_ccc, role_bob, receive_queue, thread_id=0):
        """
        Performs the Error Correction process of Bob side using Communication Channel.

        return Reconciliation Object.
         """
        qber_percentage = bob_ccc.failed_percentage  # int
        qber_flt = round(qber_percentage / 100, 2)

        if qber_flt == 0.0:
            qber_flt = 0.01

        bob_key = bob_ccc.final_key

        noisy_key = Key.create_key_from_list(bob_key)

        if __debug__:
            print(f"Noisy Key Len({noisy_key.get_size()}):\n{noisy_key}")

        reconciliation = Reconciliation("original", None, noisy_key, qber_flt,
                                        thread_id=thread_id, role=role_bob, message_queue=receive_queue)

        reconciliation.reconcile_channel_threading()


        self._bob_thread_er_objs[thread_id] = reconciliation
        return reconciliation

    def bob_run_simulation_classical_process(self, do_test, test_fraction, privacy_amplification_compression_rate):
        """
        Performs all the steps using the classical channel, the Base Sifting, Error Correction.

        It is assumed that alice has bits and bases set up, as if they are none, the post-processing cannot be performed.
        """
        role_bob = self.setup_role_bob()

        # Create the list of the queue for receiving for each thread
        bob_queues = [queue.Queue() for _ in range(1)]
        reader_thread = threading.Thread(target=self.read, args=[role_bob, bob_queues], daemon=True)
        reader_thread.start()

        # Single Thread we only need the first Queue
        receive_queue = bob_queues[0]

        start_bs_time_tick = time.perf_counter()
        bob_ccc = self.bob_process_base_sifting_classical_steps(role_bob, do_test, test_fraction, receive_queue)
        end_bs_time_tick = time.perf_counter()

        start_er_time_tick = time.perf_counter()
        bob_er = self.bob_process_error_correction_classical_steps(bob_ccc, role_bob, receive_queue)
        end_er_time_tick = time.perf_counter()

        reconciled_key = bob_er.get_reconciled_key()

        print(f"Bob Reconciled Key Len ({reconciled_key.get_size()}):\n{reconciled_key}")

        #TODO: Perform Privacy Amplification
        reconciled_key_list = reconciled_key.generate_array()
        privacy_amplification_obj = PrivacyAmplification(reconciled_key_list)

        initial_key_length = reconciled_key.get_size()
        final_key_length = int(initial_key_length * privacy_amplification_compression_rate)

        start_pa_time_tick = time.perf_counter()
        _, _, secured_key = privacy_amplification_obj.do_privacy_amplification(initial_key_length, final_key_length)
        end_pa_time_tick = time.perf_counter()

        if __debug__:
            print(f"PA Bob Key Len ({len(secured_key)}):\n{secured_key}")

        #TODO: Calculate Execution Times
        print(f"BS Time: {end_bs_time_tick - start_bs_time_tick}")
        print(f"ER Time: {end_er_time_tick - start_er_time_tick}")
        print(f"Total Classical Time: {end_er_time_tick - start_bs_time_tick}")
        print(f"PA Time: {end_pa_time_tick - start_pa_time_tick}")
        print(f"Total: {end_pa_time_tick - start_bs_time_tick}")
        #TODO: Perform Multi-threading
        #TODO: Retrieve Data (Redundant if we use Simulator to obtain metrics about data sizes).
        #TODO: Maybe later perform the junction between this and QKD Simulator.

    def bob_run_qkd_classical_process(self, detected_bits, detected_bases, detected_qubits_bytes,
                                                 detected_idxs, average_time_bin, do_test, test_fraction,
                                                  privacy_amplification_compression_rate, receive_queue, thread_id):

        start_bs_time_tick = time.perf_counter()
        bob_ccc = self.bob_process_base_sifting_classical_steps(self._role_bob, do_test, test_fraction, receive_queue,
                                                                thread_id=thread_id,
                                                                detected_bits=detected_bits,
                                                                detected_bases=detected_bases,
                                                                detected_qubits_bytes=detected_qubits_bytes,
                                                                detected_idxs=detected_idxs,
                                                                average_time_bin=average_time_bin)
        end_bs_time_tick = time.perf_counter()

        start_er_time_tick = time.perf_counter()
        bob_er = self.bob_process_error_correction_classical_steps(bob_ccc, self._role_bob, receive_queue,
                                                                   thread_id=thread_id)
        end_er_time_tick = time.perf_counter()

        reconciled_key = bob_er.get_reconciled_key()

        if __debug__:
            print(f"Bob Reconciled Key Len ({reconciled_key.get_size()}):\n{reconciled_key}")

        # TODO: Perform Privacy Amplification
        reconciled_key_list = reconciled_key.generate_array()
        privacy_amplification_obj = PrivacyAmplification(reconciled_key_list)

        initial_key_length = reconciled_key.get_size()
        final_key_length = int(initial_key_length * privacy_amplification_compression_rate)

        start_pa_time_tick = time.perf_counter()
        _, _, secured_key = privacy_amplification_obj.do_privacy_amplification(initial_key_length, final_key_length)
        end_pa_time_tick = time.perf_counter()

        if __debug__:
            print(f"PA Bob Key Len ({len(secured_key)}):\n{secured_key}")

        # TODO: Calculate Execution Times
        print(f"BS Time: {end_bs_time_tick - start_bs_time_tick}")
        print(f"ER Time: {end_er_time_tick - start_er_time_tick}")
        print(f"Total Classical Time: {end_er_time_tick - start_bs_time_tick}")
        print(f"PA Time: {end_pa_time_tick - start_pa_time_tick}")
        print(f"Total: {end_pa_time_tick - start_bs_time_tick}")
        print(f"FINISHED THREAD: {thread_id}")
        # TODO: Retrieve Data.
        # TODO: Maybe later perform the junction between this and QKD Simulator.

    def bob_join_threads(self):
        for thread in self._bob_threads:
            thread.join()

    def bob_run_qkd_classical_process_threading(self, detected_bits, detected_bases, detected_qubits_bytes,
                                                detected_idxs, average_time_bin, do_test, test_fraction,
                                                privacy_amplification_compression_rate):
        #To Run This method Effectively is required to first run setup_role_bob & start_read_communication_channel
        print(f"bits: {len(detected_bits)} |"
              f"bases: {len(detected_bases)} |"
              f"qubits bytes: {len(detected_qubits_bytes)} |"
              f"idxs: {len(detected_idxs)}")
        #Run Thread that processes the classical channel process using threads
        #TODO: Define Privacy Amplification Compression Rate by performing formula from Security Proof.
        thread_id = self._threads_counter
        thread_receive_queue = queue.Queue()
        self._queues_list.append(thread_receive_queue)
        bob_thread = threading.Thread(target=self.bob_run_qkd_classical_process,
                                        args=(detected_bits, detected_bases, detected_qubits_bytes,
                                                 detected_idxs, average_time_bin, do_test, test_fraction,
                                                  privacy_amplification_compression_rate, thread_receive_queue, thread_id))
        self._bob_threads.append(bob_thread)
        self._threads_counter += 1
        bob_thread.start()


    def bob_produce_statistical_data(self):


        pass


