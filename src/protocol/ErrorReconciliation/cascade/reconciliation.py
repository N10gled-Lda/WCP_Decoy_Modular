import copy
import heapq
import logging
import math
import pickle
import socket
import time
from collections import deque
from datetime import datetime
import os

#from Classical_Communication_Channel.communication_channel.classical_communication_packet import OutboxPacket
from classical_communication_channel.communication_channel.role import Role
from ErrorReconciliation.cascade.algorithm import get_algorithm_by_name
from ErrorReconciliation.cascade.block import Block
from ErrorReconciliation.cascade.key import Key
#from cascade.mock_classical_channel import MockClassicalChannel
from ErrorReconciliation.cascade.shuffle import Shuffle
from ErrorReconciliation.cascade.stats import Stats

# Auxiliar should not exit in final implementation
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
WHITE = '\033[97m'
RESET = '\033[0m'

GEO_DELAY = 0.238
LEO_DELAY = 0.0427
MEO_DELAY = 0.100


def _format_key(noisy_key, correct_key, start_index=0):
    """
    Format the key for visual representation. Differences between the noisy and correct key are highlighted.
    """
    if correct_key is None:
        return "Correct Key is not known"

    formatted_key = ""
    for index, bit in enumerate(noisy_key._bits.values()):
        if bit == correct_key.get_bit(index + start_index):
            # Green for correct bits
            formatted_key += '\033[92m' + str(bit) + '\033[0m'
        else:
            # Red for incorrect bits
            formatted_key += '\033[91m' + str(bit) + '\033[0m'
    return formatted_key


def _format_block_key(noisy_key_block, correct_key, start_index, end_index):
    """
        Format the key Block for visual representation. Differences between the noisy and correct key are highlighted.
    """

    if correct_key is None:
        return "Correct Key is not known"

    formatted_key = ""

    noisy_key_indexes = noisy_key_block.get_key_indexes()

    for index in noisy_key_indexes:
        bit = noisy_key_block.get_key_bit(index)
        if bit == correct_key.get_bit(index):
            formatted_key += '\033[92m' + str(bit) + '\033[0m'
        else:
            formatted_key += '\033[91m' + str(bit) + '\033[0m'

    return formatted_key


def _format_block_key_wo_shuffle(noisy_key_block, correct_key, start_index, end_index, shuffle):
    """
        Format the key Block, considering Shuffling for visual representation - The Shuffling is unmade. Differences between the noisy and correct key are highlighted.
    """
    if correct_key is None:
        return "Correct Key is not known"

    formatted_key = ""

    #noisy_key_indexes = noisy_key_block.get_key_indexes()


    for index in range(start_index, end_index):
        bit = noisy_key_block.get_key_bit(index)
        if bit == correct_key.get_bit(index):
            formatted_key += '\033[92m' + str(bit) + '\033[0m'
        else:
            formatted_key += '\033[91m' + str(bit) + '\033[0m'

    return formatted_key



class Reconciliation:
    """
    A single information reconciliation exchange between a client (Bob) and a server (Alice).
    """


    def __init__(self, algorithm_name: str, classical_channel, noisy_key: Key, estimated_bit_error_rate: float, aux_socket=None, thread_id: int=None,  role: Role=None, message_queue=None, bandwidth: int=-1, propagation_delay: int=0):
                 #number_r=0, number_runs=0, it_number=0, it_n_total=0, median_socket=None, ,
                 #send_buffer=None, lock=None,):
        """
        Create a Cascade reconciliation.

        Args:
            algorithm_name (str): The name of the Cascade algorithm.
            classical_channel (subclass of ClassicalChannel): The classical channel over which
                Bob communicates with Alice.
            noisy_key (Key): The noisy key as Bob received it from Alice that needs to be
                reconciliated.
            estimated_bit_error_rate (float): The estimated bit error rate in the noisy key.
        """

        # Store the arguments.
        self._classical_channel = classical_channel
        self._algorithm = get_algorithm_by_name(algorithm_name)
        assert self._algorithm is not None
        self._estimated_bit_error_rate = estimated_bit_error_rate
        self._noisy_key = noisy_key
        self._reconciled_key = None

        # Map key indexes to blocks.
        self._key_index_to_blocks = {}

        # USED FOR VISUALIZATION PURPOSES - SHOULD BE DELETED OR Change eventually
        # self.aux_copy_correct_key = self._classical_channel._correct_key

        # Keep track of statistics.
        self.stats = Stats()

        # A set of blocks that are suspected to contain an error, pending to be corrected later.
        # These are stored as a priority queue with items (block.size, block) so that we can correct
        # the pending blocks in order of shortest block first.
        self._pending_try_correct = []

        # A set of blocks for which we have to ask Alice for the correct parity. To minimize the
        # number of message that Bob sends to Alice (i.e. the number of channel uses), we queue up
        # these pending parity questions until we can make no more progress correcting errors. Then
        # we send a single message to Alice to ask all queued parity questions, and proceed once we
        # get the answers.
        self._pending_ask_correct_parity = []

        self.aux_ask_parity_bits_info = 0

        #self.my_ip_address = "127.0.0.1"
        #self.my_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.my_port = 5001
        # self.my_socket.bind((self.my_ip_address,self.my_port))
        #self.connection = None

        #self.alice_ip_address = "127.0.0.2"
        #self.alice_port = 5002

        #self.all_time_sleep = 0
        self.total_time = 0
        self.start_send = 0
        self.end_send = 0

        self.propagation_delay = propagation_delay
        self.bandwidth = bandwidth
        #self.number_r = number_r
        #self.number_runs = number_runs
        #self.it_number = it_number
        #self.it_n_total = it_n_total

        self.aux_socket = aux_socket

        self.largest_message_sent = (0, 0)
        self.largest_message_received = (0, 0)

        #self.send_buffer = send_buffer
        #self.receive_buffer = deque()
        self.receive_buffer = message_queue

        # Dictionary to save the block Objects based on Key: start index-end index, Value: Block
        # This is performed to guarantee that the block is not lost while using threading, and when receving correct parity of block to be able to change it
        #self.block_dict = {}
        #self.block_dict_aux = {}

        # This is done to simulate BB84 initial protocol in the workflow
        #self.median_socket = median_socket
        #self.key_divisions_bb84 = []
        self.thread_id = thread_id
        # This is made for threading and keep track of the errors that were corrected through multiple methods without return in _process_parity_messages method
        self.errors_corrected = 0

        #self.lock = lock
        self.role = role


        self.messages_sent = []
        self.number_messages_sent = 0
        self.messages_received = []
        self.number_messages_received = 0

        # LOGGING
        # Setup the logger
        if __debug__:
            if not os.path.exists("logs"):
                os.makedirs("logs")
            if not os.path.exists("log_files"):
                os.makedirs("log_files")
            self.logger = logging.getLogger(self.__class__.__name__)  # Use the class name as the logger name
            self.logger.setLevel(logging.DEBUG)  # Set the minimum level for the logger.
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_handler = logging.FileHandler(f'logs/{self.__class__.__name__.lower()}_{timestamp}.log', mode='w')
            file_handler.setLevel(logging.DEBUG)  # DEBUG and above will be written to the log file.

            self.file_logging = open(f'log_files/{self.__class__.__name__.lower()}_visual_{timestamp}.txt', "w",
                                     encoding='windows-1252')

            self.logger.addHandler(file_handler)

    # Method added by PF 4:01 PM 11/27/2024 For Debugging purposes
    def __repr__(self):
        return f"Reconciliation: algortihm: {self._algorithm}\n Classical Channel: {self._classical_channel}\n Estimated Bit Error Rate: {self._estimated_bit_error_rate}\n" \
               f"Noisy Key: {self._noisy_key}\n Reconciled Key: {self._reconciled_key}\n Key Index to Blocks: {self._key_index_to_blocks}\n Pending Try Correct: {self._pending_try_correct}\n" \
               f"Pending Ask Correct Parity: {self._pending_ask_correct_parity}\n"

    def get_noisy_key(self):
        """
        Get the noisy key, as Bob received it from Alice, that needs to be reconciled.

        Returns:
            The noisy key.
        """
        return self._noisy_key

    def get_reconciled_key(self):
        """
        Get the reconciled key, i.e. the key from which the reconciliation process attempted to
        remove the errors. There is still a small but non-zero chance that the reconciled key
        still contains errors.

        Returns:
            The reconciled key. None if the reconciliation process was not yet run.
        """
        return self._reconciled_key

    @staticmethod
    def _format_key_array(noisy_key, correct_key, start_index):
        """
        Format the key for visual representation. Differences between the noisy and correct key are highlighted.
        """
        if correct_key is None:
            return "Correct Key not Known"

        formatted_key = ""
        for index, bit in enumerate(noisy_key):
            if bit == correct_key.get_bit(index + start_index):
                # Green for correct bits
                formatted_key += '\033[92m' + str(bit) + '\033[0m'
            else:
                # Red for incorrect bits
                formatted_key += '\033[91m' + str(bit) + '\033[0m'
        return formatted_key

    def reconcile(self):
        """
        Run the Cascade algorithm to reconciliate our ("Bob's") noisy key with the server's
        ("Alice's") correct key.

        Returns:
            The reconciled key. There is still a small but non-zero chance that the corrected key
            still contains errors.
        """

        # Start measuring process and real time.
        start_process_time = time.process_time()
        start_real_time = time.perf_counter()

        if __debug__:
            self.print_message("Correct key (Alice):     " + _format_key(self._classical_channel._correct_key,
                                                                         self._classical_channel._correct_key))
            self.print_message(
                "Initial noisy key (Bob): " + _format_key(self._noisy_key, self._classical_channel._correct_key))

        # print("Correct key (Alice):     " + _format_key(self._classical_channel._correct_key, self._classical_channel._correct_key), file=self.file_logging)
        # print("Initial noisy key (Bob): " + _format_key(self._noisy_key, self._classical_channel._correct_key), file=self.file_logging)

        # Make a deep copy of the key, so that we continue to have access to the original noisy key.
        self._reconciled_key = copy.deepcopy(self._noisy_key)

        # Inform Alice that we are starting a new reconciliation.
        self._classical_channel.start_reconciliation()

        start_real_time_aux = time.perf_counter()
        # Do as many normal Cascade iterations as demanded by this particular Cascade algorithm.
        self._all_normal_cascade_iterations()
        end_real_time_aux = time.perf_counter()

        # Do as many normal BICONF iterations as demanded by this particular Cascade algorithm.
        self._all_biconf_iterations()

        # Inform Alice that we have finished the reconciliation.
        self._classical_channel.end_reconciliation()

        # Compute elapsed time.
        self.stats.elapsed_process_time = time.process_time() - start_process_time
        #self.stats.elapsed_real_time = time.perf_counter() - start_real_time
        self.stats.elapsed_real_time = end_real_time_aux - start_real_time_aux

        # Compute efficiencies.
        self.stats.unrealistic_efficiency = self._compute_efficiency(self.stats.ask_parity_blocks)
        # self.stats.unrealistic_efficiency = self._compute_efficiency(self.aux_ask_parity_bits_info + self.stats.reply_parity_bits + self.stats.start_iterations_bits)

        #
        # realistic_reconciliation_bits = self.stats.ask_parity_bits + self.stats.reply_parity_bits
        #print(self.aux_ask_parity_bits_info)
        #print(self.stats.reply_parity_bits)
        #print(self.stats.start_iterations_bits)

        realistic_reconciliation_bits = self.aux_ask_parity_bits_info + self.stats.reply_parity_bits + self.stats.start_iterations_bits
        # realistic_reconciliation_bits = self.stats.ask_parity_bits + self.stats.reply_parity_bits + self.stats.start_iterations_bits

        #print(f"ask parity blocks: {self.stats.ask_parity_blocks} \n {realistic_reconciliation_bits}")
        # realistic_reconciliation_bits /= self.stats.ask_parity_blocks
        self.stats.realistic_efficiency = self._compute_efficiency(realistic_reconciliation_bits)
        #print(f"recon_bits: {realistic_reconciliation_bits}")

        # Return the probably, but not surely, corrected key.
        if __debug__:
            self.print_message("Correct key (Alice):     " + _format_key(self._classical_channel._correct_key,
                                                                         self._classical_channel._correct_key))
            self.print_message(
                "Final noisy key (Bob):   " + _format_key(self._reconciled_key, self._classical_channel._correct_key))
            # print("Correct key (Alice):     " + _format_key(self._classical_channel._correct_key,
            #                                            self._classical_channel._correct_key), file=self.file_logging)
            # print("Final noisy key (Bob):   " + _format_key(self._reconciled_key, self._classical_channel._correct_key),
            #    file=self.file_logging)

        return self._reconciled_key

    def end_communication_channel(self):
        init_pack = b"begin_ask"

        end_pack = b"end_ask"
        self.aux_socket.sendall(init_pack)
        self.aux_socket.sendall(b"end_reconciliation_final")
        self.aux_socket.sendall(end_pack)

    def end_reconcile(self):
        init_pack = b"begin_ask"

        end_pack = b"end_ask"
        self.aux_socket.sendall(init_pack)
        self.aux_socket.sendall(b"end_reconciliation")
        self.aux_socket.sendall(end_pack)

    def reconcile_channel(self, propagation_delay: float=GEO_DELAY, bandwidth: float=-1):
        """
        Run the Cascade algorithm to reconciliate our ("Bob's") noisy key with the server's
        ("Alice's") correct key.

        Returns:
            The reconciled key. There is still a small but non-zero chance that the corrected key
            still contains errors.
        """

        # Start measuring process and real time.
        start_process_time = time.process_time()
        start_real_time = time.perf_counter()

        if __debug__:
            self.print_message("Correct key (Alice):     " + _format_key(self._classical_channel._correct_key,
                                                                         self._classical_channel._correct_key))
            self.print_message(
                "Initial noisy key (Bob): " + _format_key(self._noisy_key, self._classical_channel._correct_key))

        # print("Correct key (Alice):     " + _format_key(self._classical_channel._correct_key, self._classical_channel._correct_key), file=self.file_logging)
        # print("Initial noisy key (Bob): " + _format_key(self._noisy_key, self._classical_channel._correct_key), file=self.file_logging)

        self.propagation_delay = propagation_delay
        self.bandwidth = bandwidth

        # Make a deep copy of the key, so that we continue to have access to the original noisy key.
        self._reconciled_key = copy.deepcopy(self._noisy_key)


        # Inform Alice that we are starting a new reconciliation.
        # self._classical_channel.start_reconciliation()

        # Do as many normal Cascade iterations as demanded by this particular Cascade algorithm.
        self._all_normal_cascade_iterations_channel()

        # Do as many normal BICONF iterations as demanded by this particular Cascade algorithm.
        self._all_biconf_iterations()

        # Inform Alice that we have finished the reconciliation.
        self._classical_channel.end_reconciliation()

        # Compute elapsed time.
        self.stats.elapsed_process_time = time.process_time() - start_process_time
        self.stats.elapsed_real_time = time.perf_counter() - start_real_time

        # Compute efficiencies.
        self.stats.unrealistic_efficiency = self._compute_efficiency(self.stats.ask_parity_blocks)
        # self.stats.unrealistic_efficiency = self._compute_efficiency(self.aux_ask_parity_bits_info + self.stats.reply_parity_bits + self.stats.start_iterations_bits)

        # realistic_reconciliation_bits = self.stats.ask_parity_bits + self.stats.reply_parity_bits
        print(self.aux_ask_parity_bits_info)
        print(self.stats.reply_parity_bits)
        print(self.stats.start_iterations_bits)

        realistic_reconciliation_bits = self.aux_ask_parity_bits_info + self.stats.reply_parity_bits + self.stats.start_iterations_bits
        # realistic_reconciliation_bits = self.stats.ask_parity_bits + self.stats.reply_parity_bits + self.stats.start_iterations_bits

        # realistic_reconciliation_bits /= self.stats.ask_parity_blocks
        self.stats.realistic_efficiency = self._compute_efficiency(realistic_reconciliation_bits)

        print(f"ask parity blocks: {self.stats.ask_parity_blocks} \n {realistic_reconciliation_bits}")
        print(f"recon_bits: {realistic_reconciliation_bits}")

        # Return the probably, but not surely, corrected key.
        if __debug__:
            self.print_message("Correct key (Alice):     " + _format_key(self._classical_channel._correct_key,
                                                                         self._classical_channel._correct_key))
            self.print_message(
                "Final noisy key (Bob):   " + _format_key(self._reconciled_key, self._classical_channel._correct_key))

        self.stats.n_chunks_per_message /= self.stats.ask_parity_messages
        self.stats.n_chunks_per_message_received /= self.stats.ask_parity_messages
        self.stats.avg_time_to_send_message = self.stats.total_time_to_send_message / self.stats.ask_parity_messages
        self.stats.avg_n_blocks_per_message = self.stats.ask_parity_blocks / self.stats.ask_parity_messages
        self.stats.avg_time_to_receive_message = self.stats.total_time_to_receive_message / self.stats.ask_parity_messages
        self.stats.avg_bytes_per_block = self.stats.ask_parity_bytes / self.stats.ask_parity_blocks
        self.stats.avg_bytes_per_message = self.stats.ask_parity_bytes / self.stats.ask_parity_messages
        self.stats.reply_parity_bytes_per_message = self.stats.reply_parity_bytes / self.stats.ask_parity_messages

        # This formula is calculated due to diffulty and low performance in calculating number of chunks for each block individually.
        self.stats.n_chunks_per_block = self.stats.n_chunks_per_message / self.stats.avg_n_blocks_per_message
        self.stats.largest_message_sent = self.largest_message_sent
        self.stats.largest_message_received = self.largest_message_received

        print(f"Largest Message Sent: {self.stats.largest_message_sent}")
        print(f"Largest Message Received: {self.largest_message_received}")

        print("Bob Connection Closed")

        return self._reconciled_key

    # Essentially same as before mas using thread_id for separation
    def reconcile_channel_threading(self, propagation_delay: float=GEO_DELAY, bandwidth: float=-1):
        """
        Run the Cascade algorithm to reconciliate our ("Bob's") noisy key with the server's
        ("Alice's") correct key.

        Returns:
            The reconciled key. There is still a small but non-zero chance that the corrected key
            still contains errors.
        """

        # Start measuring process and real time.
        start_process_time = time.process_time()
        start_real_time = time.perf_counter()

        if __debug__:
            try:
                self.print_message("Correct key (Alice):     " + _format_key(self._classical_channel._correct_key,
                                                                             self._classical_channel._correct_key))
                self.print_message(
                    "Initial noisy key (Bob): " + _format_key(self._noisy_key, self._classical_channel._correct_key))
            except AttributeError:
                #In Case Classical Channel is None continue without logging - Performed this way to reuse in case of logging implementation
                pass


        self.propagation_delay = propagation_delay
        self.bandwidth = bandwidth

        # Make a deep copy of the key, so that we continue to have access to the original noisy key.
        self._reconciled_key = copy.deepcopy(self._noisy_key)

        # Inform Alice that we are starting a new reconciliation.
        # self._classical_channel.start_reconciliation()

        # Do as many normal Cascade iterations as demanded by this particular Cascade algorithm.
        # Same as before mas for threadid
        self._all_normal_cascade_iterations_channel_threading()
        self.end_reconcile_message()


        # Do as many normal BICONF iterations as demanded by this particular Cascade algorithm.
        #self._all_biconf_iterations()

        # Inform Alice that we have finished the reconciliation.
        #self._classical_channel.end_reconciliation()

        # Compute elapsed time.
        self.stats.elapsed_process_time = time.process_time() - start_process_time
        self.stats.elapsed_real_time = time.perf_counter() - start_real_time

        # Compute efficiencies.
        self.stats.unrealistic_efficiency = self._compute_efficiency(self.stats.ask_parity_blocks)
        # self.stats.unrealistic_efficiency = self._compute_efficiency(self.aux_ask_parity_bits_info + self.stats.reply_parity_bits + self.stats.start_iterations_bits)

        # realistic_reconciliation_bits = self.stats.ask_parity_bits + self.stats.reply_parity_bits
        #print(self.aux_ask_parity_bits_info)
        #print(self.stats.reply_parity_bits)
        #print(self.stats.start_iterations_bits)

        realistic_reconciliation_bits = self.aux_ask_parity_bits_info + self.stats.reply_parity_bits + self.stats.start_iterations_bits
        # realistic_reconciliation_bits = self.stats.ask_parity_bits + self.stats.reply_parity_bits + self.stats.start_iterations_bits

        # realistic_reconciliation_bits /= self.stats.ask_parity_blocks
        self.stats.realistic_efficiency = self._compute_efficiency(realistic_reconciliation_bits)

        #print(f"ask parity blocks: {self.stats.ask_parity_blocks} \n {realistic_reconciliation_bits} | Thread Id: {self.thread_id}")
        #print(f"recon_bits: {realistic_reconciliation_bits}")

        # Return the probably, but not surely, corrected key.
        if __debug__:
            try:
                self.print_message("Correct key (Alice):     " + _format_key(self._classical_channel._correct_key,
                                                                             self._classical_channel._correct_key))
                self.print_message(
                    "Final noisy key (Bob):   " + _format_key(self._reconciled_key, self._classical_channel._correct_key))
            except AttributeError:
                #In Case Classical Channel is None continue without logging - Performed this way to reuse in case of logging implementation
                pass

        self.stats.n_chunks_per_message /= self.stats.ask_parity_messages
        self.stats.n_chunks_per_message_received /= self.stats.ask_parity_messages
        self.stats.avg_time_to_send_message = self.stats.total_time_to_send_message / self.stats.ask_parity_messages
        self.stats.avg_n_blocks_per_message = self.stats.ask_parity_blocks / self.stats.ask_parity_messages
        self.stats.avg_time_to_receive_message = self.stats.total_time_to_receive_message / self.stats.ask_parity_messages
        self.stats.avg_bytes_per_block = self.stats.ask_parity_bytes / self.stats.ask_parity_blocks
        self.stats.avg_bytes_per_message = self.stats.ask_parity_bytes / self.stats.ask_parity_messages
        self.stats.reply_parity_bytes_per_message = self.stats.reply_parity_bytes / self.stats.ask_parity_messages

        # This formula is calculated due to diffulty and low performance in calculating number of chunks for each block individually.
        self.stats.n_chunks_per_block = self.stats.n_chunks_per_message / self.stats.avg_n_blocks_per_message
        self.stats.largest_message_sent = self.largest_message_sent
        self.stats.largest_message_received = self.largest_message_received

        #print(f"Largest Message Sent: {self.stats.largest_message_sent}")
        #print(f"Largest Message Received: {self.largest_message_received}")

        #print(f"Bob Connection Closed for Thread Id: {self.thread_id}")

        return self._reconciled_key

    def end_reconcile_message(self):

        data_to_send = (self.thread_id, "end_reconciliation")

        package_data = pickle.dumps(data_to_send)


        #packet = OutboxPacket(2, package_data)

        #self.role.outbox.put(packet)
        self.role.put_in_outbox(package_data)
        #self.messages_sent.append(("B", "S", self.number_messages_sent, time.process_time(), len(package_data), 1, self.thread_id))
        self.messages_sent.append(
            ("B", "S", self.number_messages_sent, time.perf_counter(), len(package_data), 1, self.thread_id))
        self.number_messages_sent += 1

    def _register_block_key_indexes(self, block: Block):
        # For every key bit covered by the block, append the block to the list of blocks that depend
        # on that partical key bit.
        for key_index in block.get_key_indexes():
            if key_index in self._key_index_to_blocks:
                self._key_index_to_blocks[key_index].append(block)
            else:
                self._key_index_to_blocks[key_index] = [block]

    def _get_blocks_containing_key_index(self, key_index: int):
        return self._key_index_to_blocks.get(key_index, [])

    def print_message(self, message: str, end: str="\n"):
        print(message, end=end, file=self.file_logging)

    def _correct_parity_is_known_or_can_be_inferred(self, block: Block):

        # Is the parity of the block already known?
        if block.get_correct_parity() is not None:
            return True

        # Try to do a very limited type of inference, using only the parity of the parent block and
        # the sibling block.

        # Cannot infer if there is no parent block.
        parent_block = block.get_parent_block()
        if parent_block is None:
            return False

        # Cannot infer if there is no sibling block (yet).
        if parent_block.get_left_sub_block() == block:
            sibling_block = parent_block.get_right_sub_block()
        else:
            sibling_block = parent_block.get_left_sub_block()
        if sibling_block is None:
            if __debug__:
                self.print_message("Cannot infer if there is no sibling block (yet)")
            return False

        # Cannot infer if the correct parity of the parent or sibling block are unknown.
        correct_parent_parity = parent_block.get_correct_parity()
        if correct_parent_parity is None:
            return False
        correct_sibling_parity = sibling_block.get_correct_parity()
        if correct_sibling_parity is None:
            return False

        # We have everything we need. Infer the correct parity.
        if correct_parent_parity == 1:
            correct_block_parity = 1 - correct_sibling_parity
        else:
            correct_block_parity = correct_sibling_parity

        if __debug__:
            self.print_message(
                f"We have everything we need to Infer the correct parity. Parent Correct Parity: {CYAN}{correct_parent_parity}{RESET} "
                f"| Sibling Correct Parity: {CYAN}{correct_sibling_parity}{RESET} => Hence this block Correct parity is {CYAN}{correct_block_parity}{RESET}")

        block.set_correct_parity(correct_block_parity)
        self.stats.infer_parity_blocks += 1
        return True

    def _schedule_ask_correct_parity(self, block, correct_right_sibling):
        # Adding an item to the end (not the start!) of a list is an efficient O(1) operation.
        entry = (block, correct_right_sibling)
        self._pending_ask_correct_parity.append(entry)


    @staticmethod
    def _produce_key_string(start_index: int, end_index: int):
        return f"{start_index}-{end_index}"

    def _have_pending_ask_correct_parity(self):
        return self._pending_ask_correct_parity != []

    @staticmethod
    def _bits_in_int(int_value: int):
        '''bits = 0
        while int_value != 0:
            bits += 1
            int_value //= 2
        if bits == 0:
            bits = 1
        return bits'''
        # Changed by PF 2:42 PM 12/2/2024
        if int_value == 0:
            return 1
        # print(f"int_value: {int_value}")
        return int((math.log(int_value) / math.log(2)) + 1)

    # Added by PF 3:54 PM 12/2/2024
    # using bit_length() improves efficiency by a lot Maybe calling this method would be redundant
    @staticmethod
    def _bits_in_int_multiple(int_value_a: int, int_value_b: int, int_value_c: int):
        a = max(int_value_a.bit_length(), 1)
        b = max(int_value_b.bit_length(), 1)
        c = max(int_value_c.bit_length(), 1)
        # print(f"Block Shuffle Identifier: {int_value_a} ; Identifier Bits: {a}\nBlock Start Index: {int_value_b} ; Index Bits: {b}\n Block End Index: {int_value_c} ; Index Bits: {c}\n")

        return a + b + c

    @staticmethod
    def _bits_in_block_ask_parity(block: Block):
        shuffle_identifier = block.get_shuffle().get_identifier()
        shuffle_start_index = block.get_start_index()
        shuffle_end_index = block.get_end_index()
        # return Reconciliation._bits_in_int(shuffle_identifier) + \
        #       Reconciliation._bits_in_int(shuffle_start_index) + \
        #       Reconciliation._bits_in_int(shuffle_end_index)
        return Reconciliation._bits_in_int_multiple(shuffle_identifier, shuffle_start_index, shuffle_end_index)

    def _service_pending_ask_correct_parity_secure_optimized(self):

        if not self._pending_ask_correct_parity:
            return

        # Prepare the question for Alice, i.e. the list of shuffle ranges over which we want Alice
        # to compute the correct parity.
        # VER EFICIENCIA DESTA PARTE TALVEZ SEJA POSSIVEL NAO NECESSITAR DESTE CICLO FOR  POIS COMPLEXIDADE O(N) para apenas colocar os blocks numa variavel auxiliar podendo a utilidade sendo mais para estatisticas
        ask_parity_blocks = []
        ask_parity_blocks_secure_optimized = []
        # print(f"--------------------------------\nMessage: {self.stats.ask_parity_messages}")
        if __debug__:
            self.logger.debug(f"--------------------------------\nMessage: {self.stats.ask_parity_messages}")
            self.print_message(f"--------------------------------\n{YELLOW}Message SENDING: {self.stats.ask_parity_messages}{RESET}")
            log_aux_parity_bits_message = self.stats.ask_parity_bits
            log_aux_number_block = 0
            log_aux_ask_parity_bits_info = self.aux_ask_parity_bits_info

        for entry in self._pending_ask_correct_parity:
            (block, _correct_right_sibling) = entry

            # print(f"------------\n{repr(block)}\n{str(block)}\nBlock Nr: {log_aux_number_block}")


            ask_parity_blocks.append(block)

            a = block.get_start_index().bit_length()
            b = block.get_end_index().bit_length()

            block_start_index = block.get_start_index()
            block_end_index = block.get_end_index()
            block_shuffle_object = block.get_shuffle()

            shuffle_index_to_key_index_dict = block_shuffle_object.get_shuffle_index_to_key_index()

            block_secure_obj_opt = (block_start_index, block_end_index, shuffle_index_to_key_index_dict)

            ask_parity_blocks_secure_optimized.append(block_secure_obj_opt)

            # self.aux_ask_parity_bits_info += a + b

            # Bits inside a block include: iteration nr | Start Index | End Index
            self.aux_ask_parity_bits_info += 2 + 18 + 18

            if __debug__:
                try:
                    self.logger.debug(f"------------\n{repr(block)}\n{str(block)}\nBlock Nr: {log_aux_number_block}")
                    self.print_message(f"------------\n{repr(block)}\n{str(block)}\n{_format_block_key(block, self._classical_channel._correct_key, block.get_start_index(), block.get_end_index())}\n{_format_block_key_wo_shuffle(block, self._classical_channel._correct_key, block.get_start_index(), block.get_end_index(), block.get_shuffle())}\n{BLUE}Block Nr{RESET}: {log_aux_number_block} | Indexes {YELLOW}{block_start_index}{RESET}:{YELLOW}{block_end_index}{RESET} | Current Parity: {CYAN}{block.get_current_parity()}{RESET}")
                except AttributeError:
                    # In Case Classical Channel is None continue without logging - Performed this way to reuse in case of logging implementation
                    pass

            aux = self.stats.ask_parity_bits
            if __debug__:
                self.logger.debug(
                    f"Block Shuffle Identifier: {block.get_shuffle().get_identifier()} ; Identifier Bits: {block.get_shuffle().get_identifier().bit_length()}\nBlock Start Index: {block.get_start_index()} ; Index Bits: {a}\n Block End Index: {block.get_end_index()} ; Index Bits: {b}\n")

            # This is made since we do not send the identifier anymore
            self.stats.ask_parity_bits += self._bits_in_block_ask_parity(block)
            # self.stats.ask_parity_bits += 2 + 18 + 18

            if __debug__:
                log_aux_number_block += 1
                # print(f"Bits in Block: {self.stats.ask_parity_bits - aux} ; Total Ask Parity Bits: {self.stats.ask_parity_bits}\n------------")
                self.logger.debug(
                    f"Bits in Block: {self.stats.ask_parity_bits - aux} ; Total Ask Parity Bits: {self.stats.ask_parity_bits} ; New Bits in Block: {a + b} ; "
                    f"Total New Ask Parity Bits: {self.aux_ask_parity_bits_info}\n------------")

                log_aux_parity_bits_message = self.stats.ask_parity_bits - log_aux_parity_bits_message
                # print(f"This Message Contains Number of Blocks: {log_aux_number_block} ; Total Bits send in this Message: {log_aux_parity_bits_message}\n--------------------------------")
                self.logger.debug(
                    f"This Message Contains Number of Blocks: {log_aux_number_block} ; Total Bits send in this Message: {log_aux_parity_bits_message} ; Total NEW Bits send in this Message: {self.aux_ask_parity_bits_info - log_aux_ask_parity_bits_info}\n--------------------------------")

        # "Send a message" to Alice to ask her to compute the correct parities for the list that
        # we prepared. For now, this is a synchronous blocking operations (i.e. we block here
        # until we get the answer from Alice).
        self.stats.ask_parity_messages += 1
        self.stats.ask_parity_blocks += len(ask_parity_blocks)

        #print(f"Ask Correct Parities of Blocks: {ask_parity_blocks_secure_optimized}")

        correct_parities = self._classical_channel.ask_parities_secure_optimized(ask_parity_blocks_secure_optimized)

        #print(len(correct_parities))
        #print(len(self._pending_ask_correct_parity))

        if __debug__:
            self.print_message(f"--------------------------------\n{YELLOW}Message RECEIVED: {self.stats.ask_parity_messages}{RESET}")

        # Process the answer from Alice. IMPORTANT: Alice is required to send the list of parities
        # in the exact same order as the ranges in the question; this allows us to zip.
        for (response_array, entry) in zip(correct_parities, self._pending_ask_correct_parity):
            self.stats.reply_parity_bits += 1
            (block, correct_right_sibling) = entry
            start_idx, end_idx, correct_parity = response_array
            #print(f"{start_idx}:{end_idx}")
            #print(block)
            block.set_correct_parity(correct_parity)
            if __debug__:
                try:
                    self.print_message(
                        f"------------\n{repr(block)}\n{str(block)}\n{_format_block_key(block, self._classical_channel._correct_key, block.get_start_index(), block.get_end_index())}\n{_format_block_key_wo_shuffle(block, self._classical_channel._correct_key, block.get_start_index(), block.get_end_index(), block.get_shuffle())}\n{BLUE}Block Nr{RESET}: {log_aux_number_block} | Indexes {YELLOW}{start_idx}{RESET}:{YELLOW}{end_idx}{RESET} | Current Parity: {CYAN}{block.get_current_parity()}{RESET} | Correct Parity: {CYAN}{block.get_correct_parity()}{RESET}")
                except AttributeError:
                    # In Case Classical Channel is None continue without logging - Performed this way to reuse in case of logging implementation
                    pass

            self._schedule_try_correct(block, correct_right_sibling)

        # Clear the list of pending questions.
        self._pending_ask_correct_parity = []



    def _service_pending_ask_correct_parity(self):

        if not self._pending_ask_correct_parity:
            return

        # Prepare the question for Alice, i.e. the list of shuffle ranges over which we want Alice
        # to compute the correct parity.
        # VER EFICIENCIA DESTA PARTE TALVEZ SEJA POSSIVEL NAO NECESSITAR DESTE CICLO FOR  POIS COMPLEXIDADE O(N) para apenas colocar os blocks numa variavel auxiliar podendo a utilidade sendo mais para estatisticas
        ask_parity_blocks = []
        # print(f"--------------------------------\nMessage: {self.stats.ask_parity_messages}")
        if __debug__:
            self.logger.debug(f"--------------------------------\nMessage: {self.stats.ask_parity_messages}")
            log_aux_parity_bits_message = self.stats.ask_parity_bits
            log_aux_number_block = 0
            log_aux_ask_parity_bits_info = self.aux_ask_parity_bits_info

        for entry in self._pending_ask_correct_parity:
            (block, _correct_right_sibling) = entry

            # print(f"------------\n{repr(block)}\n{str(block)}\nBlock Nr: {log_aux_number_block}")
            if __debug__:
                self.logger.debug(f"------------\n{repr(block)}\n{str(block)}\nBlock Nr: {log_aux_number_block}")

            ask_parity_blocks.append(block)

            a = block.get_start_index().bit_length()
            b = block.get_end_index().bit_length()

            # self.aux_ask_parity_bits_info += a + b

            # Bits inside a block include: iteration nr | Start Index | End Index
            self.aux_ask_parity_bits_info += 2 + 18 + 18

            aux = self.stats.ask_parity_bits
            if __debug__:
                self.logger.debug(
                    f"Block Shuffle Identifier: {block.get_shuffle().get_identifier()} ; Identifier Bits: {block.get_shuffle().get_identifier().bit_length()}\nBlock Start Index: {block.get_start_index()} ; Index Bits: {a}\n Block End Index: {block.get_end_index()} ; Index Bits: {b}\n")

            # This is made since we do not send the identifier anymore
            self.stats.ask_parity_bits += self._bits_in_block_ask_parity(block)
            # self.stats.ask_parity_bits += 2 + 18 + 18

            if __debug__:
                log_aux_number_block += 1
                # print(f"Bits in Block: {self.stats.ask_parity_bits - aux} ; Total Ask Parity Bits: {self.stats.ask_parity_bits}\n------------")
                self.logger.debug(
                    f"Bits in Block: {self.stats.ask_parity_bits - aux} ; Total Ask Parity Bits: {self.stats.ask_parity_bits} ; New Bits in Block: {a + b} ; "
                    f"Total New Ask Parity Bits: {self.aux_ask_parity_bits_info}\n------------")

                log_aux_parity_bits_message = self.stats.ask_parity_bits - log_aux_parity_bits_message
                # print(f"This Message Contains Number of Blocks: {log_aux_number_block} ; Total Bits send in this Message: {log_aux_parity_bits_message}\n--------------------------------")
                self.logger.debug(
                    f"This Message Contains Number of Blocks: {log_aux_number_block} ; Total Bits send in this Message: {log_aux_parity_bits_message} ; Total NEW Bits send in this Message: {self.aux_ask_parity_bits_info - log_aux_ask_parity_bits_info}\n--------------------------------")

        # "Send a message" to Alice to ask her to compute the correct parities for the list that
        # we prepared. For now, this is a synchronous blocking operations (i.e. we block here
        # until we get the answer from Alice).
        self.stats.ask_parity_messages += 1
        self.stats.ask_parity_blocks += len(ask_parity_blocks)
        correct_parities = self._classical_channel.ask_parities(ask_parity_blocks)

        # Process the answer from Alice. IMPORTANT: Alice is required to send the list of parities
        # in the exact same order as the ranges in the question; this allows us to zip.
        for (correct_parity, entry) in zip(correct_parities, self._pending_ask_correct_parity):
            self.stats.reply_parity_bits += 1
            (block, correct_right_sibling) = entry
            block.set_correct_parity(correct_parity)
            self._schedule_try_correct(block, correct_right_sibling)

        # Clear the list of pending questions.
        self._pending_ask_correct_parity = []

    def add_receiving_information(self, start_end_block: tuple):
        self.receive_buffer.appendleft(start_end_block)
        # print(f"Adding Information to Bob Thread Id: {self.thread_id} | Len First element: {len(start_end_block[0])}")

    def _receive_message_from_alice(self):
        data_mss = b""
        # Bob will wait for Alice response - Careful infinite cycle

        start_time = time.perf_counter()
        end_time = 0
        aux = 0

        n_chunks = 0

        in_message = False

        try:
            while True:
                chunk_mss = self.aux_socket.recv(4096)

                if b"begin_reply" in chunk_mss:
                    in_message = True
                    if __debug__:
                        print("Initializing Receiving Reply")
                    start_time = time.perf_counter()

                    chunk_mss = chunk_mss.split(b"begin_reply", 1)[1]

                if b"end_reply" in chunk_mss:
                    in_message = False
                    if __debug__:
                        print("Ending Receiving Reply")
                    end_time = time.perf_counter()
                    chunk_mss, _ = chunk_mss.split(b"end_reply", 1)
                    data_mss += chunk_mss
                    break

                if in_message:
                    data_mss += chunk_mss
                    n_chunks += 1
                    self.stats.total_n_chunks_received += 1
                    if __debug__:
                        print("received_chunk")


        except socket.timeout:
            print("Timeout reached, no data received")

        if start_time != 0:
            if __debug__:
                print(
                    f"total time sleep receiving message: Start: {start_time} | End: {end_time} | Time in seconds {(end_time - start_time) + self.propagation_delay} seconds")
            # time.sleep((end_time - start_time) * 1.238)
            aux = (end_time - start_time) + self.propagation_delay

        calc_time = len(data_mss) / self.bandwidth

        self.stats.total_time_to_receive_message += calc_time + self.propagation_delay

        return data_mss, aux

    #This method is not in use as messages are sent with Sendall from socket, and bandwidth limitations are calculated mathematically
    def send_with_bandwidth_limit(self, connection, data):
        bytes_sent = 0
        start_time = time.perf_counter()

        chunk_counter = 0
        n_chunks_per_message = 0

        time_to_sleep = 0

        for i in range(0, len(data), 4096):  # Send in chunks of 4096 bytes
            chunk = data[i:i + 4096]
            connection.sendall(chunk)
            chunk_bytes = len(chunk)
            bytes_sent += chunk_bytes

            chunk_counter += 1

            self.stats.total_n_chunks += 1

            # Calculate elapsed time
            elapsed_time = time.perf_counter() - start_time

            # Throttle if the bandwidth limit is exceeded
            if bytes_sent / elapsed_time > self.bandwidth:
                if __debug__:
                    # print(f"Bob Bandwidth Limit Sleeping for: {(bytes_sent / self.bandwidth) - elapsed_time}")
                    pass

                # time.sleep((bytes_sent / self.bandwidth) - elapsed_time)
                # self._simulate_time_passing((bytes_sent / self.bandwidth) - elapsed_time)
                time_to_sleep += (bytes_sent / self.bandwidth) - elapsed_time

        end_time = time.perf_counter()

        if __debug__:
            print(f"Chunk Counter (This gives number of chunks in this message: {chunk_counter}")
        total_time_message = (end_time - start_time) + self.propagation_delay
        total_time_message_aux = (len(data) / self.bandwidth) + self.propagation_delay

        # print(f"CALCULATION OF THE ALL TIME TO SEND SINGLE MESSAGE: {(len(data)/self.bandwidth)}")

        # self.stats.total_time_to_send_message += total_time_message
        self.stats.total_time_to_send_message += total_time_message_aux
        self.stats.total_time_to_send_message_math += time_to_sleep + self.propagation_delay
        # self.stats.avg_time_to_send_message += (total_time_message/self.stats.ask_parity_messages)

    #This method simply used when we want to differentiate the usage of bandwidth_limit | Not used at the moment
    def _process_send_with_bandwidth_limit(self, package_data):
        # Method if bandwidth is given to produce the bandwidth limitations
        if self.bandwidth > 0:
            init_pack = b"begin_ask"
            self.aux_socket.sendall(init_pack)
            self.send_with_bandwidth_limit(self.aux_socket, package_data)
            #self.aux_socket.sendall(package_data)
            end_pack = b"end_ask"
            self.aux_socket.sendall(end_pack)
        else:
            init_pack = b"begin_ask"
            self.aux_socket.sendall(init_pack)
            self.aux_socket.sendall(package_data)
            end_pack = b"end_ask"
            self.aux_socket.sendall(end_pack)

    def _service_pending_ask_correct_parity_channel(self):

        if not self._pending_ask_correct_parity:
            return

        # Prepare the question for Alice, i.e. the list of shuffle ranges over which we want Alice
        # to compute the correct parity.
        # This is performed to not pass any objects Between Bob and Alice
        ask_parity_blocks_secure_optimized = []

        # print(f"--------------------------------\nMessage: {self.stats.ask_parity_messages}")
        if __debug__:
            self.logger.debug(f"--------------------------------\nMessage: {self.stats.ask_parity_messages}")
            log_aux_parity_bits_message = self.stats.ask_parity_bits
            log_aux_number_block = 0
            log_aux_ask_parity_bits_info = self.aux_ask_parity_bits_info

        for entry in self._pending_ask_correct_parity:
            (block, _correct_right_sibling) = entry

            # print(f"------------\n{repr(block)}\n{str(block)}\nBlock Nr: {log_aux_number_block}")
            if __debug__:
                self.logger.debug(f"------------\n{repr(block)}\n{str(block)}\nBlock Nr: {log_aux_number_block}")

            block_start_index = block.get_start_index()
            block_end_index = block.get_end_index()
            block_shuffle_object = block.get_shuffle()

            shuffle_index_to_key_index_dict = block_shuffle_object.get_shuffle_index_to_key_index()

            block_secure_obj_opt = (block_start_index, block_end_index, shuffle_index_to_key_index_dict)

            ask_parity_blocks_secure_optimized.append(block_secure_obj_opt)




            # Bits inside a block include: iteration nr | Start Index | End Index
            self.aux_ask_parity_bits_info += 2 + 18 + 18

            aux = self.stats.ask_parity_bits
            if __debug__:
                a = block.get_start_index().bit_length()
                b = block.get_end_index().bit_length()
                self.logger.debug(
                    f"Block Shuffle Identifier: {block.get_shuffle().get_identifier()} ; Identifier Bits: {block.get_shuffle().get_identifier().bit_length()}\nBlock Start Index: {block.get_start_index()} ; Index Bits: {a}\n Block End Index: {block.get_end_index()} ; Index Bits: {b}\n")

                log_aux_number_block += 1
                # print(f"Bits in Block: {self.stats.ask_parity_bits - aux} ; Total Ask Parity Bits: {self.stats.ask_parity_bits}\n------------")
                self.logger.debug(
                    f"Bits in Block: {self.stats.ask_parity_bits - aux} ; Total Ask Parity Bits: {self.stats.ask_parity_bits} ; New Bits in Block: {a + b} ; "
                    f"Total New Ask Parity Bits: {self.aux_ask_parity_bits_info}\n------------")

                log_aux_parity_bits_message = self.stats.ask_parity_bits - log_aux_parity_bits_message
                # print(f"This Message Contains Number of Blocks: {log_aux_number_block} ; Total Bits send in this Message: {log_aux_parity_bits_message}\n--------------------------------")
                self.logger.debug(
                    f"This Message Contains Number of Blocks: {log_aux_number_block} ; Total Bits send in this Message: {log_aux_parity_bits_message} ; Total NEW Bits send in this Message: {self.aux_ask_parity_bits_info - log_aux_ask_parity_bits_info}\n--------------------------------")

        # "Send a message" to Alice to ask her to compute the correct parities for the list that
        # we prepared. For now, this is a synchronous blocking operations (i.e. we block here
        # until we get the answer from Alice).
        self.stats.ask_parity_messages += 1
        self.stats.ask_parity_blocks += len(ask_parity_blocks_secure_optimized)

        # Due to optimised version to not send any Objects
        # package_data = pickle.dumps(ask_parity_blocks_secure)
        package_data = pickle.dumps(ask_parity_blocks_secure_optimized)

        self.stats.ask_parity_bits += (len(package_data) * 8)
        self.stats.ask_parity_bytes += len(package_data)
        # Calculated by n_chunks in a set of blocks, and then divide by number of messages -> Confirmation value is correct by total_n_chunks/number of messages
        self.stats.n_chunks_per_message += (len(package_data) / 4096)

        if __debug__:
            print(f"Stats N Chunks Per Message: {self.stats.n_chunks_per_message}")
            print(f"Stats Ask Parity Blocks: {self.stats.ask_parity_blocks}")
            print(f"Number of Parity Blocks (Len(Parity Blocks)): {len(ask_parity_blocks_secure_optimized)}")
            print(f"N Chunks Per Block: {(len(package_data) / 4096) / len(ask_parity_blocks_secure_optimized)}")
            print(f"Ask Parity Blocks Going to Send... {len(package_data)} Bytes")

        if (len(package_data) * 8) > self.largest_message_sent[0]:
            self.largest_message_sent = (len(package_data) * 8, self.stats.ask_parity_messages)

        # Method if bandwidth is given to produce the bandwidth limitations
        #self.send_with_bandwidth_limit(package_data)

        #Due to bandwidth limit being calculated mathematically send information with sendall
        init_pack = b"begin_ask"
        self.aux_socket.sendall(init_pack)
        self.aux_socket.sendall(package_data)
        end_pack = b"end_ask"
        self.aux_socket.sendall(end_pack)


        # Adding the Time to Send message
        self.stats.total_time_to_send_message += (len(package_data) / self.bandwidth) + self.propagation_delay

        if __debug__:
            print(f"All Parity Blocks Sent: {len(package_data)}")
            print(f"Waiting for Alice Response...")

        package_received, to_tim = self._receive_message_from_alice()

        self.stats.reply_parity_bytes += len(package_received)

        if (len(package_received) * 8) > self.largest_message_received[0]:
            self.largest_message_received = (len(package_received) * 8, self.stats.ask_parity_messages)

        if __debug__:
            print(f"time to receive Reply: {to_tim - self.propagation_delay}")
            print(f"Total Time CORRECT (time receiving message + DELAY): {to_tim}")

        self.total_time += self.propagation_delay

        correct_parities = pickle.loads(package_received)

        if __debug__:
            print(f"Correct parities Received: {correct_parities[:10]}")

        # Process the answer from Alice. IMPORTANT: Alice is required to send the list of parities
        # in the exact same order as the ranges in the question; this allows us to zip.
        # Changes made to account received information: response_array contains start index, end index, and correct parity
        for (response_array, entry) in zip(correct_parities, self._pending_ask_correct_parity):
            self.stats.reply_parity_bits += 1
            (block, correct_right_sibling) = entry
            start_idx, end_idx, correct_parity = response_array
            block.set_correct_parity(correct_parity)
            self._schedule_try_correct(block, correct_right_sibling)

        # Clear the list of pending questions.
        self._pending_ask_correct_parity = []


    def _service_pending_ask_correct_parity_channel_threading(self):

        if __debug__:
            print(f"Service_pending_ask_correct_parity_channel_threading Thread Id: {self.thread_id}")

        if not self._pending_ask_correct_parity:
            return

        # Prepare the question for Alice, i.e. the list of shuffle ranges over which we want Alice
        # to compute the correct parity.
        # This is performed to not pass any objects Between Bob and Alice
        ask_parity_blocks_secure_optimized = []

        # print(f"--------------------------------\nMessage: {self.stats.ask_parity_messages}")
        if __debug__:
            self.logger.debug(f"--------------------------------\nMessage: {self.stats.ask_parity_messages}")
            log_aux_parity_bits_message = self.stats.ask_parity_bits
            log_aux_number_block = 0
            log_aux_ask_parity_bits_info = self.aux_ask_parity_bits_info

        for entry in self._pending_ask_correct_parity:
            (block, _correct_right_sibling) = entry

            block_start_index = block.get_start_index()
            block_end_index = block.get_end_index()
            block_shuffle_object = block.get_shuffle()

            shuffle_index_to_key_index_dict = block_shuffle_object.get_shuffle_index_to_key_index()

            block_secure_obj_opt = (block_start_index, block_end_index, shuffle_index_to_key_index_dict)

            ask_parity_blocks_secure_optimized.append(block_secure_obj_opt)

            # Bits inside a block include: iteration nr | Start Index | End Index
            self.aux_ask_parity_bits_info += 2 + 18 + 18

            aux = self.stats.ask_parity_bits
            if __debug__:
                self.logger.debug(f"------------\n{repr(block)}\n{str(block)}\nBlock Nr: {log_aux_number_block}")

                a = block.get_start_index().bit_length()
                b = block.get_end_index().bit_length()
                self.logger.debug(
                    f"Block Shuffle Identifier: {block.get_shuffle().get_identifier()} ; Identifier Bits: {block.get_shuffle().get_identifier().bit_length()}\nBlock Start Index: {block.get_start_index()} ; Index Bits: {a}\n Block End Index: {block.get_end_index()} ; Index Bits: {b}\n")

                log_aux_number_block += 1
                # print(f"Bits in Block: {self.stats.ask_parity_bits - aux} ; Total Ask Parity Bits: {self.stats.ask_parity_bits}\n------------")
                self.logger.debug(
                    f"Bits in Block: {self.stats.ask_parity_bits - aux} ; Total Ask Parity Bits: {self.stats.ask_parity_bits} ; New Bits in Block: {a + b} ; "
                    f"Total New Ask Parity Bits: {self.aux_ask_parity_bits_info}\n------------")

                log_aux_parity_bits_message = self.stats.ask_parity_bits - log_aux_parity_bits_message
                # print(f"This Message Contains Number of Blocks: {log_aux_number_block} ; Total Bits send in this Message: {log_aux_parity_bits_message}\n--------------------------------")
                self.logger.debug(
                    f"This Message Contains Number of Blocks: {log_aux_number_block} ; Total Bits send in this Message: {log_aux_parity_bits_message} ; Total NEW Bits send in this Message: {self.aux_ask_parity_bits_info - log_aux_ask_parity_bits_info}\n--------------------------------")

        # "Send a message" to Alice to ask her to compute the correct parities for the list that
        # we prepared. For now, this is a synchronous blocking operations (i.e. we block here
        # until we get the answer from Alice).
        self.stats.ask_parity_messages += 1
        self.stats.ask_parity_blocks += len(ask_parity_blocks_secure_optimized)

        # Threading we need to send the Thread id
        data_to_send = (self.thread_id, ask_parity_blocks_secure_optimized)

        package_data = pickle.dumps(data_to_send)

        #self.send_buffer.appendleft(package_data)

        #packet = OutboxPacket(2, package_data)

        #self.role.outbox.put(packet)
        #assert package_data is not None
        #print(f"RECONCILIATION OUTBOX SIZE: {self.role._outbox.qsize()}")

        # Measuring Time Sleep waiting for messages
        self.role.put_in_outbox(package_data)
        #self.messages_sent.append(("B", "S", self.number_messages_sent, time.process_time(), len(package_data), 1, self.thread_id))
        self.messages_sent.append(
            ("B", "S", self.number_messages_sent, time.perf_counter(), len(package_data), 1, self.thread_id))
        self.number_messages_sent += 1

        start_counting = time.perf_counter()
        message_received = self.receive_buffer.get()
        end_counting = time.perf_counter()
        self.stats.total_time_wait_for_receiving_messages += end_counting - start_counting


        #print(f"Envio Mensagem {len(package_data)} | Thread Id: {self.thread_id}")
        #print(f"data_to_send: {data_to_send}")
        #print(f"Message Sent Thread Id: {self.thread_id}")

        self.stats.total_time_to_send_message += (len(package_data) / self.bandwidth) + self.propagation_delay

        self.stats.ask_parity_bits += (len(package_data) * 8)
        self.stats.ask_parity_bytes += len(package_data)
        # Calculated by n_chunks in a set of blocks, and then divide by number of messages -> Confirmation value is correct by total_n_chunks/number of messages
        self.stats.n_chunks_per_message += (len(package_data) / 4096)

        if __debug__:
            print(
                f"Message Added to Buffer Thread ID: {data_to_send[0]} | Len Data ask_parity_blocks_secure_optimized: {len(data_to_send[1])}")
            print(f"Stats N Chunks Per Message: {self.stats.n_chunks_per_message}")
            print(f"Stats Ask Parity Blocks: {self.stats.ask_parity_blocks}")
            print(f"Number of Parity Blocks (Len(Parity Blocks)): {len(ask_parity_blocks_secure_optimized)}")
            print(f"N Chunks Per Block: {(len(package_data) / 4096) / len(ask_parity_blocks_secure_optimized)}")
            print(f"Ask Parity Blocks Going to Send... {len(package_data)} Bytes")
            print(f"Receive Buffer Length: {self.receive_buffer.qsize()}") # Changed from len(self.receive_buffer) to self.receive_buffer.qsize() since Queue has no Len method
            print(f"Pending Ask Correct Parity: {len(self._pending_ask_correct_parity)}")
            print(f"Pending Try Correct: {len(self._pending_try_correct)}")

        if (len(package_data) * 8) > self.largest_message_sent[0]:
            self.largest_message_sent = (len(package_data) * 8, self.stats.ask_parity_messages)

        #while not self.receive_buffer:
            #print(f"Waiting for response: Len Receive_Buffer: {len(self.receive_buffer)} Thread Id: {self.thread_id}")
        #    pass
        #print(f"Received Response: ThreadId: {self.thread_id}")
        #message_received = self.receive_buffer.pop()



        if __debug__:
            print(f"Message Received Thread Id: {self.thread_id}")
            print(f"All Parity Blocks Sent: {len(package_data)}")
            print(f"Waiting for Alice Response...")
            print(f"Correct parities Received: {message_received[:10]}")

        # package_received, to_tim = self._receive_message_from_alice()

        # self.stats.reply_parity_bytes += len(package_received)
        package_received = pickle.dumps((self.thread_id, message_received)) # Performed only to print size, added thread_id since alice sends with thread_id


        #print(f"Message Received: {len(package_received)} | Thread Id: {self.thread_id}")
        #print(f"message_received: {message_received}")
        self.stats.reply_parity_bytes += len(package_received)
        #self.stats.reply_parity_bits += len(package_received) * 8
        #self.messages_received.append(("B", "R", self.number_messages_received, time.process_time(), len(package_received), 4, self.thread_id))
        self.messages_received.append(
            ("B", "R", self.number_messages_received, time.perf_counter(), len(package_received), 4, self.thread_id))
        self.number_messages_received += 1

        self.stats.total_time_to_receive_message += (len(package_received)/self.bandwidth) + self.propagation_delay

        if (len(package_received) * 8) > self.largest_message_received[0]:
            self.largest_message_received = (len(package_received) * 8, self.stats.ask_parity_messages)

        self.total_time += self.propagation_delay

        # Process the answer from Alice. IMPORTANT: Alice is required to send the list of parities
        # in the exact same order as the ranges in the question; this allows us to zip.
        # response_array is used because in threading we passing an array with start_idx, end_idx, and correct_parity
        for (response_array, entry) in zip(message_received, self._pending_ask_correct_parity):
            self.stats.reply_parity_bits += 1
            (block, correct_right_sibling) = entry
            start_idx, end_idx, correct_parity = response_array
            block.set_correct_parity(correct_parity)
            self._schedule_try_correct(block, correct_right_sibling)

        # Clear the list of pending questions.
        self._pending_ask_correct_parity = []

    def _schedule_try_correct(self, block: Block, correct_right_sibling: bool):
        # Push the error block onto the heap. It is pushed as a tuple (block.size, block) to allow
        # us to correct the error blocks in order of shortest blocks first.
        entry = (block, correct_right_sibling)
        heapq.heappush(self._pending_try_correct, (block.get_size(), entry))

    def _have_pending_try_correct(self):
        return self._pending_try_correct != []

    def _service_pending_try_correct(self, cascade: bool):
        # print(f"Service_pending_try_correct Thread Id: {self.thread_id} | Len Pending Try Correct: {len(self._pending_try_correct)}")
        errors_corrected = 0
        while self._pending_try_correct:
            (_, entry) = heapq.heappop(self._pending_try_correct)
            #print(f"Will Try Correct Pending Block {entry}")
            # print(f"Len Pending Try Correct After Pop: {len(self._pending_try_correct)} | Thread Id: {self.thread_id}")
            (block, correct_right_sibling) = entry
            #print(f"Trying to Correct Block: {block}")
            errors_corrected += self._try_correct(block, correct_right_sibling, cascade)
        return errors_corrected

    def _compute_efficiency(self, reconciliation_bits):
        eps = self._estimated_bit_error_rate
        try:
            shannon_efficiency = (-eps * math.log2(eps)) - ((1 - eps) * math.log2(1 - eps))
            key_size = self._noisy_key.get_size()

            ratio_information = 1 - (reconciliation_bits / key_size)
            if __debug__:
                print(f"ratio_information: {ratio_information}")

            # efficiency = reconciliation_bits / (key_size * shannon_efficiency)
            efficiency = (1 - ratio_information) / shannon_efficiency
        except (ValueError, ZeroDivisionError):
            efficiency = None
        return efficiency

    def _all_normal_cascade_iterations(self):
        for iteration_nr in range(1, self._algorithm.cascade_iterations + 1):
            # Iteration bits for initial message with iteration and seed
            self._one_normal_cascade_iteration(iteration_nr)

    def _all_normal_cascade_iterations_channel(self):
        for iteration_nr in range(1, self._algorithm.cascade_iterations + 1):
            # Iteration bits for initial message with iteration and seed
            self._one_normal_cascade_iteration_channel(iteration_nr)

    def _all_normal_cascade_iterations_channel_threading(self):

        for iteration_nr in range(1, self._algorithm.cascade_iterations + 1):
            # Iteration bits for initial message with iteration and seed
            # Same as before mas for threading implementation
            self._one_normal_cascade_iteration_channel_threading(iteration_nr)

        if __debug__:
            print(f"Thread Id: {self.thread_id} Finished All Cascade Processes")

        self.stats.estimated_corrected_bits += self.errors_corrected
        self.stats.corrected_bits_error_iteration.append(self.errors_corrected)

    def _one_normal_cascade_iteration(self, iteration_nr: int):

        if __debug__:
            self.print_message(f"{YELLOW}STARTING NORMAL ITERATION NR: {iteration_nr}{RESET}")

        self.stats.normal_iterations += 1

        # Determine the block size to be used for this iteration, using the rules for this
        # particular algorithm of the Cascade algorithm.
        block_size = self._algorithm.block_size_function(self._estimated_bit_error_rate,
                                                         self._reconciled_key.get_size(),
                                                         iteration_nr)
        if __debug__:
            self.logger.debug(
                f"--------\nITERATION: {self.stats.normal_iterations}\n BLOCK SIZE: {block_size}\n---------------------------")
            self.print_message(f"BLOCK SIZE: {block_size}")

        # In the first iteration, we don't shuffle the key. In all subsequent iterations we
        # shuffle the key, using a different random shuffling in each iteration.
        if iteration_nr == 1:
            shuffle_aux = Shuffle(self._reconciled_key.get_size(), Shuffle.SHUFFLE_KEEP_SAME)
        else:
            shuffle_aux = Shuffle(self._reconciled_key.get_size(), Shuffle.SHUFFLE_RANDOM)

        # Split the shuffled key into blocks, using the block size that we chose.
        blocks = Block.create_covering_blocks(self._reconciled_key, shuffle_aux, block_size)

        # For each top-level covering block...
        for block in blocks:
            # Update the key index to block map.
            self._register_block_key_indexes(block)

            # We won't be able to do anything with the top-level covering blocks until we know what
            # the correct parity it.
            self._schedule_ask_correct_parity(block, False)

        # (self.stats.estimated_corrected_bits) Added PF for a more viable stopping mechanism without checking correct key with 10:20 AM 11/28/2024
        # Service all pending correction attempts (including Cascaded ones) and ask parity
        # messages.
        errors_corrected = self._service_all_pending_work(True)
        self.stats.estimated_corrected_bits += errors_corrected
        self.stats.corrected_bits_error_iteration.append(errors_corrected)

    def _one_normal_cascade_iteration_channel(self, iteration_nr: int):

        self.stats.normal_iterations += 1

        # Determine the block size to be used for this iteration, using the rules for this
        # particular algorithm of the Cascade algorithm.
        block_size = self._algorithm.block_size_function(self._estimated_bit_error_rate,
                                                         self._reconciled_key.get_size(),
                                                         iteration_nr)
        if __debug__:
            self.logger.debug(
                f"--------\nITERATION: {self.stats.normal_iterations}\n BLOCK SIZE: {block_size}\n---------------------------")

        # In the first iteration, we don't shuffle the key. In all subsequent iterations we
        # shuffle the key, using a different random shuffling in each iteration.
        if iteration_nr == 1:
            shuffle_aux = Shuffle(self._reconciled_key.get_size(), Shuffle.SHUFFLE_KEEP_SAME)
        else:
            shuffle_aux = Shuffle(self._reconciled_key.get_size(), Shuffle.SHUFFLE_RANDOM)

        # Split the shuffled key into blocks, using the block size that we chose.
        blocks = Block.create_covering_blocks(self._reconciled_key, shuffle_aux, block_size)

        # For each top-level covering block...
        for block in blocks:
            # Update the key index to block map.
            self._register_block_key_indexes(block)

            # We won't be able to do anything with the top-level covering blocks until we know what
            # the correct parity it.
            self._schedule_ask_correct_parity(block, False)

        # (self.stats.estimated_corrected_bits) Added PF for a more viable stopping mechanism without checking correct key with 10:20 AM 11/28/2024
        # Service all pending correction attempts (including Cascaded ones) and ask parity
        # messages.
        errors_corrected = self._service_all_pending_work_channel(True)
        self.stats.estimated_corrected_bits += errors_corrected
        self.stats.corrected_bits_error_iteration.append(errors_corrected)

    def _one_normal_cascade_iteration_channel_threading(self, iteration_nr: int):

        self.stats.normal_iterations += 1

        # Determine the block size to be used for this iteration, using the rules for this
        # particular algorithm of the Cascade algorithm.
        block_size = self._algorithm.block_size_function(self._estimated_bit_error_rate,
                                                         self._reconciled_key.get_size(),
                                                         iteration_nr)
        if __debug__:
            self.logger.debug(
                f"--------\nITERATION: {self.stats.normal_iterations}\n BLOCK SIZE: {block_size}\n---------------------------")

        if __debug__:
            print(
                f"--------\nITERATION: {self.stats.normal_iterations}\n BLOCK SIZE: {block_size} THREAD ID: {self.thread_id}\n---------------------------\n")
        # print(f"Len OF Send Buffer: {len(self.send_buffer)}")

        # In the first iteration, we don't shuffle the key. In all subsequent iterations we
        # shuffle the key, using a different random shuffling in each iteration.
        if iteration_nr == 1:
            shuffle_aux = Shuffle(self._reconciled_key.get_size(), Shuffle.SHUFFLE_KEEP_SAME)
        else:
            shuffle_aux = Shuffle(self._reconciled_key.get_size(), Shuffle.SHUFFLE_RANDOM)

        # Split the shuffled key into blocks, using the block size that we chose.
        blocks = Block.create_covering_blocks(self._reconciled_key, shuffle_aux, block_size)

        # For each top-level covering block...
        for block in blocks:
            # Update the key index to block map.
            self._register_block_key_indexes(block)

            # We won't be able to do anything with the top-level covering blocks until we know what
            # the correct parity it.
            self._schedule_ask_correct_parity(block, False)

        # self.process_parity_messages()
        # (self.stats.estimated_corrected_bits) Added PF for a more viable stopping mechanism without checking correct key with 10:20 AM 11/28/2024
        # Service all pending correction attempts (including Cascaded ones) and ask parity
        # messages.
        errors_corrected = self._service_all_pending_work_channel_threading(True)
        self.stats.estimated_corrected_bits += errors_corrected
        self.stats.corrected_bits_error_iteration.append(errors_corrected)


    def _service_all_pending_work(self, cascade: bool):

        # Keep track of how many errors were actually corrected in this call.
        errors_corrected = 0

        # Keep going while there is more work to do.
        while self._have_pending_try_correct() or self._have_pending_ask_correct_parity():
            # Attempt to correct all of blocks that are currently pending as needing a correction
            # attempt. If we don't know the correct parity of the block, we won't be able to finish
            # the attempted correction yet. In that case the block will end up on the "pending ask
            # parity" list.
            errors_corrected += self._service_pending_try_correct(cascade)

            # Now, ask Alice for the correct parity of the blocks that ended up on the "ask parity
            # list" in the above loop. When we get the answer from Alice, we may discover that the
            # block as an odd number of errors, in which case we add it back to the "pending error
            # block" priority queue.
            #self._service_pending_ask_correct_parity()
            self._service_pending_ask_correct_parity_secure_optimized()

        return errors_corrected

    def _service_all_pending_work_channel(self, cascade: bool):

        # Keep track of how many errors were actually corrected in this call.
        errors_corrected = 0

        # Keep going while there is more work to do.
        while self._have_pending_try_correct() or self._have_pending_ask_correct_parity():
            # Attempt to correct all blocks that are currently pending as needing a correction
            # attempt. If we don't know the correct parity of the block, we won't be able to finish
            # the attempted correction yet. In that case the block will end up on the "pending ask
            # parity" list.
            errors_corrected += self._service_pending_try_correct(cascade)

            # Now, ask Alice for the correct parity of the blocks that ended up on the "ask parity
            # list" in the above loop. When we get the answer from Alice, we may discover that the
            # block as an odd number of errors, in which case we add it back to the "pending error
            # block" priority queue.
            self._service_pending_ask_correct_parity_channel()

        return errors_corrected

    def _service_all_pending_work_channel_threading(self, cascade: bool):

        # Keep track of how many errors were actually corrected in this call.
        errors_corrected = 0

        # Keep going while there is more work to do.
        while self._have_pending_try_correct() or self._have_pending_ask_correct_parity():
            # Attempt to correct all blocks that are currently pending as needing a correction
            # attempt. If we don't know the correct parity of the block, we won't be able to finish
            # the attempted correction yet. In that case the block will end up on the "pending ask
            # parity" list.

            errors_corrected += self._service_pending_try_correct(cascade)

            # Now, ask Alice for the correct parity of the blocks that ended up on the "ask parity
            # list" in the above loop. When we get the answer from Alice, we may discover that the
            # block as an odd number of errors, in which case we add it back to the "pending error
            # block" priority queue.
            self._service_pending_ask_correct_parity_channel_threading()

        return errors_corrected

    def _all_biconf_iterations(self):

        # Do nothing if BICONF is disabled.
        if not self._algorithm.biconf_iterations:
            return

        # If we are not cascading during BICONF, clear the key indexes to blocks map to avoid
        # wasting time keeping it up to date as correct blocks during the BICONF phase.
        if not self._algorithm.biconf_cascade:
            self._key_index_to_blocks = {}

        # Do the required number of BICONF iterations, as determined by the protocol.
        iterations_to_go = self._algorithm.biconf_iterations
        while iterations_to_go > 0:
            errors_corrected = self._one_biconf_iteration()
            if self._algorithm.biconf_error_free_streak and errors_corrected > 0:
                iterations_to_go = self._algorithm.biconf_iterations
            else:
                iterations_to_go -= 1

    def _one_biconf_iteration(self):

        self.stats.biconf_iterations += 1

        cascade = self._algorithm.biconf_cascade

        # Randomly select half of the bits in the key. This is exactly the same as doing a new
        # random shuffle of the key and selecting the first half of newly shuffled key.
        key_size = self._reconciled_key.get_size()
        shuffle = Shuffle(key_size, Shuffle.SHUFFLE_RANDOM)
        mid_index = key_size // 2
        chosen_block = Block(self._reconciled_key, shuffle, 0, mid_index, None)
        if cascade:
            self._register_block_key_indexes(chosen_block)

        # Ask Alice what the correct parity of the chosen block is.
        self._schedule_ask_correct_parity(chosen_block, False)

        # If the algorithm wants it, also ask Alice what the correct parity of the complementary
        # block is.
        if self._algorithm.biconf_correct_complement:
            complement_block = Block(self._reconciled_key, shuffle, mid_index, key_size, None)
            if cascade:
                self._register_block_key_indexes(complement_block)
            self._schedule_ask_correct_parity(complement_block, False)

        # Service all pending correction attempts (potentially including Cascaded ones) and ask
        # parity messages.
        errors_corrected = self._service_all_pending_work(cascade)
        return errors_corrected

    def _try_correct(self, block: Block, correct_right_sibling: bool, cascade: bool):

        if __debug__:
            """
            print(
                f"\nBlock try to correct | Start Index: {block.get_start_index()} | End Index: {block.get_end_index()}"
                f" | Left Sub Block: {block.get_left_sub_block()} | "
                f"Right Sub Block: {block.get_right_sub_block()} | Current Parity: {block.get_current_parity()} | "
                f"Correct Parity: {block.get_correct_parity()}: {_format_block_key(block, self._classical_channel._correct_key, block.get_start_index(), block.get_end_index())}",
                file=self.file_logging)
            """
            try:
                self.print_message(f"\nBlock try to correct | Start Index: {YELLOW}{block.get_start_index()}{RESET} | End Index: {YELLOW}{block.get_end_index()}{RESET}"
                    f" | Left Sub Block: {block.get_left_sub_block()} | "
                    f"Right Sub Block: {block.get_right_sub_block()} | Current Parity: {CYAN}{block.get_current_parity()}{RESET} | "
                    f"Correct Parity: {CYAN}{block.get_correct_parity()}{RESET}: {_format_block_key(block, self._classical_channel._correct_key, block.get_start_index(), block.get_end_index())} | {_format_block_key_wo_shuffle(block, self._classical_channel._correct_key, block.get_start_index(), block.get_end_index(), block.get_shuffle())}")

                parent_block = block.get_parent_block()

                if parent_block is None:
                    print("This block is Top Level", file=self.file_logging)
                else:
                    print(
                        f"Parent Block: {_format_block_key(parent_block, self._classical_channel._correct_key, parent_block.get_start_index(), parent_block.get_end_index())} | {_format_block_key_wo_shuffle(parent_block, self._classical_channel._correct_key, parent_block.get_start_index(), parent_block.get_end_index(), parent_block.get_shuffle())}",
                        file=self.file_logging)
            except AttributeError:
                #In Case Classical Channel is None continue without logging - Performed this way to reuse in case of logging implementation
                pass

        #print(f"TRYING TO CORRECT BLOCK: {block}")

        # If we don't know the correct parity of the block, we cannot make progress on this block
        # until Alice has told us what the correct parity is.
        if not self._correct_parity_is_known_or_can_be_inferred(block):
            #print(f"Exit cannot know correct parity or infer it | Block: {block.get_start_index()}:{block.get_end_index()}")

            if __debug__:
                self.print_message(
                    f"Not possible to Infer this block parity. {GREEN}Scheduling for asking Alice the Correct Parity and trying to correct later.{RESET}")
            self._schedule_ask_correct_parity(block, correct_right_sibling)
            return 0


        #print(f"I AM TRYING TO CORRECT BLOCK: {block}")

        # If there is an even number of errors in this block, we don't attempt to fix any errors
        # in this block. But if asked to do so, we will attempt to fix an error in the right
        # sibling block.
        if block.get_error_parity() == Block.ERRORS_EVEN:
            #print("Exist Block Errors Even")
            if __debug__:
                self.print_message(f"{MAGENTA}Even Number of Errors{RESET} (Or None)")
            if correct_right_sibling:
                return self._try_correct_right_sibling_block(block, cascade)
            return 0

        # If this block contains a single bit, we have finished the recursion and found an error.
        # Correct the error by flipping the key bit that corresponds to this block.
        if block.get_size() == 1:
            if __debug__:
                self.print_message(f"This Block contains a Single Bit and an Error was found: {GREEN}Flip Bit!{RESET}")
            self._flip_key_bit_corresponding_to_single_bit_block(block, cascade)
            return 1

        # If we get here, it means that there is an odd number of errors in this block and that
        # the block is bigger than 1 bit.
        if __debug__:
            self.print_message(
                f"Block has an Odd number of errors and is bigger than one bit. {GREEN}Dividing block and Trying to correct error{RESET}")

        # Recurse to try to correct an error in the left sub-block first, and if there is no error
        # there, in the right sub-block alternatively.
        left_sub_block = block.get_left_sub_block()
        if left_sub_block is None:
            left_sub_block = block.create_left_sub_block()
            self._register_block_key_indexes(left_sub_block)
        return self._try_correct(left_sub_block, True, cascade)

    def _try_correct_right_sibling_block(self, block: Block, cascade: bool):
        parent_block = block.get_parent_block()
        right_sibling_block = parent_block.get_right_sub_block()
        if right_sibling_block is None:
            right_sibling_block = parent_block.create_right_sub_block()
            self._register_block_key_indexes(right_sibling_block)
        if __debug__:
            self.print_message(f"{MAGENTA}Trying to correct Right Sibling Block:{RESET} {right_sibling_block}")
        return self._try_correct(right_sibling_block, False, cascade)


    def _flip_key_bit_corresponding_to_single_bit_block(self, block: Block, cascade: bool):

        flipped_shuffle_index = block.get_start_index()
        block.flip_bit(flipped_shuffle_index)

        # For every block that covers the key bit that was corrected...
        flipped_key_index = block.get_key_index(flipped_shuffle_index)

        for affected_block in self._get_blocks_containing_key_index(flipped_key_index):

            # Flip the parity of that block.
            affected_block.flip_parity()

            # If asked to do cascading, do so for blocks with an odd number of errors.
            if cascade and affected_block.get_error_parity() != Block.ERRORS_EVEN:
                # If sub_block_reuse is disabled, then only cascade top-level blocks.
                if self._algorithm.sub_block_reuse or affected_block.is_top_block():
                    if __debug__:
                        try:
                            self.print_message(
                                f"Scheduling Try Correct Affected Block from Flip Bit Due to CASCADE: {CYAN}{affected_block}{RESET}")
                            self.print_message(
                                f"Scheduling Try Correct Affected Block from Flip Bit Due to CASCADE: {_format_block_key(affected_block, self._classical_channel._correct_key, affected_block.get_start_index(), affected_block.get_end_index())} | {_format_block_key_wo_shuffle(affected_block, self._classical_channel._correct_key, affected_block.get_start_index(), affected_block.get_end_index(), affected_block.get_shuffle())} | Indexes: {YELLOW}{affected_block.get_start_index()}{RESET}:{YELLOW}{affected_block.get_end_index()}{RESET}")

                            #print(f"Scheduling Try Correct Affected Block from Flip Bit Due to CASCADE: CYAN{affected_block}RESET", file=self.file_logging)
                            self.logger.debug(f"Scheduling Try Correct Affected Block from Flip Bit Due to CASCADE: {affected_block}")
                        except AttributeError:
                            # In Case Classical Channel is None continue without logging - Performed this way to reuse in case of logging implementation
                            pass
                    # Test if correct_right_sibling shouldnt be true -- Answer No; correct_right_sibling - True will create None Exception
                    #print(f"Will PERFORM CASCADE EFFECT - affected block: {affected_block}")
                    #print(f"Affected Block: {affected_block}")
                    self._schedule_try_correct(affected_block, False)


