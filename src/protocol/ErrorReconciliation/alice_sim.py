import queue
import socket
import pickle
import sys
import time
from os import access


from ..PrivacyAmplification.privacy_amplification import PrivacyAmplification
#from multiprocessing.connection import Connection

#from matplotlib.backends.backend_nbagg import connection_info

#from classical_communication_channel.communication_channel.channel_manager import ChannelManager
from ..classical_communication_channel.communication_channel.connection_info import ConnectionInfo
#from classical_communication_channel.communication_channel.common import generate_connection_info
#from classical_communication_channel.communication_channel.participant import Participant
from ..classical_communication_channel.communication_channel.role import Role
#from classical_communication_channel.communication_channel.classical_communication_packet import OutboxPacket
from ..classical_communication_channel.communication_channel.mac_config import MAC_Config, MAC_Algorithms


#from cascade.algorithm import ALGORITHMS
from .cascade.key import Key
#from cascade.mock_classical_channel import MockClassicalChannel
#import threading
from collections import deque
import threading

#from cascade.reconciliation import Reconciliation

#GEO_DELAY = 0.238
#LEO_DELAY = 0.0427
#MEO_DELAY = 0.100

#This Class defines all the managing required to the Alice. Multiple threading and key exchanging in communication
class AliceManager:
    def __init__(self, alice_ip="127.0.0.2", alice_port=5002, artificial_alice_ip="127.0.0.4", artificial_alice_port=4994, bob_ip = "127.0.0.1", bob_port=5001):
        # Alice's configuration
        self.alice_ip = alice_ip
        self.alice_port = alice_port

        self.artificial_alice_ip = artificial_alice_ip
        self.artificial_alice_port = artificial_alice_port

        # Bob's configuration
        self.bob_ip = bob_ip
        self.bob_port = bob_port  # Bob's listening port

        # Create a TCP socket for Alice
        self.alice_socket_aux = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.alice_socket_aux.settimeout(30)

        self.alice_socket_aux.bind((artificial_alice_ip, artificial_alice_port))  # Bind to Alice's IP and port

        self.alice_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.alice_socket.bind((alice_ip, alice_port))

        self.connection = None
        self.address = None

        self.connection_w_bob = None
        self.address_w_bob = None

        self.correct_key = None
        self.propagation_delay = None
        self.bandwidth = None

        #For Threading
        self.key_divisions_bb84 = None
        self.dict_key_divisions_bb84 = None

        self.alice_obj = None

        self.compression_rate = None


    def establish_connection(self):

        try:
            self.alice_socket_aux.listen(1)

            print("Alice is listening for connections...")

            self.connection, self.address = self.alice_socket_aux.accept()

            print("Alice Connected for Key Exchange")

            print("Alice is listening for the connection with Bob")
            self.alice_socket.listen(5)

            self.connection_w_bob, self.address_w_bob = self.alice_socket.accept()

            print("Alice Connected with Bob")
        except Exception as e:
            print(f"Some problem occurred when trying to connect: {e}")
            self.alice_socket.close()
            self.alice_socket_aux.close()
            return None

        return self.connection, self.address, self.connection_w_bob, self.address_w_bob

    def exchange_initial_key(self):

        if __debug__:
            print("---------------NEW RUN------------------")

        print("Alice waiting for Key from Bob")

        # Listen for incoming messages and allow sending in a loop
        data = self.receive_key_from_bob(self.connection)

        # Deserialize the received data
        try:
            deserialized_data = pickle.loads(data)
            if isinstance(deserialized_data, list) and len(deserialized_data) == 4:
                self.correct_key, self.propagation_delay, self.bandwidth, self.compression_rate = deserialized_data
                #self.bandwidth = self.bandwidth * 0.000001
                print(
                    f"Received correct key and propagation delay and bandwidth and compression rate:\nKey: {self.correct_key}\nPropagation Delay: {self.propagation_delay}\nBandwidth: {self.bandwidth}\nCompression Rate: {self.compression_rate}")
            else:
                self.correct_key = deserialized_data
                print(f"Received correct key:\nKey: {self.correct_key}")
        except Exception as e_initial_key:
            print(f"Error while deserializing data: {e_initial_key}")
            self.alice_socket.close()
            self.alice_socket_aux.close()
            exit()


        # only first 10 bits of the key for better visibility
        print("Received Correct Key", self.correct_key.get_size())


        #self.alice_obj = Alice(0, self.correct_key)


    def process_set_of_runs(self):

        while True:

            self.exchange_initial_key()

            self.alice_obj = Alice(0, self.correct_key,compression_rate=self.compression_rate)

            try:
                self.process_receiving_message_calculate_parity_and_reply()
                self.alice_obj.process_privacy_amplification()
            except Exception as e_run:
                print(f"Alice: Error occurred in Run - {e_run}")
            finally:
                # Close the connection and server sockets
                print("Alice: Finishing this Set of Runs")




    def close_sockets(self):
        self.alice_socket.close()
        self.alice_socket_aux.close()
        self.connection.close()
        self.connection_w_bob.close()


    #Single Thread
    def process_receiving_message_calculate_parity_and_reply(self):

        total_time = 0
        total_time_sleep = 0

        while True:
            message_received, to_tim = self.receive_message_from_bob(self.connection_w_bob, self.propagation_delay)
            # threading.Thread(target=thread_receiving_messages, daemon=True,)
            aux = to_tim - self.propagation_delay
            total_time += to_tim

            total_time_sleep += self.propagation_delay

            if __debug__:
                print(f"Total Time: {total_time}")
                print(f"Time receiving message: {aux}")
                print(f"Total Time CORRECT (time receiving message + DELAY: {to_tim}")
                print(f"Total Time Sleep: {total_time_sleep}")

            if __debug__:
                print(f"Blocks Received (Bytes): {len(message_received)}")

            if b"end_reconciliation_final" in message_received:
                print(f"End Reconciliation message received")
                # end_reconciliation_final_array.append(thread_id)
                if 'connection' in locals():
                    self.connection.close()

                self.connection_w_bob.close()
                self.alice_socket.close()
                self.alice_socket_aux.close()
                exit()

            if b"end_reconciliation" in message_received:
                print(f"End Reconciliation message received")
                # end_reconciliation_array.append(thread_id)
                break

            # Process Message Blocks

            if message_received:
                #thread_id, ask_parity_blocks = pickle.loads(message_received)

                # NON-THREADING
                ask_parity_blocks = pickle.loads(message_received)

                parities = self.alice_obj.ask_parities_secure_optimized(ask_parity_blocks)

                if __debug__:
                    print(f"Parities Computed: {parities[:10]}")

                serialized_correct_parities = pickle.dumps(parities)

                if __debug__:
                    print(f"Size of the serialized correct parities (Bytes): {len(serialized_correct_parities)}")

                self.send_message_to_bob(serialized_correct_parities)
            else:
                print(f"Message Received is Empty: {len(message_received)}")



    def send_message_to_bob(self, message):

        begin_reply = b"begin_reply"
        self.connection_w_bob.sendall(begin_reply)
        if self.bandwidth > 0:
            #send_with_bandwidth_limit(self.connection_w_bob, message, self.bandwidth)
            self.connection_w_bob.sendall(message)
            # connection_w_bob.sendall(serialized_correct_parities)
        else:
            self.connection_w_bob.sendall(message)
        end_reply = b"end_reply"
        self.connection_w_bob.sendall(end_reply)

        if __debug__:
            print(f"Correct parities finished sending")


    @staticmethod
    def receive_key_from_bob(connections):
        data_mss = b""

        print("Waiting receiving KEY from bob")

        in_message = False
        connections.settimeout(5)
        try:
            while True:
                chunk_mss = connections.recv(4096)  # Receive data in chunks of 4KB

                if b"begin_key" in chunk_mss:
                    print("initiating receiving KEY")
                    in_message = True
                    chunk_mss = chunk_mss.split(b"begin_key", 1)[1]  # Strip the marker

                if b"end_key" in chunk_mss:  # If no more data is sent, break the loop
                    print("end receiving KEY")
                    in_message = False
                    chunk_mss, _ = chunk_mss.split(b"end_key", 1)  # Strip the marker
                    data_mss += chunk_mss
                    break

                if in_message:
                    data_mss += chunk_mss

        except socket.timeout:
            print("Timeout reached, no data received")
        except Exception as e:
            print(f"Alice: Error occurred When Receiving Key - {e}")

        return data_mss

    @staticmethod
    def receive_message_from_bob(connections, propagation_delay_aux):
        data_mss = b""

        print("Waiting receiving message from bob")

        # Listen for incoming messages and allow sending in a loop
        start_time = 0
        end_time = 0
        aux2 = 0

        buffer = b""

        in_message = False
        try:
            while True:


                chunk_mss = connections.recv(4096)  # Receive data in chunks of 4KB

                if not chunk_mss:
                    print("Testing breaking condition")
                    break

                if len(chunk_mss) == 0:
                    # Connection closed or no data received
                    print("No more data received. Exiting loop.")
                    break

                buffer += chunk_mss

                if not in_message:
                    if b"begin_ask" in buffer:
                        print("initiating receiving message")
                        in_message = True
                        start_time = time.perf_counter()
                        _, buffer = buffer.split(b"begin_ask", 1)  # Remove the `begin_ask` delimiter
                        # data_mss += buffer.split(b"begin_ask", 1)[1]

                if in_message:
                    if b"end_ask" in (data_mss[-4096:] + buffer):
                        print("end receiving message")
                        in_message = False
                        end_time = time.perf_counter()
                        data_mss += buffer.split(b"end_ask", 1)[0]  # Extract data before `end_ask`
                        break

                    # Accumulate the data
                    data_mss += buffer
                    buffer = b""  # Clear the buffer after processing


        except socket.timeout:
            print("Timeout reached, no data received")
        except BlockingIOError:
            # Handle the case where no data is available
            print("No data available yet, try again later")

        if start_time != 0:
            aux2 = (end_time - start_time) + propagation_delay_aux
            if __debug__:
                print(
                    f"total time sleep receiving message: Start: {start_time} | End: {end_time} | Time in seconds {aux2} seconds")
            # time.sleep((end_time - start_time) * 1.238)
            # aux = (end_time - start_time) + GEO_DELAY

        return data_mss, aux2

    @staticmethod
    def produce_bb84_key_division(key_param, data_packet_size=5000):
        key_size = key_param.get_size()

        thread_id_aux = 0

        key_divisions_bb84_aux = []

        dict_key_divisions_bb84_aux = {}

        # Assert data_packet_size < key_size
        for i in range(0, key_size, data_packet_size):

            aux_div_key = {}

            for j in range(data_packet_size):

                if i + j >= key_size:
                    break

                noisy_key_bit = key_param.get_bit(i + j)

                aux_div_key[j] = noisy_key_bit

            key_divisions_bb84_aux.append((thread_id_aux, Key(size=len(aux_div_key), bits=aux_div_key)))
            dict_key_divisions_bb84_aux[thread_id_aux] = Key(size=len(aux_div_key), bits=aux_div_key)

            thread_id_aux += 1

        return key_divisions_bb84_aux, dict_key_divisions_bb84_aux


class AliceManagerChannel(AliceManager):
    def __init__(self):
        super().__init__()

        self.participant_alice = None

        self.continue_thread_run = True

        self.alice_array = []

        self.compression_rate = None

        self.join_key = []


    #This method should be run only after exchange_initial_bandwidth
    def create_classical_channel_link(self):
        access_info_participants = ConnectionInfo("127.0.0.23", 5020)

        #bandwidth_limit = round(float(self.input_vars["Transfer Rate (Mbps)"].get()), 2)
        bandwidth_limit = self.bandwidth

        print(f"BANDWIDTH_LIMIT: {bandwidth_limit}")

        if bandwidth_limit <= 0:
            bandwidth_limit = 5000

        #Conversion from Mbps to MBps for channel usage
        bandwidth_limit = int(bandwidth_limit * 0.000001)

        print(f"BANDWIDTH_LIMIT: {bandwidth_limit}")

        testing_with_mac = False

        if testing_with_mac:
            shared_key = b'IzetXlgAnY4oye56'  # This is an example. Must be 16-bits long.
            mac_configuration = MAC_Config(MAC_Algorithms.CMAC, shared_key)  # Example using Cipher-based message
            #participant = Participant.get_participant(access_info_participants, 2, bandwidth_limit, mac_configuration)
            role = Role.get_instance(access_info_participants, bandwidth_limit, bandwidth_limit, bandwidth_limit, mac_configuration)
            #participant = Participant.get_participant(access_info_participants, 2, 15, mac_configuration)
        else:
            role = Role.get_instance(access_info_participants)
            #participant = Participant.get_participant(access_info_participants, 2, bandwidth_limit)
            #participant = Participant.get_participant(access_info_participants, 2, 15)


        #while role.id != 2:
        #    print("Sleeping for 5 seconds...")
        #    time.sleep(5)

        self.participant_alice = role

        return access_info_participants, role

    def exchange_initial_bandwidth(self):
        print("Alice waiting for Bandwidth from Bob")

        data = self.receive_key_from_bob(self.connection)

        # Deserialize the received data
        try:
            deserialized_data = pickle.loads(data)
            self.bandwidth, self.compression_rate = deserialized_data
            print(f"Received correct bandwidth:\nBandwidth: {self.bandwidth}")
            print(f"Received correct Compression Rate:\nCompression Rate: {self.compression_rate}")
        except Exception as e_initial_key:
            print(f"Error while deserializing data: {e_initial_key}")
            self.alice_socket.close()
            self.alice_socket_aux.close()
            exit()

    def create_alice_array(self, n_threads):

        self.alice_array = []

        self.exchange_initial_key()

        data_packet_size = int(self.correct_key.get_size()/n_threads)

        key_divisions_bb84_aux, dict_key_divisions_bb84_aux = self.produce_bb84_key_division(self.correct_key, data_packet_size)

        for thread_idx, key in key_divisions_bb84_aux:
            self.alice_array.append(AliceChannel(thread_idx, key, self.participant_alice, self.compression_rate))


    def process_receiving_inbox(self):
        while self.continue_thread_run:
            try:
                #message_received = self.participant_alice.inbox.get(False)
                message_received = self.participant_alice.get_from_inbox()

                thread_idx, ask_parity_blocks = pickle.loads(message_received)

                print(f"Mensagem RECEBIDA: {len(message_received)}")

                self.alice_array[thread_idx].add_receiving_information(ask_parity_blocks)
            except queue.Empty as e:
                continue


    def process_threading_reconciliation(self):

        self.continue_thread_run = True



        thread_receiving = threading.Thread(target=self.process_receiving_inbox, daemon=True)
        thread_receiving.start()
        #thread_sending = threading.Thread(target=self.send_message_to_bob_thread, daemon=True).start()

        thread_array = []

        for alice in self.alice_array:
            thread_alice = threading.Thread(target=alice.process_compute_parities_threading)
            thread_alice.start()
            thread_array.append(thread_alice)

        for thr in thread_array:
            thr.join()

        self.continue_thread_run = False
        thread_receiving.join()


        joined_key = self.join_keys()

        print(f"Joined Key:\n {joined_key}")

        secured_key = self._process_privacy_amplification_final_key(joined_key)






    def join_keys(self):
        joined_key = []

        for alice in self.alice_array:
            aux_key = alice.correct_key.generate_array()
            joined_key.extend(aux_key)

        return joined_key


    def process_set_of_runs_thread(self, n_threads):

        while True:

            self.create_alice_array(n_threads)

            self.process_threading_reconciliation()


    def _process_privacy_amplification_final_key(self, list_key_bits):
        print(f"Starting Privacy Amplification Alice for JOINED KEY")

        privacy_amplification_object = PrivacyAmplification(list_key_bits)

        initial_key_length = len(list_key_bits)
        #compression_rate = 0.8
        compression_rate = self.compression_rate

        final_key_length = int(initial_key_length * compression_rate)

        print(f"Initial_key_length: {initial_key_length}")
        print(f"Final_key_length: {final_key_length}")

        _, _, secured_key = privacy_amplification_object.do_privacy_amplification(initial_key_length, final_key_length)

        print(f"Initial Correct Key Size: {initial_key_length}")
        print(f"Correct Key: {self.correct_key}")
        print(f"JOIN KEY: {list_key_bits}")
        print(f"Final Secured Key Size: {len(secured_key)}")
        #print(f"Final Key: {secured_key}")
        print(f"SECURED KEY:\n{secured_key.tolist()}\n-----------\n")

        print(f"FINISHED Privacy Amplification Alice")

        return secured_key






class Alice:
    def __init__(self, thread_id_alice, correct_key_alice, send_buffer_alice=None, lock=None, compression_rate=None, message_queue=None):

        self.thread_id = thread_id_alice
        self.correct_key = correct_key_alice
        self.send_buffer = send_buffer_alice
        self.lock = lock

        #self.receive_buffer = deque()
        self.receive_buffer = message_queue

        self.compression_rate = compression_rate

    def send_message_to_bob(self, data_to_send):
    #with self.lock:
        print(f"Alice Thread Id: {self.thread_id} adding information to Send Buffer")
        data = pickle.dumps((self.thread_id,data_to_send))
        self.send_buffer.appendleft(data)

    def add_receiving_information(self, data_to_receive):
        self.receive_buffer.appendleft(data_to_receive)

    def ask_parities_secure_optimized(self, blocks):
        parities_aux = []
        for block in blocks:
            start_index, end_index, shuffle_index_to_key_index_dict = block
            parity = self.calculate_parity_alice(self.correct_key, start_index, end_index, shuffle_index_to_key_index_dict)
            parities_aux.append((start_index, end_index, parity))

        return parities_aux

    def process_compute_parities_threading(self):

        to_compute = True

        while to_compute:

            if self.receive_buffer:
                blocks_to_calculate_parity = self.receive_buffer.pop()

                if "end_reconciliation" in blocks_to_calculate_parity:
                    to_compute = False
                    print(f"End Reconciliation Alice Thread Id: {self.thread_id}")
                    #Process Privacy Amplification

                    break

                parities_calculated = self.ask_parities_secure_optimized(blocks_to_calculate_parity)

                self.send_message_to_bob(parities_calculated)





    # Due to not sending entire Objects, Alice performs the calculation given the dictionary not needing the Mock Classical Channel Class
    @staticmethod
    def calculate_parity_alice(key_param, shuffle_start_index, shuffle_end_index,
                               shuffle_index_to_key_index_dictionary):
        parity = 0
        for shuffle_index in range(shuffle_start_index, shuffle_end_index):
            key_index = shuffle_index_to_key_index_dictionary[shuffle_index]
            if key_param.get_bit(key_index):
                parity = 1 - parity

        return parity


    def process_privacy_amplification(self):
        print(f"Starting Privacy Amplification Alice for Thread Id: {self.thread_id}")

        list_key_bits = list(self.correct_key._bits.values())

        privacy_amplification_object = PrivacyAmplification(list_key_bits)

        initial_key_length = self.correct_key.get_size()
        #compression_rate = 0.8
        compression_rate = self.compression_rate

        final_key_length = int(initial_key_length * compression_rate)

        _, _, secured_key = privacy_amplification_object.do_privacy_amplification(initial_key_length, final_key_length)

        print(f"Initial Correct Key Size: {initial_key_length}")
        print(f"Correct Key: {self.correct_key}")
        print(f"Final Secured Key Size: {len(secured_key)}")
        #print(f"Final Key: {secured_key}")
        print(f"THREAD ID: {self.thread_id} | SECURED KEY:\n{secured_key.tolist()}\n-----------\n")

        print(f"FINISHED Privacy Amplification Alice for Thread Id: {self.thread_id}")


class AliceChannel(Alice):

    def __init__(self,thread_id_alice, correct_key_alice, participant, compression_rate, message_queue=None):
        super().__init__(thread_id_alice, correct_key_alice, message_queue=message_queue)
        self.participant = participant
        #print(f"Participant in Alice Creation: {participant}")
        self.compression_rate = compression_rate

        self.messages_sent = []
        self.number_messages_sent = 0
        self.messages_received = []
        self.number_messages_received = 0

    def send_message_to_bob(self, data_to_send):
        #packet = OutboxPacket(1, data_to_send)

        #self.participant.outbox.put(packet)

        self.participant.put_in_outbox(data_to_send)
        #self.messages_sent.append(("A", "S", self.number_messages_sent, time.process_time(), len(data_to_send), 3, self.thread_id))
        self.messages_sent.append(
            ("A", "S", self.number_messages_sent, time.perf_counter(), len(data_to_send), 3, self.thread_id))
        self.number_messages_sent += 1


    def process_compute_parities_threading(self):

        to_compute = True

        print(f"Alice Thread: {self.thread_id} Is waiting for information to computing parities")

        while to_compute:

            if self.receive_buffer:
                #blocks_to_calculate_parity = self.receive_buffer.pop()
                blocks_to_calculate_parity = self.receive_buffer.get()

                print(f"Received Block: {len(blocks_to_calculate_parity)}")
                #print(f"Received Message: {blocks_to_calculate_parity}")

                aux_payload = pickle.dumps((self.thread_id, blocks_to_calculate_parity)) # Performed just to see messages size
                print(f"Alice Received Message: {len(aux_payload)} | ThreadId: {self.thread_id}")

                #self.messages_received.append(("A", "R", self.number_messages_received, time.process_time(), len(aux_payload), 2, self.thread_id))
                self.messages_received.append(
                    ("A", "R", self.number_messages_received, time.perf_counter(), len(aux_payload), 2, self.thread_id))
                self.number_messages_received += 1


                if "end_reconciliation" in blocks_to_calculate_parity:
                    print(f"End Reconciliation Received ThreadID: {self.thread_id}")
                    to_compute = False
                    #if self.compression_rate > 0.0:
                    #    self._process_privacy_amplification()
                    break

                parities_calculated = self.ask_parities_secure_optimized(blocks_to_calculate_parity)

                send_data = pickle.dumps((self.thread_id, parities_calculated))
                print(f"Alice Sent Message to Bob {len(send_data)} | Thread Id: {self.thread_id}")
                #print(f"data_to_send: {(self.thread_id, parities_calculated)}")

                self.send_message_to_bob(send_data)

        print("Exit Alice Channel Correct Parities Loop")


    def _process_privacy_amplification(self):
        print(f"Starting Privacy Amplification Alice for Thread Id: {self.thread_id}")

        list_key_bits = list(self.correct_key._bits.values())

        privacy_amplification_object = PrivacyAmplification(list_key_bits)

        initial_key_length = self.correct_key.get_size()
        #compression_rate = 0.8
        compression_rate = self.compression_rate

        final_key_length = int(initial_key_length * compression_rate)

        _, _, secured_key = privacy_amplification_object.do_privacy_amplification(initial_key_length, final_key_length)

        print(f"Initial Correct Key Size: {initial_key_length}")
        print(f"Correct Key: {self.correct_key}")
        print(f"Final Secured Key Size: {len(secured_key)}")
        #print(f"Final Key: {secured_key}")
        print(f"THREAD ID: {self.thread_id} | SECURED KEY:\n{secured_key.tolist()}\n-----------\n")

        print(f"FINISHED Privacy Amplification Alice for Thread Id: {self.thread_id}")


'''
def process_receiving_and_sending_messages(connections_param, send_buffer_deque, bandwidth_param, propagation_delay_param, lock_param, reconciliation_array):
    done = True

    end_array = []

    while done:

        message_received_aux, to_tim_aux = receive_message_from_bob(connections_param, propagation_delay_param)

        print(f"LEN MESSAGE RECEIVED AUX: {len(message_received_aux)}")

        if len(message_received_aux):

            thread_id_aux, package_data = pickle.loads(message_received_aux)

            print(f"LEN MESSAGE RECEIVED AUX: {len(message_received_aux)} | Thread ID: {thread_id_aux}")

            reconcile_alice = reconciliation_array[thread_id_aux]

            reconcile_alice.add_receiving_information(package_data)

            print(f"Added Receiving Information to Thread Id: {thread_id_aux} | Alice Thread Id: {reconcile_alice.thread_id}")



            if package_data == "end_reconciliation":
                end_array.append(1)



        message_to_send = ()

        with lock_param:
            if send_buffer_deque:
                message_to_send = send_buffer_deque.pop()

        if message_to_send != ():

            thread_id_send, parity_blocks = message_to_send

            package_data = pickle.dumps(message_to_send)

            print(f"Initiating Sending Message to Bob Thread Id: {thread_id_send}")

            # Method if bandwidth is given to produce the bandwidth limitations
            if bandwidth_param > 0:
                init_pack = b"begin_reply"
                connections_param.sendall(init_pack)
                #send_with_bandwidth_limit(connections_param, package_data, bandwidth_param)
                connections_param.sendall(package_data)
                end_pack = b"end_reply"
                connections_param.sendall(end_pack)
            else:
                init_pack = b"begin_reply"
                connections_param.sendall(init_pack)
                connections_param.sendall(package_data)
                end_pack = b"end_reply"
                connections_param.sendall(end_pack)
        else:
            if len(end_array) >= 2:
                done = False
                break

'''

'''
def run_reconciliation_process_threading(connections_aux, propagation_delay_aux, key_divisions_bb84_aux, alice_socket_param, bandwidth_param):

    send_buffer_aux = deque()

    lock = threading.Lock()

    reconciliation_array = []

    for thread_id_aux, key in key_divisions_bb84_aux:
        lock_2 = threading.Lock()
        reconciliation_alice = Alice(thread_id_aux, key, send_buffer_aux, lock_2)
        reconciliation_array.append(reconciliation_alice)



    process_receiving_and_sending_messages_thread = threading.Thread(target=process_receiving_and_sending_messages, args=(connections_aux, send_buffer_aux, bandwidth_param, propagation_delay_aux, lock, reconciliation_array))



    thr_array = []

    process_receiving_and_sending_messages_thread.start()

    for reconcile_alice in reconciliation_array:
        thread_alice = threading.Thread(target=reconcile_alice.process_compute_parities_threading)
        thr_array.append(thread_alice)
        thread_alice.start()

    for thr in thr_array:
        thr.join()

    process_receiving_and_sending_messages_thread.join()
'''


if __name__ == "__main__":


    input_var = input("Alice use Threads? Number of threads (0 = no threads): ")

    if int(input_var) > 0:
        alice_manager = AliceManagerChannel()
        alice_manager.establish_connection()

        #alice_manager.create_alice_array(int(input_var))
        #alice_manager.process_threading_reconciliation()
        alice_manager.exchange_initial_bandwidth()
        alice_manager.create_classical_channel_link()
        alice_manager.process_set_of_runs_thread(int(input_var))
    else:

        alice_manager = AliceManager()

        alice_manager.establish_connection()

        alice_manager.process_set_of_runs()