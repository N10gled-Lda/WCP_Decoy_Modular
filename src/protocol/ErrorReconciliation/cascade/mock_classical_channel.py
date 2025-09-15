from ErrorReconciliation.cascade.classical_channel import ClassicalChannel
import random
import copy
import socket
import struct
import pickle

from ErrorReconciliation.cascade.reconciliation import Reconciliation


class MockClassicalChannel(ClassicalChannel):
    """
    A mock concrete implementation of the ClassicalChannel base class, which is used for the
    experiments.
    """

    """
    PF 3:12 PM 12/5/2024 
    Considering this as Alice for testing purposes of other ask_parities options
    """


    def __init__(self, correct_key):
        self._correct_key = correct_key
        self._id_to_shuffle = {}
        self._reconciliation_started = False
        self.shuffled_key = []

        self.artificial_alice_ip_address = "127.0.0.3"
        self.artificial_alice_port = 5003

        self.alice_ip_address = "127.0.0.4"
        self.alice_port = 5004



    def send_correct_key(self):
        median_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        median_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:

            median_socket.bind((self.artificial_alice_ip_address, self.artificial_alice_port))

            median_socket.connect((self.alice_ip_address, self.alice_port))

            serialized_key = pickle.dumps(self._correct_key)

            print(f"size in bytes of the key: {len(serialized_key)}\nsize in bits of the key:{len(serialized_key) * 8}")

            median_socket.sendall(serialized_key)

            median_socket.close()
        except Exception as e:
            print(f"Median - error occurred - {e}")
        finally:
            # Close the connection and server sockets
            print("Median: Closing the connection.")
            median_socket.close()

    def send_correct_key_and_propagation_delay_and_bandwidth(self, propagation_delay, bandwidth, n_r, it_n):
        median_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:

            median_socket.bind((self.artificial_alice_ip_address, self.artificial_alice_port))

            print(self.alice_port + ((10*n_r) * it_n))

            median_socket.connect((self.alice_ip_address, self.alice_port + ((10*n_r)* it_n)))

            serialized_key = pickle.dumps([self._correct_key,propagation_delay,bandwidth])

            print(f"size in bytes of the key: {len(serialized_key)}\nsize in bits of the key:{len(serialized_key) * 8}")

            median_socket.sendall(serialized_key)

            median_socket.close()
        except Exception as e:
            print(f"Median - error occurred - {e}")
        finally:
            # Close the connection and server sockets
            print("Median: Closing the connection.")
            median_socket.close()








    def start_reconciliation(self):
        self._reconciliation_started = True

        #This method is used for the "reconciliation_minimum"
    def start_reconciliation_minimum(self, iteration_number, shuffle_seed, block_size, reconciliation):
        #self._reconciliation_started = True

        #Create copy of correct key for consistency and not losing the original key
        #copy_correct_key = self._correct_key.copy()
        copy_correct_key = copy.deepcopy(self._correct_key)

        #Necessary shuffle before breaking key
        if shuffle_seed:
            random_generator = random.Random(shuffle_seed)
            #Possible using deepcopy here, without calling key method with none parameters
            random_generator.shuffle(copy_correct_key._bits)

        parity_bits_blocks = []

        block_generator = self._divide_chunks(copy_correct_key, block_size)

        if __debug__:
            reconciliation.print_message("Alice Blocks:            ", end=" ")
            add_start_string = " " * (iteration_number - 1)
            add_end_string = " " * (iteration_number - 1)

        #print("Alice Blocks:            ", end=" ", file=file)



        counter = 0
        #Generate the parity bits for each block
        for block in block_generator:
            bit = self.calculate_parity_sub_block(block)
            parity_bits_blocks.append(bit)
            if __debug__:
                reconciliation.print_message(Reconciliation._format_key_array(noisy_key=block, correct_key=copy_correct_key, start_index=counter*block_size), end=add_start_string + " | " + add_end_string)
                #print(Reconciliation._format_key_array(noisy_key=block, correct_key=copy_correct_key, start_index=counter*block_size), end=add_start_string + " | " + add_end_string, file=file)

            counter += 1

        if __debug__:
            #print(f"\nAlice Block Parity Bits: {parity_bits_blocks}", file=file)
            reconciliation.print_message(f"\nAlice Block Parity Bits: {parity_bits_blocks}")

        #Return the parity values to Bob
        return parity_bits_blocks, copy_correct_key


    @staticmethod
    def _divide_chunks(lst, n):
        #Yield the division of the key in blocks - This allows for a better efficiency as there is not time in dividing the key
        #This is not the correct way - It is only done due to the original way of key organization - This should be changed for efficiency
        for i in range(0, lst.get_size(), n):
            aux_block = []
            for index in range(i, i+n):
                if index < lst.get_size():
                    aux_bit = lst.get_bit(index)
                    aux_block.append(aux_bit)
            #yield aux_block[i:i+n]
            yield aux_block


    def calculate_parity_sub_block(self, block):
        parity = 0
        for el in block:
            if el:
                parity = 1 - parity

        return parity


    def end_reconciliation(self):
        self._reconciliation_started = False
        self._id_to_shuffle = {}

    def ask_parities(self, blocks):
        parities = []
        for block in blocks:
            shuffle = block.get_shuffle()
            start_index = block.get_start_index()
            end_index = block.get_end_index()
            parity = shuffle.calculate_parity(self._correct_key, start_index, end_index)
            #block.set_correct_parity(parity)
            parities.append(parity)
        return parities

    #Due to security issues in Sending Entire Block Objects as it contains the entire key changing the way the parities are calculate without the Block Objects
    def ask_parities_secure(self, blocks):
        parities = []
        for block in blocks:
            #print(f"Block: {block}")
            start_index, end_index, shuffle = block
            #print(f"Start Index: {start_index}")
            #print(f"End Index: {start_index}")

            parity = shuffle.calculate_parity(self._correct_key, start_index, end_index)

            parities.append(parity)

        return parities

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

    def ask_parities_secure_optimized(self, blocks):
        parities_aux = []
        for block in blocks:
            start_index, end_index, shuffle_index_to_key_index_dict = block
            parity = self.calculate_parity_alice(self._correct_key, start_index, end_index, shuffle_index_to_key_index_dict)
            parities_aux.append((start_index, end_index, parity))

        return parities_aux


    def shuffle_key(self, shuffle_seed):
        generator = random.Random(shuffle_seed)

        #This copy takes O(n)
        self.shuffled_key = self._correct_key[:]

        generator.shuffle(self.shuffled_key)



