import sys
import numpy as np
import time

from scipy.linalg import toeplitz, matmul_toeplitz
sys.path.append("..")


from ErrorReconciliation.cascade.key import Key
import random


class PrivacyAmplification:

    def __init__(self, key=None, thread_id=None, seed=2):
        #Key has a bit_array as a first approach
        #Should be changed to BitArray for a better implementation
        self.key = key

        #Seed for calculation
        self.seed = seed

        self._secured_key = None
        self.thread_id = thread_id


    def get_key_size(self):
        return len(self.key)

    def get_secured_key(self):
        return self._secured_key

    def generate_toeplitz_matrix(self, nr: int, mc: int):

        init_bit_row_aux, init_bit_col_aux = self._generate_row_and_column_seeded(nr, mc)
        #init_bit_row, init_bit_col = self.ge_2(nr, mc)

        init_bit_row_aux = np.array(init_bit_row_aux, dtype='uint8')
        init_bit_col_aux = np.array(init_bit_col_aux, dtype='uint8')

        toeplitz_matrix = toeplitz(init_bit_row_aux, init_bit_col_aux)

        return init_bit_row_aux, init_bit_col_aux, toeplitz_matrix


    def _generate_row_and_column_seeded(self, number_rows, number_columns):
        gen = random.Random(self.seed)

        # init_bit_row = gen.getrandbits(number_rows)
        # init_bit_col = gen.getrandbits(number_columns)

        init_bit_row_aux = [gen.randint(0, 1) for _ in range(number_rows)]
        init_bit_col_aux = [gen.randint(0, 1) for _ in range(number_columns)]

        #print(init_bit_row_aux)

        return init_bit_row_aux, init_bit_col_aux


    def _generate_row_and_column_seeded_slower(self, number_rows: int, number_columns: int):


        rng = np.random.RandomState(self.seed)

        init_bit_row_aux = rng.randint(0, 2, size=number_rows).astype('uint8')
        init_bit_col_aux = rng.randint(0, 2, size=number_columns).astype('uint8')
        print(init_bit_row_aux)
        print(init_bit_col_aux)
        
        return init_bit_row_aux, init_bit_col_aux


    def do_privacy_amplification(self, n, m):
        #key_arr_aux = [[random.randint(0, 1)] for _ in range(key_size)]

        #key_np_arr_aux = np.array(key_arr_aux, dtype=np.uint8)
        print("TOEPLITZ PRIVACY AMPLIFICATION")
        key_np_arr_aux = np.array(self.key, dtype=np.uint8).reshape(-1, 1)

        #print(f"Key NP ARR Aux:\n {key_np_arr_aux}")

        init_row_aux, init_col_aux = self._generate_row_and_column_seeded(n, m)

        print(f"Length init_row_aux: {len(init_row_aux)}")
        print(f"Length init_col_aux: {len(init_col_aux)}")

        secured_key_aux = np.round(matmul_toeplitz((np.array(init_col_aux, dtype=np.uint8), np.array(init_row_aux, dtype=np.uint8)), key_np_arr_aux)).astype(np.uint8) % 2

        #secured_key_aux = np.round(matmul_toeplitz((init_col_aux, init_row_aux), key_arr_aux)).astype(np.uint8) % 2

        secured_key_aux = secured_key_aux.T

        self._secured_key = secured_key_aux[0]

        return init_row_aux, init_col_aux, secured_key_aux[0]




    def do_privacy_amplification_numpy(self, n, m):
        key_arr_aux = [random.randint(0, 1) for _ in range(key_size)]

        key_np_arr_aux = np.array(key_arr_aux, dtype=np.uint8)


        init_row_aux, init_col_aux, hash_matrix_aux = self.generate_toeplitz_matrix(n, m)

        hash_np_matrix_aux = np.array(hash_matrix_aux, dtype=np.uint8)

        secured_key_aux = np.dot(key_np_arr_aux, hash_np_matrix_aux) % 2

        return init_row_aux, init_col_aux, hash_np_matrix_aux, secured_key_aux




    def xor_privacy_amplification(self):
        print("XOR PRIVACY AMPLIFICATION")
        xor_bit_array = []

        mid = int(len(self.key) / 2)

        first_part = self.key[:mid]
        second_part = self.key[mid:]

        print(f"Total Length: {len(self.key)}")
        print(f"First Part Length: {len(first_part)}")
        print(f"First Part Key:\n {first_part}")
        print(f"Second Part Length: {len(second_part)}")
        print(f"Second Part Key:\n {second_part}")


        #k & m represent each a single integer (bit) of 0 or 1 of each key and mask respectively
        for first_bit, second_bit in zip(first_part, second_part):
            res = first_bit ^ second_bit
            xor_bit_array.append(res)

        #This simply to convert to a smaller size np array - Done for simplicity, more efficient methods should be studied
        return np.array(xor_bit_array,dtype=np.uint8)


if __name__ == "__main__":




    key_size = 10000

    key_arr = [random.randint(0, 1) for _ in range(key_size)]

    security_coefficient = 0.8

    privacy_amplification = PrivacyAmplification(key=key_arr)

    hashed_size = int(key_size * security_coefficient)

    print(f"Initial Key:\n{key_arr}")

    secured_key = privacy_amplification.xor_privacy_amplification()


    print(f"Secured Key: \n {secured_key}")




    '''
    print("VERSION NUMPY DOT")

    start_time = time.perf_counter()
    init_row, init_col, hash_matrix, secured_key = privacy_amplification.do_privacy_amplification_numpy(key_size, hashed_size)
    end_time = time.perf_counter()

    print(f"Init Row:\n {init_row}")
    print(f"Init Col:\n {init_col}")
    print(f"Hash Matrix:\n {hash_matrix}")

    print(f"Secured Key:\n {secured_key}")




    print("VERSION SCIPY MATMUL")

    start_time_matmul = time.perf_counter()
    init_row_matmul, init_col_matmul, secured_key_matmul = privacy_amplification.do_privacy_amplification(key_size, hashed_size)
    end_time_matmul = time.perf_counter()

    print(f"Init Row MATMUL:\n {init_row_matmul}")
    print(f"Init Col MATMUL:\n {init_col_matmul}")

    #np.set_printoptions(threshold=sys.maxsize)

    print(f"Secured Key MATMUL:\n {secured_key_matmul.tolist()}")

    print(f"Total Time: {end_time - start_time}")
    print(f"Total Time MATMUL: {end_time_matmul - start_time_matmul}")

    thisdict = {}


    for i in range(20):
        x = random.randint(0, 10)
        thisdict[i] = x


    print(list(thisdict.items()))
    print(list(thisdict.values()))
    
    '''

    pass