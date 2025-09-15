import random
import logging
from datetime import datetime

class Shuffle:
    """
    A shuffling (i.e. re-ordering) of the bits in a key.
    """

    _shuffle_seed_random_generator = random.Random()

    SHUFFLE_KEEP_SAME = 0
    """Do not shuffle the bits in the key."""
    SHUFFLE_RANDOM = 1
    """Randomly shuffle the bits in the key."""

    _MAX_KEY_SIZE = 1_000_000_000
    _MAX_ALGORITHM = 100
    _MAX_SHUFFLE_SEED = 1_000_000_000_000
    #_MAX_SHUFFLE_SEED = 1

    """if __debug__:
        _logger = logging.getLogger("Shuffle")
        _logger.setLevel(logging.DEBUG)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        _file_handler = logging.FileHandler(f'logs/shuffle_{timestamp}.log', mode='w')
        _file_handler.setLevel(logging.DEBUG)

        _logger.addHandler(_file_handler)"""

    def __init__(self, size, algorithm, shuffle_seed=None):
        """
        Create a shuffle. A shuffle represents a permutation of the bits in a key. The shuffle
        can be random or deterministic depending on the shuffle algorithm. A Shuffle object is
        de-coupled from the Key objects: the same Shuffle object can be applied to multiple
        different Key objects, to permute (shuffle) the bits in those different keys according to
        the same pattern.

        Args:
            size (int): The size of the shuffle, i.e. the number of bits in the keys that this
                shuffle will be applied to. Must be >= 0 (i.e. empty shuffles are allowed).
            algorithm (int): The algorithm for generating the shuffle pattern:
                SHUFFLE_KEEP_SAME: Do not shuffle the key (keep the key bits in the original order).
                SHUFFLE_RANDOM: Randomly shuffle the key.
            shuffle_seed (None or int): The seed value for the isolated shuffle random number
                generator that is used to generate the shuffling permutation. If shuffle_seed is
                None, then a random shuffle_seed value will be generated.
        """
        self._size = size
        self._shuffle_index_to_key_index = {}
        self._aux_shuffle_index_to_key_index = {}
        for shuffle_index in range(0, size):
            self._shuffle_index_to_key_index[shuffle_index] = shuffle_index
            self._aux_shuffle_index_to_key_index[shuffle_index] = shuffle_index
        if algorithm == self.SHUFFLE_RANDOM:
            if shuffle_seed is None:
                shuffle_seed = \
                    Shuffle._shuffle_seed_random_generator.randint(1, Shuffle._MAX_SHUFFLE_SEED - 1)
            shuffle_random_generator = random.Random(shuffle_seed)

            #aux_shuffle_random_generator = random.Random(shuffle_seed)

            #Shuffle._logger.debug(f"-------\nShuffle_index_to_key_index Before: {self._shuffle_index_to_key_index}")
            #Shuffle._logger.debug(f"-------\nAux Shuffle_index_to_key_index Before: {self._aux_shuffle_index_to_key_index}")

            shuffle_random_generator.shuffle(self._shuffle_index_to_key_index)

            #l = list(self._aux_shuffle_index_to_key_index.items())

            #Shuffle._logger.debug(f"l: {l}")

            #aux_shuffle_random_generator.shuffle(l)
            #d = dict(l)

            #Shuffle._logger.debug(f"Shuffle_index_to_key_index After: {self._shuffle_index_to_key_index}")
            #Shuffle._logger.debug(f"Aux Shuffle_index_to_key_index After: {d}")
            #random.shuffle(self._shuffle_index_to_key_index, shuffle_random_generator.random)
        else:
            shuffle_seed = 0
        self._identifier = Shuffle._encode_identifier(size, algorithm, shuffle_seed)

        #LOGGING
        #self.logger = logging.getLogger(self.__class__.__name__)
        #self.logger.setLevel(logging.DEBUG)

        #timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        #file_handler = logging.FileHandler(f'logs/{self.__class__.__name__.lower()}_{timestamp}.log', mode='w')
        #file_handler.setLevel(logging.DEBUG)

        #self.logger.addHandler(file_handler)


    @staticmethod
    def create_shuffle_from_identifier(identifier):
        """
        Create a shuffle object from a shuffle identifier.

        Alice and Bob need to agree on how to shuffle the bits in each pass. Bob could send complete
        shuffle objects to Alice, but that would be expensive because shuffle objects are large.
        Instead, Bob sends a short shuffle identifier from Alice from which Alice can reconstruct
        the shuffle object.

        Args:
            identifier (int): The shuffle identifier.
        """
        print("Entra aqui")
        (size, algorithm, shuffle_seed) = Shuffle._decode_identifier(identifier)
        shuffle = Shuffle(size, algorithm, shuffle_seed)
        return shuffle

    @staticmethod
    def _encode_identifier(size, algorithm, shuffle_seed):
        identifier = shuffle_seed

        #Shuffle._logger.debug(f"------ENCODE IDENTIFIER:\nIdentifier - Shuffle Seed: {identifier}")
        identifier *= Shuffle._MAX_ALGORITHM
        #Shuffle._logger.debug(f"Identifier - Shuffle Seed * Shuffle._MAX_ALGORITHM: {identifier}")
        identifier += algorithm
        #Shuffle._logger.debug(f"Identifier - Shuffle Seed * Shuffle._MAX_ALGORITHM + algorithm: {identifier} | algorithm: {algorithm}")
        identifier *= Shuffle._MAX_KEY_SIZE
        #Shuffle._logger.debug(f"Identifier - Shuffle Seed * Shuffle._MAX_ALGORITHM + algorithm * Shuffle._MAX_KEY_SIZE: {identifier}")
        identifier += size
        #Shuffle._logger.debug(f"Identifier - Shuffle Seed * Shuffle._MAX_ALGORITHM + algorithm * Shuffle._MAX_KEY_SIZE + size: {identifier} | {size}\n----------------")
        return identifier

    @staticmethod
    def _decode_identifier(identifier):
        size = identifier % Shuffle._MAX_KEY_SIZE
        #Shuffle._logger.debug(f"------DECODE IDENTIFIER:\nIdentifier - size: {size}")
        identifier //= Shuffle._MAX_KEY_SIZE
        algorithm = identifier % Shuffle._MAX_ALGORITHM
        #Shuffle._logger.debug(f"Identifier algorithm: {algorithm}")
        identifier //= Shuffle._MAX_ALGORITHM
        shuffle_seed = identifier
        #Shuffle._logger.debug(f"Identifier Shuffle_seed: {shuffle_seed}\n--------------")
        return (size, algorithm, shuffle_seed)

    def __repr__(self):
        """
        Get the unambiguous string representation of the shuffle.

        Returns:
            The unambiguous string representation of the shuffle.

        Example:
            >>> shuffle.__repr__()
            'Shuffle: 0->3 1->1 2->2 3->0 4->4 5->5'
        """
        return "Shuffle: " + self.__str__()

    def __str__(self):
        """
        Get the human-readable string representation of the shuffle.

        Returns:
            The human-readable string representation of the shuffle.

        Example:
            >>> shuffle.__str__()
            '0->3 1->1 2->2 3->0 4->4 5->5'
        """
        string = ""
        for shuffle_index in range(self._size):
            key_index = self._shuffle_index_to_key_index[shuffle_index]
            if string:
                string += " "
            string += f"{shuffle_index}->{key_index}"
        return string

    @staticmethod
    def set_random_seed(seed):
        """
        Set the seed for the isolated random number generated that is used only in the shuffle
        module and nowhere else. If two applications set the seed to the same value, the shuffle
        module produces the exact same sequence of shuffles. This is used to make experiments
        reproduceable.

        Args:
            seed (int): The seed value for the random number generator which is isolated to the
                shuffle module.
        """
        Shuffle._shuffle_seed_random_generator = random.Random(seed)

    def get_size(self):
        """
        Get the size of the shuffle in bits.

        Returns:
            The size of the shuffle in bits.
        """
        return self._size

    def get_identifier(self):
        """
        Get the shuffle identifier.

        Returns:
            The shuffle identifier.
        """
        return self._identifier

    def get_key_index(self, shuffle_index):
        """
        Get the key index that a given shuffle index is mapped to.

        Args:
            shuffle_index (int): The shuffle index of the bit. Index must be in range
                [0, shuffle._size).

        Returns:
            The key index.
        """
        return self._shuffle_index_to_key_index[shuffle_index]

    def get_shuffle_index_to_key_index(self):
        return self._shuffle_index_to_key_index

    def get_bit(self, key, shuffle_index):
        """
        Get a bit from a shuffled key.

        Args:
            key (Key): The key. We first shuffle this key according to this shuffle pattern and
                then retrieve the bit at shuffle_index in the shuffled key. The size of the key
                must be equal to the size of this shuffle.
            shuffle_index (int): The index of the bit in the shuffled key. The index must be in
                range [0, shuffle.size).

        Returns:
            The value (0 or 1) of the shuffled key bit at the given index.
        """
        key_index = self._shuffle_index_to_key_index[shuffle_index]
        return key.get_bit(key_index)

    def set_bit(self, key, shuffle_index, value):
        """
        Set a bit in a shuffled key to a given value.

        Args:
            key (Key): The key. We first shuffle this key according to this shuffle pattern and
                then set the bit at shuffle_index in the shuffled key to the given value. The size
                of the key must be equal to the size of this shuffle.
            shuffle_index (int): The index of the bit in the shuffled key. The index must be in
                range [0, shuffle.size).
            value (int): The new value of the bit. Must be 0 or 1.
        """
        key_index = self._shuffle_index_to_key_index[shuffle_index]
        key.set_bit(key_index, value)

    def flip_bit(self, key, shuffle_index):
        """
        Flip a bit in a shuffled key (flip 0 to 1 and vice versa).

        Args:
            key (Key): The key. We first shuffle this key according to this shuffle pattern and
                then flip the bit at shuffle_index in the shuffled key. The size of the key must be
                equal to the size of this shuffle.
            shuffle_index (int): The index of the bit in the shuffled key. The index must be in
                range [0, shuffle.size).
        """
        key_index = self._shuffle_index_to_key_index[shuffle_index]
        key.flip_bit(key_index)

    def calculate_parity(self, key, shuffle_start_index, shuffle_end_index):
        """
        Calculate the parity of a contiguous sub-range of bits in a shuffled key.

        Args:
            key (Key): The key for which to calculate the parity after shuffling it.
            shuffle_start_index (int): The index of the first bit (inclusive) in the range of
                bits in the shuffled key over which to calculate the parity.
            shuffle_end_index (int): The index of the last bit (exclusive) in the range of
                bits in the shuffled key over which to calculate the parity.

        Returns:
            The parity of the contiguous sub-range of bits in the shuffled key.
        """
        parity = 0
        for shuffle_index in range(shuffle_start_index, shuffle_end_index):
            key_index = self._shuffle_index_to_key_index[shuffle_index]
            if key.get_bit(key_index):
                parity = 1 - parity
        return parity
