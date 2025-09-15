from enum import Enum

from cryptography.hazmat.primitives import cmac, hmac, hashes
from cryptography.hazmat.primitives.ciphers import algorithms


class MAC_Algorithms(Enum):
    HMAC = 'hmac'
    CMAC = 'cmac'


class MAC_Config:
    def __init__(self, algorithm: MAC_Algorithms, mac_secret_key_16_bits: bytes):
        """
        Creates a configuration for using Message Authentication Codes (MAC) with participants
        :param algorithm: the algorithm to be used
        :param mac_secret_key_16_bits: the 16-bit shared secret key between the participants that use MAC
        """
        self.algorithm = algorithm
        self._mac_secret_key_16_bits = mac_secret_key_16_bits

    def get_mac(self):
        match self.algorithm:
            case MAC_Algorithms.HMAC:
                return hmac.HMAC(self._mac_secret_key_16_bits, hashes.SHA256())
            case MAC_Algorithms.CMAC:
                return cmac.CMAC(algorithms.AES(self._mac_secret_key_16_bits))
