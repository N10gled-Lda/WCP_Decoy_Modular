import unittest

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.cmac import CMAC

from communication_channel.common import timestamp, MAX_CREATION_TIMESTAMP_AGE_SECONDS, generate_connection_info, \
    convert_float_to_bytes, convert_int_to_bytes, TIMESTAMP_SIZE_BYTES
from communication_channel.mac_config import MAC_Config, MAC_Algorithms
from classical_communication_channel.communication_channel.role import Role, SEQUENCE_NUMBER_LENGTH, OFFSET_LENGTH_BYTES


class TestParticipantMAC(unittest.TestCase):
    def setUp(self):
        """
        Create a Role with a known secret key.
        Use a 16-byte key for AES CMAC (128 bits).
        """
        self.secret_key = b'\x01' * 16  # simple repeating 0x01
        self.participant_id = 42
        self.bandwidth_limit_MBps = 5
        self.mac_config = MAC_Config(MAC_Algorithms.CMAC, self.secret_key)
        self.role = Role.get_instance(generate_connection_info(), self.bandwidth_limit_MBps, 5,
                                      5, self.mac_config)

    def test_calculate_mac_returns_correct_length(self):
        """Test that _calculate_mac returns a MAC of the expected size (16 bytes for AES-128 CMAC)."""
        message = b"Test packet"
        mac = self.role._calculate_mac(message)
        self.assertEqual(len(mac), 16, "MAC size should be 16 bytes for AES CMAC.")

    def test_calculate_mac_matches_reference_cmac(self):
        """
        Compare _calculate_mac to an external/known correct CMAC using the same key.
        You can compute this in-line or store a known MAC if the packet is fixed.
        """
        message = b"Known reference packet"
        # Calculate "ground truth" MAC using the same library externally
        reference_cmac = CMAC(algorithms.AES(self.secret_key))
        reference_cmac.update(message)
        expected_mac = reference_cmac.finalize()

        # Compare to the participant's method
        actual_mac = self.role._calculate_mac(message)
        self.assertEqual(actual_mac, expected_mac, "MAC should match the reference CMAC.")

    def test_verify_message_mac_valid(self):
        """_verify_message_mac should not raise if the packet and MAC are correct."""
        message = b"Valid packet"
        mac = self.role._calculate_mac(message)

        # Combine packet + mac to emulate how the code calls _verify_message_mac
        message_with_mac = message + mac

        # Should not raise an exception
        try:
            self.role._has_valid_mac(message_with_mac)
        except InvalidSignature:
            self.fail("Valid MAC caused InvalidSignature")

    def test_verify_message_mac_invalid(self):
        """_verify_message_mac should raise InvalidSignature if the MAC is incorrect."""
        message = b"Some packet"
        correct_mac = self.role._calculate_mac(message)

        # Tamper with the last byte of the correct MAC
        wrong_mac = correct_mac[:-1] + b'\x00' if correct_mac[-1] != 0 else b'\x01'
        message_with_wrong_mac = message + wrong_mac

        # Expect an InvalidSignature
        self.assertFalse(self.role._has_valid_mac(message_with_wrong_mac))

    def test_verify_message_mac_tampered_data(self):
        """
        If the packet was tampered with (but the appended MAC is still the 'correct' one
        for the original packet), the verification should fail.
        """
        original_message = b"Original"
        mac = self.role._calculate_mac(original_message)

        # Tamper with the first byte of the packet
        tampered_message = b"\xff" + original_message[1:]

        # Combine tampered packet with the correct MAC for the original packet
        tampered_message_with_mac = tampered_message + mac

        self.assertFalse(self.role._has_valid_mac(tampered_message_with_mac))

    def test_verify_message_expired_timestamp(self):
        payload = bytearray(self.role._generate_data_packet_bytes(0, 0, b'Hello'))
        payload[1 + SEQUENCE_NUMBER_LENGTH + OFFSET_LENGTH_BYTES:
                  1 + SEQUENCE_NUMBER_LENGTH + OFFSET_LENGTH_BYTES + TIMESTAMP_SIZE_BYTES] = convert_float_to_bytes(timestamp() - MAX_CREATION_TIMESTAMP_AGE_SECONDS)
        self.assertFalse(self.role._has_valid_timestamp(payload))
        payload[1 + SEQUENCE_NUMBER_LENGTH + OFFSET_LENGTH_BYTES:
                1 + SEQUENCE_NUMBER_LENGTH + OFFSET_LENGTH_BYTES + TIMESTAMP_SIZE_BYTES] = convert_float_to_bytes(timestamp() + MAX_CREATION_TIMESTAMP_AGE_SECONDS)
        self.assertTrue(self.role._has_valid_timestamp(payload))

    def test_verify_message_mac_tampered_timestamp(self):
        payload = convert_int_to_bytes(0) + convert_float_to_bytes(timestamp()) + bytearray(64)
        mac = self.role._calculate_mac(payload)
        tampered_payload = convert_int_to_bytes(0) + convert_float_to_bytes(timestamp() + 150) + bytearray(64) + mac
        self.assertFalse(self.role._has_valid_mac(tampered_payload))



if __name__ == '__main__':
    unittest.main()
