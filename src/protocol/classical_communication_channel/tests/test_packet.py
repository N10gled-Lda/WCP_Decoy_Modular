import os
import pickle
import unittest
from random import random, randint

from communication_channel.common import convert_int_to_bytes
from communication_channel.packet import (get_payloads_from_received_frames, FRAME_END, FRAME_ESCAPE,
                                          TRANSPOSED_FRAME_END, TRANSPOSED_FRAME_ESCAPE, convert_to_kiss_frame)

NUM_PACKETS = 100
NUM_PAYLOADS = NUM_PACKETS

class TestKissPacketFixedPayloadSize(unittest.TestCase):

    def test_frame_separation(self):
        _payload = pickle.dumps([1 for i in range(NUM_PACKETS)])
        payloads = [_payload for j in range(NUM_PACKETS)]
        message = bytearray()
        for payload in payloads:
            message.extend(convert_to_kiss_frame(payload))
        payloads_from_message = get_payloads_from_received_frames(message)
        self.assertEqual(len(payloads_from_message), NUM_PACKETS)
        self.assertTrue(all(payload_from_message == _payload for payload_from_message in payloads_from_message))

    def test_frame_separation_with_escaped_frame_end(self):
        _payload = bytearray()
        for i in range(NUM_PACKETS):
            if random() > 0.5:
                _payload.extend(FRAME_END)
            else:
                _payload.extend(int.to_bytes(1))
        payloads = [_payload for j in range(NUM_PACKETS)]
        message = bytearray()
        for payload in payloads:
            message.extend(convert_to_kiss_frame(payload))
        payloads_from_message = get_payloads_from_received_frames(message)
        self.assertEqual(len(payloads_from_message), NUM_PACKETS)
        self.assertTrue(all(payload_from_message == _payload for payload_from_message in payloads_from_message))

    def test_frame_separation_with_escaped_frame_escape(self):
        _payload = bytearray()
        for i in range(NUM_PACKETS):
            if random() > 0.5:
                _payload.extend(FRAME_ESCAPE)
            else:
                _payload.extend(int.to_bytes(1))
        payloads = [_payload for j in range(NUM_PACKETS)]
        message = bytearray()
        for payload in payloads:
            message.extend(convert_to_kiss_frame(payload))
        payloads_from_message = get_payloads_from_received_frames(message)
        self.assertEqual(len(payloads_from_message), NUM_PACKETS)
        self.assertTrue(all(payload_from_message == _payload for payload_from_message in payloads_from_message))

    def test_frame_separation_with_escaped_frame_escape_and_escaped_frame_end(self):
        _payload = bytearray()
        for i in range(NUM_PACKETS):
            if random() > 0.5:
                if random() > 0.5:
                    _payload.extend(FRAME_END)
                else:
                    _payload.extend(FRAME_ESCAPE)
            else:
                _payload.extend(int.to_bytes(1))
        payloads = [_payload for j in range(NUM_PACKETS)]
        message = bytearray()
        for payload in payloads:
            message.extend(convert_to_kiss_frame(payload))
        payloads_from_message = get_payloads_from_received_frames(message)
        self.assertEqual(len(payloads_from_message), NUM_PACKETS)
        self.assertTrue(all(payload_from_message == _payload for payload_from_message in payloads_from_message))


class TestKissPacketVariablePayloadSize(unittest.TestCase):

    def test_frame_separation(self):
        payloads = []
        for i in range(NUM_PAYLOADS):
            payload = pickle.dumps([randint(0,10) for i in range(randint(1, NUM_PACKETS))])
            payloads.append(payload)
        message = bytearray()
        for payload in payloads:
            message.extend(convert_to_kiss_frame(payload))
        payloads_from_message = get_payloads_from_received_frames(message)
        self.assertEqual(len(payloads_from_message), NUM_PACKETS)
        self.assertTrue(all(payloads_from_message[i] == payloads[i] for i in range(NUM_PACKETS)))


    def test_frame_separation_with_escaped_frame_end(self):
        payloads = []
        for i in range(NUM_PAYLOADS):
            payload = bytearray()
            for j in range(randint(1, NUM_PACKETS)):
                if random() > 0.5:
                    payload.extend(FRAME_END)
                else:
                    payload.extend(int.to_bytes(randint(0,10)))
            payloads.append(payload)

        message = bytearray()
        for payload in payloads:
            message.extend(convert_to_kiss_frame(payload))
        payloads_from_message = get_payloads_from_received_frames(message)
        self.assertEqual(len(payloads_from_message), NUM_PACKETS)
        self.assertTrue(all(len(payloads_from_message[i]) == len(payloads[i]) for i in range(NUM_PACKETS)))
        self.assertTrue(all(payloads_from_message[i] == payloads[i] for i in range(NUM_PACKETS)))
        print()

    def test_frame_separation_with_escaped_frame_escape(self):
        payloads = []
        for i in range(NUM_PAYLOADS):
            payload = bytearray()
            for j in range(randint(1, NUM_PACKETS)):
                if random() > 0.5:
                    payload.extend(FRAME_ESCAPE)
                else:
                    payload.extend(int.to_bytes(randint(0, 10)))
            payloads.append(payload)

        message = bytearray()
        for payload in payloads:
            message.extend(convert_to_kiss_frame(payload))
        payloads_from_message = get_payloads_from_received_frames(message)
        self.assertEqual(len(payloads_from_message), NUM_PACKETS)
        self.assertTrue(all(len(payloads_from_message[i]) == len(payloads[i]) for i in range(NUM_PACKETS)))
        self.assertTrue(all(payloads_from_message[i] == payloads[i] for i in range(NUM_PACKETS)))
        print()

    def test_frame_separation_with_escaped_frame_escape_and_escaped_frame_end(self):
        payloads = []
        for i in range(NUM_PAYLOADS):
            payload = bytearray()
            for j in range(randint(1, NUM_PACKETS)):
                if random() > 0.5:
                    if random() > 0.5:
                        payload.extend(FRAME_ESCAPE)
                    else:
                        payload.extend(FRAME_END)
                else:
                    payload.extend(int.to_bytes(randint(0, 10)))
            payloads.append(payload)

        message = bytearray()
        for payload in payloads:
            message.extend(convert_to_kiss_frame(payload))
        packets_from_message = get_payloads_from_received_frames(message)
        self.assertEqual(len(packets_from_message), NUM_PACKETS)
        self.assertTrue(all(len(packets_from_message[i]) == len(payloads[i]) for i in range(NUM_PACKETS)))
        self.assertTrue(all(packets_from_message[i] == payloads[i] for i in range(NUM_PACKETS)))
        print()

class TestGetPacketFromBytes(unittest.TestCase):
    def test_valid_message(self):
        random_payload = os.urandom(10)
        for j in range(1, 1_000):
            payloads = [random_payload + convert_int_to_bytes(j) for j in range(1, j + 1)]
            message = bytearray()
            for payload in payloads:
                message.extend(convert_to_kiss_frame(payload))
            packets_from_message = get_payloads_from_received_frames(message)
            self.assertEqual(payloads, packets_from_message)

    def test_message_without_proper_start_byte(self):
        random_payload = os.urandom(10)
        for j in range(1, 1_000):
            payloads = [random_payload + convert_int_to_bytes(j) for j in range(1, j + 1)]
            message = bytearray()
            for payload in payloads:
                message.extend(convert_to_kiss_frame(payload))
            message.pop(0)  # remove FRAME_END from the first packet (as bytes)
            payloads_from_message = get_payloads_from_received_frames(message)
            payloads.pop(0)
            self.assertEqual(payloads, payloads_from_message)

    def test_message_without_proper_end_byte(self):
        random_payload = os.urandom(10)
        for j in range(1, 1_000):
            payloads = [random_payload + convert_int_to_bytes(j) for j in range(1, j + 1)]
            message = bytearray()
            for payload in payloads:
                message.extend(convert_to_kiss_frame(payload))
            message.pop()
            payloads_from_message = get_payloads_from_received_frames(message)
            payloads.pop()
            self.assertEqual(payloads, payloads_from_message)

    def test_message_without_proper_start_and_end_byte(self):
        random_payload = os.urandom(10)
        for j in range(1, 1_000):
            payloads = [random_payload + convert_int_to_bytes(j) for j in range(1, j + 1)]
            message = bytearray()
            message.extend(convert_to_kiss_frame(payloads[0])[1:-1])
            for payload in payloads[1:]:
                message.extend(convert_to_kiss_frame(payload))
            payloads_from_message = get_payloads_from_received_frames(message)
            payloads.pop(0)
            self.assertEqual(payloads, payloads_from_message)

    def test_empty_data(self):
        """Test that empty input returns no payloads and no incomplete frames."""
        packets = get_payloads_from_received_frames(b"")
        self.assertEqual(packets, [])

    def test_single_complete_packet(self):
        """Test a single well-formed KISS packet with start/end delimiters."""
        for i in range(1_000):
            unfiltered_payload = os.urandom(4096)
            filtered_payload = bytearray()
            for _byte in unfiltered_payload:
                if _byte != FRAME_END[0]:
                    filtered_payload.append(_byte)
            message = FRAME_END + filtered_payload + FRAME_END
            payloads = get_payloads_from_received_frames(message)
            self.assertEqual(len(payloads), 1, "Should parse exactly one complete packet.")

        payload = b'1234' * 1_000
        message = FRAME_END + payload + FRAME_END
        payloads = get_payloads_from_received_frames(message)
        self.assertEqual(len(payloads), 1, "Should parse exactly one complete packet.")
        self.assertEqual(payloads[0], payload)

    def test_escaped_data(self):
        """Test typical KISS escaping with FRAME_ESCAPE + TRANSPOSED_FRAME_END/TRANSPOSED_FRAME_ESCAPE."""
        # Suppose we want one packet with a literal FRAME_END and FRAME_ESCAPE in it:
        #   payload = [0xC0, 0xDB]
        # The KISS encoding for that is:
        #   FRAME_END + (escape + transposed_frame_end) + (escape + transposed_frame_escape) + FRAME_END
        payload_encoded = FRAME_ESCAPE + TRANSPOSED_FRAME_END + FRAME_ESCAPE + TRANSPOSED_FRAME_ESCAPE
        message = FRAME_END + payload_encoded + FRAME_END
        packets = get_payloads_from_received_frames(message)
        self.assertEqual(len(packets), 1, "Should parse one complete packet.")
        # If the code is correct, the payload should decode to b"\xC0\xDB"
        expected_payload = b"\xC0\xDB"
        self.assertEqual(packets[0], expected_payload)

if __name__ == '__main__':
    unittest.main()
