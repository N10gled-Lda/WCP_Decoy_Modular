import queue

from mock.mock import MagicMock

class MockSockets:
    def __init__(self):
        self._bob_to_alice_direction = queue.Queue()
        self._alice_to_bob_direction = queue.Queue()

        self.bob_send_socket = MagicMock()
        self.bob_receive_socket = MagicMock()
        self.alice_send_socket = MagicMock()
        self.alice_receive_socket = MagicMock()

        self.bob_send_socket.sendall.side_effect = self._bob_send
        self.bob_send_socket.accept.return_value = self.bob_send_socket, ""
        self.bob_send_socket.__enter__.return_value = self.bob_send_socket
        self.bob_send_socket.__exit__.return_value = self.bob_send_socket

        self.alice_receive_socket.recv.side_effect = self._alice_recv
        self.alice_receive_socket.accept.return_value = self.alice_receive_socket, ""
        self.alice_receive_socket.__enter__.return_value = self.alice_receive_socket
        self.alice_receive_socket.__exit__.return_value = self.alice_receive_socket

        self.alice_send_socket.sendall.side_effect = self._alice_send
        self.alice_send_socket.accept.return_value = self.alice_send_socket, ""
        self.alice_send_socket.__enter__.return_value = self.alice_send_socket
        self.alice_send_socket.__exit__.return_value = self.alice_send_socket

        self.bob_receive_socket.recv.side_effect = self._bob_recv
        self.bob_receive_socket.accept.return_value = self.bob_receive_socket, ""
        self.bob_receive_socket.__enter__.return_value = self.bob_receive_socket
        self.bob_receive_socket.__exit__.return_value = self.bob_receive_socket


    # Define how the bob send works: push data into _bob_to_alice_direction.
    def _bob_send(self, data):
        #print(f"Bob Send to Alice Direction")
        self._bob_to_alice_direction.put(data)


    # Define how the alice receives: get data from _bob_to_alice_direction.
    def _alice_recv(self, bufsize):
        #print(f"Alice Receives from Bob Direction")
        return self._bob_to_alice_direction.get()


    # Define how the alice send works: push data into _alice_to_bob_direction.
    def _alice_send(self, data):
        #print(f"Alice Send to Bob Direction")
        self._alice_to_bob_direction.put(data)


    # Define how the bob receives: get data from _alice_to_bob_direction.
    def _bob_recv(self, bufsize):
        #print(f"Bob Receives from Alice Direction")
        return self._alice_to_bob_direction.get()



