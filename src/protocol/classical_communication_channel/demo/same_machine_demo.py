from communication_channel.common import wait_synchronously
from communication_channel.connection_info import ConnectionInfo
from communication_channel.mac_config import MAC_Config, MAC_Algorithms
from classical_communication_channel.communication_channel.role import Role

if __name__ == "__main__":
    alice_ip = input("Please enter the ip address where Alice will receive messages: ")
    alice_port = int(input("Please enter the port where Alice will receive messages: "))
    alice_connection_info = ConnectionInfo(alice_ip, alice_port)
    alice_id = "Alice"

    bob_ip = input("Please enter the ip address where Bob will receive messages: ")
    bob_port = int(input("Please enter the port where Bob will receive messages: "))
    bob_connection_info = ConnectionInfo(bob_ip, bob_port)
    bob_id = "Bob"

    bandwidth_limit_MBps = int(input("Please enter the simulated bandwidth (MB/s): "))
    inbox_capacity_MB = int(input("Please enter the inbox capacity (MB): "))
    outbox_capacity_MB = int(input("Please enter the outbox capacity (MB): "))

    shared_secret_key = input("Please enter the shared secret key (256 bits) if using "
                              "MAC (press \'enter\' if not using MAC): ")

    if shared_secret_key == "":
        alice = Role.get_instance(alice_connection_info, bandwidth_limit_MBps, inbox_capacity_MB, outbox_capacity_MB)
        bob = Role.get_instance(bob_connection_info, bandwidth_limit_MBps, inbox_capacity_MB, outbox_capacity_MB)
    else:
        shared_secret_key = bytearray(shared_secret_key.strip().encode())
        mac_configuration = MAC_Config(MAC_Algorithms.CMAC, shared_secret_key)
        alice = Role.get_instance(alice_connection_info, bandwidth_limit_MBps, inbox_capacity_MB, outbox_capacity_MB,
                                  mac_configuration)
        bob = Role.get_instance(bob_connection_info, bandwidth_limit_MBps, inbox_capacity_MB, outbox_capacity_MB,
                                mac_configuration)

    alice.peer_connection_info = bob_connection_info
    bob.peer_connection_info = alice_connection_info

    stop = False
    while not stop:
        message = input("Write a message to exchange between Alice and Bob."
                        "To stop this interaction, press \'enter\': ")
        if message.strip().lower() != "":
            alice.put_in_outbox(message.encode("UTF-8"))
            print("Waiting for Bob to get the message...")
            message_received = bob.get_from_inbox()
            while message_received is None:
                wait_synchronously(0.01)
                message_received = bob.get_from_inbox()
            message_received = message_received.decode("UTF-8")
            print(f"Bob received the following message: \'{message_received}\'")
        else:
            stop = True
    alice.clean()
    bob.clean()
