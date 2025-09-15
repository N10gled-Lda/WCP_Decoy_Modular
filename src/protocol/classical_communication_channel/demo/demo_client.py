import sys
sys.path.append('classical_communication_channel')
from communication_channel.connection_info import ConnectionInfo
from communication_channel.mac_config import MAC_Config, MAC_Algorithms
from communication_channel.role import Role

if __name__ == "__main__":
    # client_ip = input("Please enter your IP address: ")
    # client_port = int(input("Please enter your port: "))
    client_ip = 'localhost'
    client_port = 65434
    client_connection_info = ConnectionInfo(client_ip, client_port)

    bandwidth_limit_MBps = 100
    inbox_capacity_MB = 100
    outbox_capacity_MB = 100

    shared_secret_key = "IzetXlgAnY4oye56"

    if shared_secret_key == "":
        client = Role.get_instance(client_connection_info, bandwidth_limit_MBps, inbox_capacity_MB, outbox_capacity_MB)
    else:
        shared_secret_key = bytearray(shared_secret_key.strip().encode())
        mac_configuration = MAC_Config(MAC_Algorithms.CMAC, shared_secret_key)
        client = Role.get_instance(client_connection_info, bandwidth_limit_MBps, inbox_capacity_MB, outbox_capacity_MB,
                                  mac_configuration)

    try:
        while True:
            message_received = client.get_from_inbox()
            message_received = message_received.decode("UTF-8")
            print(f"Received the following message: \'{message_received}\'")
    except KeyboardInterrupt:
        client.clean()
