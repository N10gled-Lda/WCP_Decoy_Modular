import sys
sys.path.append('classical_communication_channel')
from communication_channel.connection_info import ConnectionInfo
from communication_channel.mac_config import MAC_Config, MAC_Algorithms
from communication_channel.role import Role

if __name__ == "__main__":
    # server_ip = input("Please enter your IP address: ")
    # server_port = int(input("Please enter your port: "))
    server_ip = 'localhost'
    server_port = 65432
    server_connection_info = ConnectionInfo(server_ip, server_port)

    # client_ip = input("Please enter the other machine's ip address: ")
    # client_port = int(input("Please enter the other machine's port: "))
    client_ip = 'localhost'
    client_port = 65434
    client_connection_info = ConnectionInfo(client_ip, client_port)

    bandwidth_limit_MBps = 100
    inbox_capacity_MB = 100
    outbox_capacity_MB = 100

    shared_secret_key = "IzetXlgAnY4oye56"

    if shared_secret_key == "":
        server = Role.get_instance(server_connection_info, bandwidth_limit_MBps, inbox_capacity_MB, outbox_capacity_MB)
    else:
        shared_secret_key = bytearray(shared_secret_key.strip().encode())
        mac_configuration = MAC_Config(MAC_Algorithms.CMAC, shared_secret_key)
        server = Role.get_instance(server_connection_info, bandwidth_limit_MBps, inbox_capacity_MB, outbox_capacity_MB,
                                  mac_configuration)

    # Must be set before sending data to the client
    server.peer_connection_info = client_connection_info

    stop = False
    while not stop:
        message = input("Write a message to exchange with the other machine."
                        "To stop this interaction, press \'enter\': ")
        if message.strip().lower() != "":
            server.put_in_outbox(message.encode("UTF-8"))
        else:
            stop = True
    server.clean()
