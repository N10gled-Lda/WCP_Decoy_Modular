class PacketNotClassicalCommunicationPacketException(Exception):
    def __init__(self):
        super().__init__("Packet must be of type \'ClassicalCommunicationPacket\'")
