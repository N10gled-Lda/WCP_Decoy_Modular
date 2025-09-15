class ChannelFullException(Exception):
    def __init__(self):
        super().__init__("Channel is full and cannot accept new participants.")
