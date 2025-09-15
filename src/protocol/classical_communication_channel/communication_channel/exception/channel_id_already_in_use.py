class ChannelIdInUseException(Exception):
    def __init__(self):
        super().__init__(f'channel id is already in use')
