class MessageIncomplete(Exception):
    def __init__(self):
        super().__init__("Message is incomplete")
