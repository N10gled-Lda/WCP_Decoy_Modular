class OutboxFullException(Exception):
    def __init__(self):
        super().__init__("Outbox is at full capacity.")