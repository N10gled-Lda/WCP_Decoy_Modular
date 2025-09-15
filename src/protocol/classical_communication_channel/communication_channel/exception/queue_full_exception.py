class QueueFullException(Exception):
    def __init__(self):
        super().__init__("Queue is full.")