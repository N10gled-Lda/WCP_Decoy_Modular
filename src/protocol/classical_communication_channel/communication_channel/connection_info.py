class ConnectionInfo:
    def __init__(self, ip: str, port: int):
        self.ip = ip
        self.port = port

    def to_tuple(self) -> tuple[str, int]:
        return self.ip, self.port
