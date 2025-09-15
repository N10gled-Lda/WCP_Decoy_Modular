class RoleAlreadyHasPeerException(Exception):
    def __init__(self):
        super().__init__("Role already has a peer registered")
