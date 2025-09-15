class RoleHasNoPeerException(Exception):
    def __init__(self):
        super().__init__("Role has no peer registered")
