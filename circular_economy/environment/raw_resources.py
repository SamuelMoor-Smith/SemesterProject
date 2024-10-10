class RawResources:
    def __init__(self, amount):
        self.supply = amount

    def subtract(self):
        self.supply -= 1