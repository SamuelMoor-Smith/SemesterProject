class Recycler:
    def __init__(self):
        self.supply = 0

    def add(self):
        self.supply += 1
    
    def subtract(self):
        self.supply -= 1