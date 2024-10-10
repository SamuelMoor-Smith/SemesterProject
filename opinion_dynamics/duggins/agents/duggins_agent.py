class DugginsAgent:
    def __init__(self, id, x, y, opinion, tolerance, conformity, susceptibility, r):
        self.id = id
        self.x = x
        self.y = y
        self.r = r
        self.opinion = opinion
        self.tolerance = tolerance
        self.conformity = conformity
        self.susceptibility = susceptibility