import random

class Agent:
    def __init__(self, agent_id):
        self.id = agent_id
        self.value_score = random.random()  # Initial random score between 0 and 1
    
    def update_value(self, neighbors):
        """Update the value score based on the actions of neighboring agents."""
        neighbor_values = [neighbor.value_score for neighbor in neighbors]
        if neighbor_values:
            self.value_score = sum(neighbor_values) / len(neighbor_values)
    
    def step(self, neighbors):
        """Each agent performs its role's action and updates its value score."""
        self.update_value(neighbors)