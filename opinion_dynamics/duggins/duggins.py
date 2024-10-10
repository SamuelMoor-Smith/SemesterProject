import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from scipy.spatial.distance import cdist
from agents.duggins_agent import DugginsAgent

N_AGENTS = 500
N_STEPS = 500

def create_agents():
    agents = []
    for i in range(N_AGENTS):
        agents.append(
                    DugginsAgent(   
                        i,                  # agent_id
                        np.random.rand(),   # x
                        np.random.rand(),   # y
                        op_dist.rvs(),      # opinion
                        tol_dist.rvs(),     # tolerance
                        conf_dist.rvs(),    # conformity
                        sus_dist.rvs(),     # susceptibility
                        reach_dist.rvs()    # social reach
                    )
                )
        
    return agents


# Initial opinion, tolerance, conformity, susceptibility, and social reach are all drawn from normal distributions

# Opinion distribution
# Agents’ opinions, interpreted as beliefs on a single subjective issue, lie on a continuous 0−100 scale.
op_dist = uniform(0, 100)

# Tolerance distribution
tol_mean = 0.7
tol_std_dev = 0.1
tol_dist = norm(loc=tol_mean, scale=tol_std_dev)

# Conformity distribution
conf_mean = 0.5
conf_std_dev = 0.1
conf_dist = norm(loc=conf_mean, scale=conf_std_dev)

# Susceptibility distribution
sus_mean = 5.0
sus_std_dev = 0.5
sus_dist = norm(loc=sus_mean, scale=sus_std_dev)

# Social reach distribution
reach_mean = 0.1
reach_std_dev = 0.05
reach_dist = norm(loc=reach_mean, scale=reach_std_dev)

# Initialize the agents and type cast 
agents: list[DugginsAgent] = create_agents() 

# Extract the (x, y) positions of agents into a numpy array
positions = np.array([[agent.x, agent.y] for agent in agents])

# Calculate the pairwise distances between all agents
dist_matrix = cdist(positions, positions)

# Create the social network matrix based on individual social reach
social_network = np.zeros((N_AGENTS, N_AGENTS), dtype=bool)

for i in range(N_AGENTS):
    for j in range(N_AGENTS):
        if i != j and dist_matrix[i, j] < agents[i].r: # should i also have i connected to i?
            social_network[i, j] = True  # Agent i is connected to agent j if within social reach

def update_opinions(step):
    for i in range(N_AGENTS):
        # Initiate a dialogue with the social network of agent i
        neighbors = np.where(social_network[i])[0]
        if len(neighbors) == 0:
            continue
        
        # Agent i expresses its true opinion
        expressed_opinions = [agents[i].opinion]
        
        # Other agents in the network express opinions
        for j in neighbors:
            avg_opinion = np.mean(expressed_opinions)
            conformity_factor = agents[j].conformity * (avg_opinion - agents[j].opinion)
            expressed_opinions.append(agents[j].opinion + conformity_factor)
        
        # Agent i updates its opinion based on the dialogue
        weighted_influence = sum([1 - agents[i].tolerance * abs(e - agents[i].opinion) / 50 for e in expressed_opinions])
        influence = np.sum([e - agents[i].opinion for e in expressed_opinions])
        agents[i].opinion += agents[i].susceptibility * influence / weighted_influence
    
# Run the simulation
for step in range(N_STEPS):
    update_opinions(step)

# # Plot 1: Final opinion distribution
# plt.figure()
# plt.hist(opinions, bins=20, range=opinion_range, edgecolor='black')
# plt.title("Final Opinion Distribution")
# plt.xlabel("Opinion")
# plt.ylabel("Number of Agents")
# plt.show()

# # Plot 2: Opinion trajectory plot
# plt.figure()
# for i in range(num_agents):
#     plt.plot(range(num_steps), opinion_history[:, i], alpha=0.5)
# plt.title("Opinion Trajectories Over Time")
# plt.xlabel("Time Step")
# plt.ylabel("Opinion")
# plt.show()

# # Plot 3: Opinion diversity (standard deviation of opinions over time)
# opinion_diversity = np.std(opinion_history, axis=1)
# plt.figure()
# plt.plot(range(num_steps), opinion_diversity)
# plt.title("Opinion Diversity (Standard Deviation) Over Time")
# plt.xlabel("Time Step")
# plt.ylabel("Opinion Diversity")
# plt.show()

# # Plot 4: Spatial map of initial and final opinions
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.scatter(agents[:, 0], agents[:, 1], c=opinion_history[0], cmap='coolwarm', s=50, edgecolor='black')
# plt.colorbar(label="Initial Opinion")
# plt.title("Initial Opinion Distribution")

# plt.subplot(1, 2, 2)
# plt.scatter(agents[:, 0], agents[:, 1], c=opinions, cmap='coolwarm', s=50, edgecolor='black')
# plt.colorbar(label="Final Opinion")
# plt.title("Final Opinion Distribution")
# plt.show()
