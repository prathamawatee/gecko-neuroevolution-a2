#Configuration

# Evolutionary Algorithm (EA) parameters
EVOLUTION_CONFIG = {
    "population_size": 20,      # Number of robots per generation
    "generations": 30,          # Total number of generations
    "genotype_size": 64,        # Length of each body gene vector
    "tournament_size": 3,       # Number of individuals in tournament selection
    "mutation_rate": 0.15,      # Probability of mutation per individual
    "crossover_rate": 0.7,      # Probability of crossover
    "elite_size": 2             # Number of top individuals carried forward (elitism)
}

# Environment setup for OlympicArena
ENVIRONMENT_CONFIG = {
    "spawn_pos": [-0.8, 0, 0.1],      # Starting position of robots
    "target_pos": [5, 0, 0.5],        # Finish line target position
    "finish_radius": 0.2,             # Distance threshold to count as finished
    "simulation_time": 15,            # Duration of each simulation in seconds
    "num_modules": 12                 # Number of modules in the robot body
}