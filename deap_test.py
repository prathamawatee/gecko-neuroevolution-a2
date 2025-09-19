import random
from deap import base, creator, tools

# 1. Define the problem types (Fitness + Individual)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  
# weights=(1.0,) means we want to maximize the objective
creator.create("Individual", list, fitness=creator.FitnessMax)

# 2. Create toolbox and register initialization
toolbox = base.Toolbox()

# Say individuals have 5 floating-point numbers between 0 and 1
IND_SIZE = 5
toolbox.register("attr_float", random.random)  
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 3. Define evaluation function
def evaluate(individual):
    # For example: sum of the elements
    return (sum(individual),)  # return a tuple

toolbox.register("evaluate", evaluate)

# 4. Register genetic operators
toolbox.register("mate", tools.cxTwoPoint)  
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(42)  # for reproducibility

    # parameters
    pop_size = 20
    crossover_prob = 0.7
    mutation_prob = 0.2
    n_generations = 10

    # initialize population
    pop = toolbox.population(n=pop_size)

    # Evaluate the initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Begin the evolution
    for gen in range(1, n_generations + 1):
        # Select the next generation
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate individuals with invalid (stale) fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population
        pop[:] = offspring

        # Print stats
        fits = [ind.fitness.values[0] for ind in pop]
        print(f"Generation {gen}: max fitness = {max(fits):.3f}, avg fitness = {sum(fits)/len(fits):.3f}")

    # return final population
    return pop

if __name__ == "__main__":
    final_pop = main()
    # optionally, print best individual
    best = tools.selBest(final_pop, 1)[0]
    print("Best individual is:", best, "with fitness:", best.fitness.values[0])
