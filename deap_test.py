import random
from deap import base, creator, tools

import mujoco
from mujoco import viewer

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

from deap_gecko import gecko_controller, show_qpos_history, evaluate_base, HISTORY
from deap_gecko_cma_es import cma_es

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

# 4. Register genetic operators
toolbox.register("mate", tools.cxTwoPoint)  
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def main(fitness_function, pop):
    toolbox.register("evaluate", fitness_function)

    # parameters
    pop_size = 10
    crossover_prob = 0.7
    mutation_prob = 0.2
    n_generations = 5

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

def perform_run(individual):

    mujoco.set_mjcb_control(None)  # reset controller hook

    # World + gecko
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0.1])

    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Hardcode a genome to test
    test_genome = individual

    # Attach controller built from genome
    mujoco.set_mjcb_control(gecko_controller(test_genome))

    # Launch interactive viewer to watch movement
    viewer.launch(model=model, data=data)

    show_qpos_history(HISTORY)

if __name__ == "__main__":
    random.seed(42)  # for reproducibility
    pop_size = 10
    # initialize population
    pop = toolbox.population(n=pop_size)
    print(pop[0])

    final_pop = main(evaluate, pop)
    # optionally, print best individual
    best = tools.selBest(final_pop, 1)[0]
    print("Run 1: Best individual is:", best, "with fitness:", best.fitness.values[0], "distance traveled: ", evaluate_base(best))
    # perform_run(best)

    best, fitness = cma_es(individual=pop[0], fitness_func = evaluate_base, sigma=0.5, population_size=10, generations=5)
    print("Run 2: Best individual is:", best, "with fitness:", fitness, "distance traveled: ", evaluate_base(best))
    # perform_run(best)

    final_pop = main(evaluate_base, pop)
    # optionally, print best individual
    best = tools.selBest(final_pop, 1)[0]
    print("Run 2: Best individual is:", best, "with fitness:", best.fitness.values[0], "distance traveled: ", evaluate_base(best))
    # perform_run(best)
