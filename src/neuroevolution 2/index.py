"""
Neuroevolution Framework for Gecko Robot

This script demonstrates how to use evolutionary algorithms (EAs) to train 
a neural network controller for a simulated Gecko robot. Instead of using 
backpropagation + gradient descent, we directly evolve the neural network weights.

The workflow is:
1. Define a neural network structure (the "brain").
2. Simulate the Gecko robot controlled by this brain.
3. Measure fitness = how far the Gecko walks.
4. Apply evolutionary operators (selection, crossover, mutation).
5. Repeat for multiple generations → better walking strategies emerge.

We compare:
- Baseline (random controllers, no evolution).
- Small network.
- Wide network.
"""

import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
from deap import algorithms, base, creator, tools
import random
from datetime import datetime
from functools import partial
from neuroevolution.config.experiments_config import experiments



# Local simulation libraries
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# Local result management (ensures everything goes into results/)
import sys, os

from neuroevolution.utils.io import save_best_controller, save_plot, save_log



# Neural Network
class NeuralNetwork:
    """
    Simple feedforward neural network (1 hidden layer).
    
    - Input: robot state (position + joint angles).
    - Hidden: tanh activation.
    - Output: joint control signals (mapped to [-pi/2, pi/2]).
    
    Instead of learning via gradient descent, weights are represented as a 
    flat genome vector and evolved using EA.
    """

    def __init__(self, input_size, hidden_size, output_size):
        # Architecture definition
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Number of parameters (weights + biases) to evolve
        self.weights_ih = input_size * hidden_size
        self.bias_h = hidden_size
        self.weights_ho = hidden_size * output_size
        self.bias_o = output_size
        self.total_params = self.weights_ih + self.bias_h + self.weights_ho + self.bias_o

    def forward(self, inputs, weights):
        """
        Run a forward pass through the network.
        
        Parameters:
            inputs (ndarray): robot state input.
            weights (list): flattened genome representing NN weights.
        Returns:
            output (ndarray): control signals for robot joints.
        """
        idx = 0
        # Input → Hidden
        W_ih = np.array(weights[idx:idx + self.weights_ih]).reshape(self.input_size, self.hidden_size)
        idx += self.weights_ih
        b_h = np.array(weights[idx:idx + self.bias_h])
        idx += self.bias_h

        # Hidden → Output
        W_ho = np.array(weights[idx:idx + self.weights_ho]).reshape(self.hidden_size, self.output_size)
        idx += self.weights_ho
        b_o = np.array(weights[idx:idx + self.bias_o])

        # Activations
        hidden = np.tanh(np.dot(inputs, W_ih) + b_h)
        output = np.tanh(np.dot(hidden, W_ho) + b_o)
        return output



# Gecko Controller

class GeckoController:
    """
    Wraps a neural network into a robot controller.
    
    - Reads robot state (position + joint angles).
    - Normalizes/pads input into expected NN format.
    - Runs NN to get control signals.
    - Applies signals to robot actuators.
    - Optionally logs robot positions for later fitness evaluation.
    """

    def __init__(self, network, weights, track_history=True):
        self.network = network
        self.weights = weights
        self.track_history = track_history
        self.history = []  # store positions over time

    def control(self, model, data, to_track):
        # Collect robot state
        robot_pos = to_track[0].xpos.copy() if to_track else np.zeros(3)
        joint_angles = data.qpos.copy() if len(data.qpos) > 0 else np.zeros(model.nu)

        # Input vector = [x, y, first 8 joint angles]
        inputs = np.concatenate([
            robot_pos[:2],
            joint_angles[:min(len(joint_angles), 8)]
        ])

        # Pad/truncate to match NN input size
        if len(inputs) < self.network.input_size:
            inputs = np.pad(inputs, (0, self.network.input_size - len(inputs)))
        else:
            inputs = inputs[:self.network.input_size]

        # NN forward pass
        output = self.network.forward(inputs, self.weights)

        # Scale outputs to valid joint control range
        scaled_output = output * (np.pi / 2)

        # Apply control to MuJoCo
        if len(scaled_output) >= model.nu:
            data.ctrl[:] = np.clip(scaled_output[:model.nu], -np.pi/2, np.pi/2)
        else:
            data.ctrl[:len(scaled_output)] = np.clip(scaled_output, -np.pi/2, np.pi/2)

        # Track position for later distance calculation
        if self.track_history and to_track:
            self.history.append(robot_pos.copy())



# Fitness Evaluation

def evaluate_individual(individual, network_config, simulation_time=8.0, verbose=False):
    """
    Simulates one Gecko robot controlled by a given genome (set of weights).
    
    Returns:
        fitness (tuple): distance walked in simulation (higher = better).
    """
    try:
        # Reset simulation
        mujoco.set_mjcb_control(None)
        world = SimpleFlatWorld()
        gecko_core = gecko()
        world.spawn(gecko_core.spec, spawn_position=[0, 0, 0.1])  # small lift from ground

        model = world.spec.compile()
        data = mujoco.MjData(model)

        # Track the core body to measure distance
        geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
        to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

        # Build NN controller
        network = NeuralNetwork(**network_config)
        controller = GeckoController(network, individual, track_history=True)
        mujoco.set_mjcb_control(lambda m, d: controller.control(m, d, to_track))

        # Run simulation loop
        start_time = data.time
        while data.time - start_time < simulation_time:
            mujoco.mj_step(model, data)

        # Calculate distance traveled (fitness)
        if controller.history:
            positions = np.array(controller.history)
            start_pos, end_pos = positions[0][:2], positions[-1][:2]
            distance = np.linalg.norm(end_pos - start_pos)

            # Add small bonus for consistent movement direction
            if len(positions) > 10:
                mid_pos = positions[len(positions)//2][:2]
                consistent_movement = np.linalg.norm(end_pos - mid_pos)
                distance += consistent_movement * 0.1

            return (distance,)
        else:
            return (0.0,)
    except Exception as e:
        if verbose:
            print(f"Evaluation error: {e}")
        return (0.0,)



# Baseline (Random Movement)

def random_baseline(network_config, simulation_time=8.0, trials=40, seed=None):
    """
    Random movement baseline.
    
    - Creates many random genomes (random weights).
    - Runs simulation for each.
    - Returns average, max, min fitness.
    
    This is important: gives us a reference to check if evolution
    actually finds controllers better than random chance.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    net = NeuralNetwork(**network_config)
    fitnesses = []

    for _ in range(trials):
        genome = [random.uniform(-2.0, 2.0) for _ in range(net.total_params)]
        fit, = evaluate_individual(genome, network_config, simulation_time=simulation_time)
        fitnesses.append(fit)

    return {
        'generations': [0],  # baseline has no generations
        'avg_fitness': [np.mean(fitnesses)],
        'max_fitness': [np.max(fitnesses)],
        'min_fitness': [np.min(fitnesses)],
        'std_fitness': [np.std(fitnesses)],
        'best_individual': None,
        'best_fitness': np.max(fitnesses)
    }


# Evolutionary Algorithm

def run_evolution(network_config, ea_params, experiment_name, run_number):
    """
    Runs a full evolutionary optimization:
    
    1. Initialize random population of genomes.
    2. Evaluate fitness (distance walked).
    3. Select parents (tournament).
    4. Apply crossover + mutation.
    5. Repeat for N generations.
    
    Returns full logbook of evolution (avg, max, min fitness per gen).
    """
    print(f"\nStarting {experiment_name} - Run {run_number}")

    # Define fitness and individual representation
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Toolbox = EA operators (selection, mutation, crossover, evaluation)
    toolbox = base.Toolbox()
    network = NeuralNetwork(**network_config)

    toolbox.register("attr_float", random.uniform, -2.0, 2.0)  # genes = floats
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_float, n=network.total_params)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", partial(evaluate_individual,
                                         network_config=network_config,
                                         simulation_time=ea_params['simulation_time']))
    toolbox.register("mate", tools.cxBlend, alpha=ea_params['crossover_alpha'])
    toolbox.register("mutate", tools.mutGaussian,
                     mu=0, sigma=ea_params['mutation_sigma'],
                     indpb=ea_params['mutation_indpb'])
    toolbox.register("select", tools.selTournament, tournsize=ea_params['tournament_size'])

    # Track statistics per generation
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)
    stats.register("std", np.std)

    # Create population and hall of fame
    population = toolbox.population(n=ea_params['population_size'])
    hof = tools.HallOfFame(1)

    # Run standard EA loop (from DEAP)
    population, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=ea_params['crossover_prob'],
        mutpb=ea_params['mutation_prob'],
        ngen=ea_params['generations'],
        stats=stats, halloffame=hof,
        verbose=True
    )

    # Pack results
    fitness_data = {
        'generations': list(range(ea_params['generations'] + 1)),
        'avg_fitness': [record['avg'] for record in logbook],
        'max_fitness': [record['max'] for record in logbook],
        'min_fitness': [record['min'] for record in logbook],
        'std_fitness': [record['std'] for record in logbook],
        'best_individual': hof[0],
        'best_fitness': hof[0].fitness.values[0]
    }

    # Cleanup DEAP creators (prevents duplicate errors)
    del creator.FitnessMax
    del creator.Individual
    return fitness_data


# Plotting

def plot_results(results_dict):
    """
    Plots average & max fitness curves across generations for each experiment.
    - Shaded regions show standard deviation across multiple runs.
    - Useful for visually comparing experiments.
    - Displays final fitness values on top of the chart.
    """
    num_experiments = len(results_dict)
    fig, axes = plt.subplots(1, num_experiments, figsize=(6 * num_experiments, 6))

    if num_experiments == 1:
        axes = [axes]

    colors = ['blue', 'red', 'green', 'purple', 'orange']

    for idx, (exp_name, runs_data) in enumerate(results_dict.items()):
        ax = axes[idx]

        if "Baseline" in exp_name:
            # Collect average fitness from all runs
            all_avg = [run['avg_fitness'][0] for run in runs_data]

            # Boxplot for average fitness only
            ax.boxplot(all_avg, labels=['Avg Fitness'])
            ax.set_title(f'{exp_name} (Baseline)')
            ax.set_ylabel('Fitness')
            ax.grid(True, alpha=0.3)

            # Annotate final fitness stats at the top
            textstr = f"Mean: {np.mean(all_avg):.2f}\nMax: {np.max(all_avg):.2f}"
            ax.text(0.5, 1.05, textstr,
                    transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

        else:
            # Evolutionary: line plot as before
            all_avg_fitness = [run['avg_fitness'] for run in runs_data]
            all_max_fitness = [run['max_fitness'] for run in runs_data]

            avg_array = np.array(all_avg_fitness)
            max_array = np.array(all_max_fitness)

            generations = runs_data[0]['generations']

            mean_avg = np.mean(avg_array, axis=0)
            std_avg = np.std(avg_array, axis=0)
            mean_max = np.mean(max_array, axis=0)
            std_max = np.std(max_array, axis=0)

            ax.plot(generations, mean_avg, color=colors[idx % len(colors)], linestyle='--',
                    label='Avg Fitness', alpha=0.8)
            ax.fill_between(generations, mean_avg - std_avg, mean_avg + std_avg,
                            color=colors[idx % len(colors)], alpha=0.2)

            ax.plot(generations, mean_max, color=colors[idx % len(colors)], linestyle='-',
                    label='Max Fitness', linewidth=2)
            ax.fill_between(generations, mean_max - std_max, mean_max + std_max,
                            color=colors[idx % len(colors)], alpha=0.3)

            ax.set_title(f'{exp_name}')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Annotate final fitness values at the top
            textstr = f"Final Avg: {mean_avg[-1]:.2f}\nFinal Max: {mean_max[-1]:.2f}"
            ax.text(0.5, 1.05, textstr,
                    transform=ax.transAxes, ha='center', va='bottom',
                    fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    plt.tight_layout()
    return fig



# Main
def main():
    """
    Main entry point:
    - Runs all experiments (baseline + evolutionary ones).
    - Repeats each experiment multiple times.
    - Saves logs, controllers, and plots to results/.
    """
    all_results = {}
    num_runs = 3

    for exp_name, config in experiments.items():
        print(f"Running {exp_name}")
        
        runs_data = []
        best_overall = None
        best_fitness = -float('inf')

        for run in range(num_runs):
            if "Baseline" in exp_name:
                result = random_baseline(
                    config["network_config"],
                    simulation_time=config["ea_params"]["simulation_time"],
                    trials=40,
                    seed=run
                )
            else:
                result = run_evolution(
                    config["network_config"],
                    config["ea_params"],
                    exp_name,
                    run + 1
                )

            runs_data.append(result)
            save_log(result, exp_name, run + 1)   # save raw data per run

            # Track best controller across runs
            if result['best_fitness'] > best_fitness and result['best_individual'] is not None:
                best_fitness = result['best_fitness']
                best_overall = result['best_individual']

        all_results[exp_name] = runs_data

        # Save the best controller weights
        if best_overall is not None:
            save_best_controller(best_overall, config["network_config"], exp_name)

    # Save plot of all experiments
    print("\nGenerating plots...")
    fig = plot_results(all_results)
    save_plot(fig, "all_experiments")

    print("\nAll experiments completed!")


if __name__ == "__main__":
    main()
