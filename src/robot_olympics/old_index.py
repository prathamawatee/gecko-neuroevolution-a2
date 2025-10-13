"""
Assignment 3 – Robot Olympics: Co-evolution of Body and Brain
=============================================================

Goal:
-----
Develop a system that evolves both the **morphology (body)** and **controller (brain)** 
of a robot to navigate the *OlympicArena* as fast and as stably as possible.

Core Idea:
----------
- Use Evolutionary Computation to simulate natural selection.
- Randomly initialize a population of robots, evaluate them, and 
  iteratively evolve better ones based on fitness.

Implementation Summary:
------------------------
- **Body evolution**: 3 genotype vectors (v1, v2, v3) evolved via mutation/crossover → 
  Neural Developmental Encoder (NDE) → robot body graph (phenotype).
- **Brain evolution**: Neural network weights (w1, w2, w3) evolved → controller.
- **Fitness**: Encourages forward progress toward goal, penalizes sideways drift and backward motion.
- **Stability tweaks**: body-freeze for early gens, bounded control outputs, damping, smaller morphology.

Outcome:
--------
The system evolves robots that stay on track, walk forward, and reach the finish line
on mixed terrains (smooth, rugged, uphill).

FIXES APPLIED:
--------------
1. Deep copy for elite preservation
2. Fixed-size neural network with padding
3. Smooth blending crossover
4. Proper fitness curve plotting
5. Best robot path visualization per generation
"""

# Imports 
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import mujoco as mj
import copy

# ARIEL 
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.renderers import single_frame_renderer

from robot_olympics.config.config import EVOLUTION_CONFIG, ENVIRONMENT_CONFIG

if TYPE_CHECKING:
    from networkx import DiGraph

# Setup & Evolutionary Parameters 
SEED = 42
RNG = np.random.default_rng(SEED)

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "results" / SCRIPT_NAME
DATA.mkdir(exist_ok=True, parents=True)

# Create subdirectories
(DATA / "paths").mkdir(exist_ok=True, parents=True)
(DATA / "plots").mkdir(exist_ok=True, parents=True)

# Evolution Configuration
POPULATION_SIZE = EVOLUTION_CONFIG["population_size"]
NUM_GENERATIONS = EVOLUTION_CONFIG["generations"]
GENOTYPE_SIZE = EVOLUTION_CONFIG["genotype_size"]
TOURNAMENT_SIZE = EVOLUTION_CONFIG["tournament_size"]
MUTATION_RATE = EVOLUTION_CONFIG["mutation_rate"]
CROSSOVER_RATE = EVOLUTION_CONFIG["crossover_rate"]
ELITE_SIZE = EVOLUTION_CONFIG["elite_size"]

# Environment Configuration
SPAWN_POS = ENVIRONMENT_CONFIG["spawn_pos"]
TARGET_POSITION = ENVIRONMENT_CONFIG["target_pos"]
FINISH_RADIUS = ENVIRONMENT_CONFIG["finish_radius"]
SIM_DURATION = ENVIRONMENT_CONFIG["simulation_time"]
NUM_OF_MODULES = ENVIRONMENT_CONFIG["num_modules"]

# Fixed neural network architecture
STANDARD_INPUT_SIZE = 32
HIDDEN_SIZE = 8
STANDARD_OUTPUT_SIZE = 10


# Helper Function: goal direction using dot product
def _goal_dir():
    """
    Compute normalized direction vector from spawn position to target.
    """
    s = np.array(SPAWN_POS)
    t = np.array(TARGET_POSITION)
    v = t - s
    return v / (np.linalg.norm(v) + 1e-8)


# Fitness Function 
def fitness_function(history: list[float], time_taken: float) -> float:
    """
    Evaluate how well a robot performed during simulation.
    """
    if not history or len(history) < 2:
        return 0.0

    start = np.array(history[0][:3], dtype=float)
    end = np.array(history[-1][:3], dtype=float)
    xc, yc, zc = end
    
    tgt = np.array(TARGET_POSITION)  # [5, 0, 0.5] - the checker island
    spawn = np.array(SPAWN_POS)
    max_dist = float(np.linalg.norm(tgt - spawn)) + 1e-8
    dist_to_target = float(np.linalg.norm(tgt - end))

    # === PRIMARY REWARD: Forward Progress ===
    forward_progress = xc - spawn[0]  # How far forward from start
    
    # HUGE PENALTY for backward movement - this kills backward robots
    if forward_progress < 0:
        return -1000.0 * abs(forward_progress)  # Massive negative fitness
    
    # MAJOR REWARD for forward movement - exponential growth
    forward_reward = forward_progress * 300.0  # 300 points per meter forward
    if forward_progress > 2.0:
        forward_reward += (forward_progress - 2.0) * 200.0  # Extra bonus beyond 2m
    if forward_progress > 4.0:
        forward_reward += (forward_progress - 4.0) * 500.0  # Huge bonus near goal
    
    # === DIRECTION REWARD: Stay straight toward goal ===
    # Calculate how straight the path is toward target
    ideal_direction = (tgt - spawn) / np.linalg.norm(tgt - spawn)
    actual_direction = (end - spawn) / (np.linalg.norm(end - spawn) + 1e-8)
    directional_alignment = np.dot(ideal_direction, actual_direction)
    direction_reward = max(0.0, directional_alignment) * 200.0  # Reward straight movement
    
    # === LATERAL POSITION PENALTY: Stay centered (Y=0) ===
    # Penalize being off the straight path
    lateral_deviation = abs(yc)
    lateral_penalty = lateral_deviation * 100.0  # Linear penalty for drift
    if lateral_deviation > 1.0:  # Extra penalty for going way off course
        lateral_penalty += (lateral_deviation - 1.0) * 200.0
    
    # === PATH EFFICIENCY: Punish zigzag movement ===
    path_efficiency_bonus = 0.0
    if len(history) > 5:
        # Check if robot is moving efficiently (straight line vs actual path)
        total_path_length = 0.0
        backward_segments = 0
        
        for i in range(1, len(history)):
            if len(history[i]) >= 3 and len(history[i-1]) >= 3:
                prev_pos = np.array(history[i-1][:3])
                curr_pos = np.array(history[i][:3])
                segment = curr_pos - prev_pos
                total_path_length += np.linalg.norm(segment)
                
                # Count backward segments
                if segment[0] < -0.01:  # Moving backward in X
                    backward_segments += 1
        
        # Severe penalty for backward segments
        if backward_segments > 0:
            backward_penalty = backward_segments * 50.0
        else:
            backward_penalty = 0.0
            
        # Efficiency bonus for straight movement
        straight_line_distance = np.linalg.norm(end - start)
        if total_path_length > 0:
            efficiency = straight_line_distance / total_path_length
            path_efficiency_bonus = efficiency * 100.0
    else:
        backward_penalty = 0.0
    
    # === DISTANCE TO GOAL REWARD ===
    # Big reward for getting close to the checker island
    closeness_progress = (max_dist - dist_to_target) / max_dist
    closeness_reward = closeness_progress * 400.0
    
    # === SPEED REWARD ===
    # Reward faster completion
    speed_bonus = max(0.0, (SIM_DURATION - time_taken) / SIM_DURATION) * 100.0
    
    # === HEIGHT STABILITY ===
    # Small penalty for being too low (but not harsh)
    height_penalty = 0.0
    if zc < 0.05:
        height_penalty = 30.0
    
    # === MILESTONE BONUSES ===
    milestone_bonus = 0.0
    if forward_progress > 1.0:
        milestone_bonus += 200.0
    if forward_progress > 2.0:
        milestone_bonus += 300.0
    if forward_progress > 3.0:
        milestone_bonus += 500.0
    if forward_progress > 4.0:
        milestone_bonus += 800.0
    
    # === CALCULATE BASE FITNESS ===
    base_fitness = (forward_reward + direction_reward + closeness_reward + 
                   path_efficiency_bonus + speed_bonus + milestone_bonus - 
                   lateral_penalty - height_penalty - backward_penalty)
    
    # === ULTIMATE GOAL: FINISH LINE BONUS ===
    if dist_to_target <= FINISH_RADIUS:
        # MASSIVE reward for reaching the checker island
        finish_time_bonus = max(0.0, (SIM_DURATION - time_taken) / SIM_DURATION * 1000.0)
        finish_bonus = 5000.0 + finish_time_bonus  # 5000+ points for finishing!
        base_fitness += finish_bonus
        
        console.log(f"[bold green]🏁 CHECKER ISLAND REACHED! 🏁[/bold green]")
        console.log(f"[bold cyan]Finish bonus: {finish_bonus:.1f} points![/bold cyan]")
        console.log(f"[bold yellow]Time: {time_taken:.2f}s, Forward: {forward_progress:.2f}m[/bold yellow]")
    
    return max(0.0, base_fitness)
# Individual Class (Robot Genome)
class Individual:
    """
    Represents one robot in the evolutionary population.
    """
    def __init__(self, body_genes=None, brain_genes=None):
        if body_genes is None:
            self.body_genes = [RNG.random(GENOTYPE_SIZE).astype(np.float32) for _ in range(3)]
        else:
            self.body_genes = [g.copy() for g in body_genes]

        if brain_genes is None:
            self.brain_genes = {
                "w1": RNG.normal(0.0, 0.2, (STANDARD_INPUT_SIZE, HIDDEN_SIZE)),
                "w2": RNG.normal(0.0, 0.2, (HIDDEN_SIZE, HIDDEN_SIZE)),
                "w3": RNG.normal(0.0, 0.2, (HIDDEN_SIZE, STANDARD_OUTPUT_SIZE)),
            }
        else:
            self.brain_genes = {k: v.copy() for k, v in brain_genes.items()}

        self.fitness = 0.0
        self.trajectory = []  # Store trajectory for visualization


# Body Generation (Phenotype Construction)
def generate_robot_from_genes(individual: Individual):
    """
    Convert genes to robot morphology.
    """
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_mats = nde.forward(individual.body_genes)
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph = hpd.probability_matrices_to_graph(*p_mats)
    robot_spec = construct_mjspec_from_graph(robot_graph)
    return robot_spec, robot_graph


# Neural Controller (Brain)
def evolved_nn_controller(model, data, brain):
    """
    Compute actuator commands using evolved neural network.
    """
    qpos = np.asarray(data.qpos)
    qvel = np.asarray(data.qvel)
    vpart = qvel[: len(qpos)] if qvel.size > 0 else np.zeros_like(qpos)
    raw_inputs = np.concatenate([qpos, vpart, np.array([1.0])])

    # Standardize input size
    if len(raw_inputs) < STANDARD_INPUT_SIZE:
        inputs = np.pad(raw_inputs, (0, STANDARD_INPUT_SIZE - len(raw_inputs)), mode='constant')
    else:
        inputs = raw_inputs[:STANDARD_INPUT_SIZE]

    w1, w2, w3 = brain["w1"], brain["w2"], brain["w3"]
    l1 = np.tanh(inputs @ w1)
    l2 = np.tanh(l1 @ w2)
    raw_output = np.tanh(l2 @ w3)

    actual_output_size = model.nu
    if actual_output_size > 0:
        outputs = 0.3 * raw_output[:actual_output_size]
    else:
        outputs = np.array([])

    return np.clip(outputs, -0.3, 0.3)


# Individual Evaluation (Simulation + Fitness) 
def evaluate_individual(individual: Individual):
    """
    Simulate one robot and compute fitness.
    """
    try:
        robot_spec, robot_graph = generate_robot_from_genes(individual)
        
        # Reject obviously broken robots
        if robot_graph is None or len(robot_graph.nodes) < 2:
            return 1.0
            
        # Reject overly complex robots that are hard to control
        if len(robot_graph.nodes) > 20:
            return 5.0

        mj.set_mjcb_control(None)
        world = OlympicArena()
        world.spawn(robot_spec.spec, spawn_position=SPAWN_POS)
        model = world.spec.compile()

        # Optimize physics for stable forward movement
        model.opt.timestep = 0.002  # Slightly larger for stability
        model.dof_damping[:] = np.maximum(model.dof_damping, 0.1)  # Moderate damping
        data = mj.MjData(model)
        mj.mj_resetData(model, data)

        tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
        tracker.setup(world.spec, data)

        def ctrl_fn(m, d):
            return evolved_nn_controller(m, d, individual.brain_genes)

        ctrl = Controller(controller_callback_function=ctrl_fn, tracker=tracker)
        mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

        # Run simulation with progress monitoring
        start_time = data.time
        steps = 0
        max_steps = int(SIM_DURATION / model.opt.timestep)
        last_progress_check = 0.0
        
        while data.time - start_time < SIM_DURATION and steps < max_steps:
            mj.mj_step(model, data)
            steps += 1
            
            # Check for physics failures
            if not np.isfinite(data.qacc).all() or not np.isfinite(data.qpos).all():
                # Give partial credit for what was achieved
                if len(tracker.history.get("xpos", [])) > 0 and len(tracker.history["xpos"][0]) > 0:
                    partial_fitness = fitness_function(tracker.history["xpos"][0], data.time - start_time)
                    return max(10.0, partial_fitness * 0.3)
                return 5.0
            
            # Check progress every 500 steps
            if steps % 500 == 0 and len(tracker.history.get("xpos", [])) > 0:
                if len(tracker.history["xpos"][0]) > 0:
                    current_pos = np.array(tracker.history["xpos"][0][-1])
                    current_progress = current_pos[0] - SPAWN_POS[0]
                    
                    # Early finish detection
                    dist_to_goal = np.linalg.norm(np.array(TARGET_POSITION) - current_pos)
                    if dist_to_goal < FINISH_RADIUS:
                        elapsed_time = data.time - start_time
                        early_finish_bonus = max(0.0, 2000.0 * (1 - elapsed_time / SIM_DURATION))
                        console.log(f"[bold green]🎯 EARLY FINISH! Time: {elapsed_time:.2f}s[/bold green]")
                        return 5000.0 + early_finish_bonus
                    
                    # Terminate if robot is going severely backward
                    if current_progress < -1.0:  # More than 1m backward
                        return fitness_function(tracker.history["xpos"][0], data.time - start_time)
                    
                    last_progress_check = current_progress

        # Final evaluation
        if len(tracker.history.get("xpos", [])) > 0 and len(tracker.history["xpos"][0]) > 0:
            trajectory = tracker.history["xpos"][0]
            individual.trajectory = trajectory
            final_fitness = fitness_function(trajectory, data.time - start_time)
            
            # Log progress for monitoring
            final_pos = np.array(trajectory[-1])
            final_progress = final_pos[0] - SPAWN_POS[0]
            if final_progress > 0.5:  # Significant forward movement
                console.log(f"[green]Forward progress: {final_progress:.2f}m, Y-drift: {abs(final_pos[1]):.2f}m[/green]")
            
            return final_fitness
        
        return 2.0

    except Exception as e:
        console.log(f"[yellow]Evaluation error: {e}[/yellow]")
        return 1.0



# Evolutionary Operators
def tournament_selection(pop, k):
    """Select best among k random candidates."""
    import random
    return max(random.sample(pop, k), key=lambda ind: ind.fitness)


def crossover(p1, p2):
    """
    Smooth blending crossover.
    """
    child = Individual()
    
    for i in range(3):
        if RNG.random() < CROSSOVER_RATE:
            alpha = RNG.random()
            child.body_genes[i] = alpha * p1.body_genes[i] + (1 - alpha) * p2.body_genes[i]
        else:
            child.body_genes[i] = p1.body_genes[i].copy() if RNG.random() < 0.5 else p2.body_genes[i].copy()

    if RNG.random() < CROSSOVER_RATE:
        alpha = RNG.random()
        child.brain_genes = {
            k: alpha * p1.brain_genes[k] + (1 - alpha) * p2.brain_genes[k] 
            for k in p1.brain_genes
        }
    else:
        parent = p1 if RNG.random() < 0.5 else p2
        child.brain_genes = {k: parent.brain_genes[k].copy() for k in parent.brain_genes}
    
    return child


def mutate(ind, gen, max_gen):
    """
    Adaptive mutation.
    """
    progress = gen / max_gen
    
    # Aggressive exploration early, refinement later
    if gen < 5:
        base_rate = MUTATION_RATE * 2.0
        strength = 0.4
    elif gen < 15:
        base_rate = MUTATION_RATE * 1.5
        strength = 0.3
    else:
        base_rate = MUTATION_RATE * (1 - 0.3 * progress)
        strength = 0.2 * (1 - 0.4 * progress)
    
    # Body evolution strategy
    body_freeze_period = 10
    if gen < body_freeze_period:
        body_rate = 0.0  # Let controllers adapt to stable bodies first
        brain_rate = base_rate * 3.0  # Focus on brain evolution
    else:
        body_rate = base_rate * 0.4  # Gentle body changes
        brain_rate = base_rate * 1.2  # Continue brain optimization

    # Body mutation (conservative after freeze)
    for i in range(3):
        if RNG.random() < body_rate:
            noise = RNG.normal(0, strength * 0.3, GENOTYPE_SIZE)  # Gentle body changes
            ind.body_genes[i] = np.clip(ind.body_genes[i] + noise, 0, 1)

    # Brain mutation (more aggressive)
    if RNG.random() < brain_rate:
        for k in ind.brain_genes:
            noise = RNG.normal(0, strength, ind.brain_genes[k].shape)
            ind.brain_genes[k] += noise
    
    return ind


# Visualization Functions
def plot_robot_path(history, generation, fitness_val, save_path):
    """
    Plot robot trajectory on arena background.
    """
    try:
        # Create background image
        camera = mj.MjvCamera()
        camera.type = mj.mjtCamera.mjCAMERA_FREE
        camera.lookat = [2.5, 0, 0]
        camera.distance = 10
        camera.azimuth = 0
        camera.elevation = -90

        mj.set_mjcb_control(None)
        world = OlympicArena()
        model = world.spec.compile()
        data = mj.MjData(model)
        bg_path = str(DATA / "background.png")
        single_frame_renderer(model, data, camera=camera, save_path=bg_path, save=True)

        img = plt.imread(bg_path)
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)

        # Convert trajectory to image coordinates
        w, h, _ = img.shape
        pos = np.array(history)
        
        # Calibration points
        x0, y0 = int(h * 0.483), int(w * 0.815)
        xc, yc = int(h * 0.483), int(w * 0.9205)
        ym0, ymc = 0, SPAWN_POS[0]
        pixel_to_dist = -((ymc - ym0) / (yc - y0))
        
        pos_pix = [[xc, yc]]
        for i in range(len(pos) - 1):
            xi, yi, _ = pos[i][:3]
            xj, yj, _ = pos[i + 1][:3]
            xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
            xn, yn = pos_pix[i]
            pos_pix.append([xn + int(xd), yn + int(yd)])
        
        pos_pix = np.array(pos_pix)

        # Plot trajectory
        ax.plot(x0, y0, "k+", markersize=15, markeredgewidth=3, label="Origin")
        ax.plot(xc, yc, "go", markersize=12, label="Start")
        ax.plot(pos_pix[:, 0], pos_pix[:, 1], "b-", linewidth=2, label="Path", alpha=0.7)
        ax.plot(pos_pix[-1, 0], pos_pix[-1, 1], "ro", markersize=12, label="End")

        ax.set_xlabel("X Position (pixels)", fontsize=12)
        ax.set_ylabel("Y Position (pixels)", fontsize=12)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_title(f"Generation {generation} | Best Robot Path | Fitness: {fitness_val:.2f}", fontsize=14)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        console.log(f"[cyan]Saved path plot: {save_path}[/cyan]")
        
    except Exception as e:
        console.log(f"[yellow]Could not plot path: {e}[/yellow]")


def plot_fitness_curves(best_hist, avg_hist, save_path):
    """
    Plot fitness evolution over generations.
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 7))
        
        generations = list(range(1, len(best_hist) + 1))
        
        # Plot best fitness
        ax.plot(generations, best_hist, 'b-o', linewidth=2, markersize=6, 
                label='Best Fitness', markerfacecolor='blue', markeredgecolor='darkblue')
        
        # Plot average fitness
        ax.plot(generations, avg_hist, 'r--s', linewidth=2, markersize=5,
                label='Average Fitness', markerfacecolor='red', markeredgecolor='darkred', alpha=0.7)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # Labels and title
        ax.set_xlabel("Generation", fontsize=14, fontweight='bold')
        ax.set_ylabel("Fitness", fontsize=14, fontweight='bold')
        ax.set_title("Fitness Evolution Over Generations", fontsize=16, fontweight='bold')
        
        # Legend
        ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
        
        # Add annotations for best and final fitness
        if len(best_hist) > 0:
            max_fitness = max(best_hist)
            max_gen = best_hist.index(max_fitness) + 1
            ax.annotate(f'Peak: {max_fitness:.2f}',
                       xy=(max_gen, max_fitness),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            final_fitness = best_hist[-1]
            ax.annotate(f'Final: {final_fitness:.2f}',
                       xy=(len(best_hist), final_fitness),
                       xytext=(-80, -20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Set axis limits with padding
        if len(best_hist) > 0:
            y_min = min(min(best_hist), min(avg_hist))
            y_max = max(max(best_hist), max(avg_hist))
            y_padding = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - y_padding, y_max + y_padding)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        console.log(f"[green]✓ Saved fitness curves: {save_path}[/green]")
        
    except Exception as e:
        console.log(f"[red]Error plotting fitness curves: {e}[/red]")


# Evolution Loop 
def evolve():
    """
    Run evolutionary algorithm with visualization.
    """
    pop = [Individual() for _ in range(POPULATION_SIZE)]
    best_hist, avg_hist = [], []
    
    global_best = None
    global_best_fitness = -float('inf')

    for g in range(NUM_GENERATIONS):
        console.log(f"\n[bold cyan]═══ Generation {g+1}/{NUM_GENERATIONS} ═══[/bold cyan]")
        
        # Evaluate population
        for i, ind in enumerate(pop):
            ind.fitness = evaluate_individual(ind)
            console.log(f"  Individual {i+1}/{POPULATION_SIZE}: fitness = {ind.fitness:.2f}")
            
            if ind.fitness > global_best_fitness:
                global_best_fitness = ind.fitness
                global_best = copy.deepcopy(ind)

        # Sort by fitness
        pop.sort(key=lambda i: i.fitness, reverse=True)
        
        best_fitness = pop[0].fitness
        avg_fitness = np.mean([i.fitness for i in pop])
        best_hist.append(best_fitness)
        avg_hist.append(avg_fitness)
        
        console.log(f"[green]Best: {best_fitness:.2f} | Avg: {avg_fitness:.2f} | Global Best: {global_best_fitness:.2f}[/green]")
        
        # Visualize best robot path
        if pop[0].trajectory and len(pop[0].trajectory) > 0:
            path_plot = DATA / "paths" / f"generation_{g+1:03d}_path.png"
            plot_robot_path(pop[0].trajectory, g+1, best_fitness, path_plot)
        
        # Check for winner
        if best_fitness >= 500:
            console.log(f"[bold green]🏆 Winner found in generation {g+1}![/bold green]")

        # Generate next generation
        if g < NUM_GENERATIONS - 1:
            new_pop = [copy.deepcopy(ind) for ind in pop[:ELITE_SIZE]]
            console.log(f"  Preserved {ELITE_SIZE} elites")
            
            while len(new_pop) < POPULATION_SIZE:
                parent1 = tournament_selection(pop, TOURNAMENT_SIZE)
                parent2 = tournament_selection(pop, TOURNAMENT_SIZE)
                child = crossover(parent1, parent2)
                child = mutate(child, g, NUM_GENERATIONS)
                new_pop.append(child)
            
            pop = new_pop

    # Return global best
    if global_best is not None and global_best.fitness > pop[0].fitness:
        console.log(f"[yellow]Using global best (fitness: {global_best_fitness:.2f})[/yellow]")
        pop[0] = global_best

    return pop, best_hist, avg_hist


def main():
    """
    Main execution function.
    """
    console.log("[bold blue]═══════════════════════════════════════[/bold blue]")
    console.log("[bold blue]   Robot Olympics - Evolution System   [/bold blue]")
    console.log("[bold blue]═══════════════════════════════════════[/bold blue]")
    console.log(f"Population: {POPULATION_SIZE} | Generations: {NUM_GENERATIONS}")
    console.log(f"Target: {TARGET_POSITION} | Finish Radius: {FINISH_RADIUS}")
    console.log("")
    
    # Run evolution
    pop, best_hist, avg_hist = evolve()

    # Plot fitness curves
    fitness_plot_path = DATA / "plots" / "fitness_evolution.png"
    plot_fitness_curves(best_hist, avg_hist, fitness_plot_path)
    
    # Get best robot
    best_robot = max(pop, key=lambda i: i.fitness)
    console.log(f"\n[bold green]🏆 Final Best Robot Fitness: {best_robot.fitness:.2f}[/bold green]")
    
    # Save best robot graph
    try:
        robot_spec, robot_graph = generate_robot_from_genes(best_robot)
        if robot_graph is not None:
            graph_path = DATA / "best_robot_graph.json"
            save_graph_as_json(robot_graph, graph_path)
            console.log(f"[green]✓ Saved robot graph: {graph_path}[/green]")
    except Exception as e:
        console.log(f"[yellow]Could not save robot graph: {e}[/yellow]")
    
    # Final path plot
    if best_robot.trajectory and len(best_robot.trajectory) > 0:
        final_path_plot = DATA / "plots" / "final_best_robot_path.png"
        plot_robot_path(best_robot.trajectory, NUM_GENERATIONS, best_robot.fitness, final_path_plot)
    
    console.log("\n[bold green]═══════════════════════════════════════[/bold green]")
    console.log("[bold green]   Evolution Complete! ✓               [/bold green]")
    console.log("[bold green]═══════════════════════════════════════[/bold green]")
    console.log(f"Results saved to: {DATA}")
    console.log(f"  - Fitness plot: {fitness_plot_path}")
    console.log(f"  - Path plots: {DATA / 'paths'}")


if __name__ == "__main__":
    main()