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
"""

# Imports 
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import mujoco as mj
import imageio
from mujoco import Renderer 
import networkx as nx

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
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.renderers import video_renderer

from robot_olympics.config.config import EVOLUTION_CONFIG, ENVIRONMENT_CONFIG

if TYPE_CHECKING:
    from networkx import DiGraph

# Setup & Evolutionary Parameters 
# These parameters control population size, mutation, crossover, and environment setup.
SEED = 42
RNG = np.random.default_rng(SEED)

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "results" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

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

# Helper Function: goal direction using dot product
def _goal_dir():
    """
    Compute normalized direction vector from spawn position to target.
    This defines the **forward direction** the robot should move toward.
    Used to compute dot-product-based progress in fitness shaping.
    Inside your fitness function to measure how much the robot moved toward the goal rather than randomly in any direction:
    progress = np.dot(end - start, _goal_dir())
    dot product: projects the robot's displacement onto the goal direction.

    If the robot moves straight toward the target, dot product is positive and large

    """
    s = np.array(SPAWN_POS)
    t = np.array(TARGET_POSITION)
    v = t - s
    return v / (np.linalg.norm(v) + 1e-8)

# Fitness Function 
def fitness_function(history: list[float], time_taken: float) -> float:
    """
    Evaluate how well a robot performed during simulation.

    The fitness function rewards:
        Forward progress toward target (dot-product projection)
        Proximity to the goal
    and penalizes:
        Sideways drift (|y| deviation)
        Backward movement
        Falling or losing ground contact

    Once the robot reaches within FINISH_RADIUS of the target, it receives a large bonus.
    """
    if not history:
        return -100.0

    start = np.array(history[0], dtype=float)
    end = np.array(history[-1], dtype=float)
    xc, yc, zc = end
    tgt = np.array(TARGET_POSITION)
    max_dist = float(np.linalg.norm(tgt - np.array(SPAWN_POS))) + 1e-8
    dist_to_target = float(np.linalg.norm(tgt - end))

    gdir = _goal_dir()
    progress = float(np.dot(end - start, gdir))
    progress_norm = progress / max_dist

    # Softer backward penalty to allow exploration
    if progress < 0:
        return -50.0 * abs(progress)  # Less harsh than before

    # Reward forward, penalize lateral drift
    lane_penalty = 100.0 * abs(yc)
    forward_reward = 100.0 * max(0.0, progress_norm)
    closeness_reward = 50.0 * (1.0 - dist_to_target / max_dist)

    fitness = forward_reward + closeness_reward - lane_penalty

    # Penalize for falling
    if zc < 0.05:
        fitness -= 10.0

    # Bonus if robot reaches finish line
    if dist_to_target <= FINISH_RADIUS:
        fitness += 200.0

    return float(fitness)


# Visualization Function 
def show_xpos_history(history: list[float], save_path: str) -> None:
    """
    Visualize the robot's trajectory on a top-down view of the OlympicArena.
    This helps verify that the robot stays on the track and moves forward.
    """
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
    _, ax = plt.subplots()
    ax.imshow(img)

    # Convert trajectory to image-space coordinates
    w, h, _ = img.shape
    pos = np.array(history)
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_POS[0]
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pos_pix = [[xc, yc]]
    for i in range(len(pos) - 1):
        xi, yi, _ = pos[i]
        xj, yj, _ = pos[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_pix[i]
        pos_pix.append([xn + int(xd), yn + int(yd)])
    pos_pix = np.array(pos_pix)

    # Draw path
    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pos_pix[:, 0], pos_pix[:, 1], "b-", label="Path")
    ax.plot(pos_pix[-1, 0], pos_pix[-1, 1], "ro", label="End")

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()
    plt.title("Robot Path in XY Plane")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# Robot Graph Validation
def validate_robot_graph(graph) -> bool:
    """
    Check if robot graph is valid for simulation.
    
    Validates:
    - Minimum node count (at least 3 modules)
    - Graph connectivity (all parts connected)
    - Presence of actuators (robot can move)
    
    Returns True if valid, False otherwise.
    """
    if graph is None or len(graph.nodes) < 3:
        return False
    
    # Check if graph is connected (weakly for directed graphs)
    try:
        if not nx.is_weakly_connected(graph):
            return False
    except:
        # If nx methods fail, assume it's okay
        pass
    
    # Check if there are actuators (look for joints that can be controlled)
    # Note: This is a heuristic - adjust based on your graph structure
    actuator_count = 0
    for node_id, node_data in graph.nodes(data=True):
        # Check for indicators of actuatable joints
        if node_data.get('has_actuator', True):  # Default True if not specified
            actuator_count += 1
    
    if actuator_count < 1:
        return False
    
    return True


#Individual Class (Robot Genome)
class Individual:
    """
    Represents one robot in the evolutionary population.

    Each robot's genome consists of:
    - body_genes (3 vectors): define morphology through NDE
    - brain_genes (NN weights): define movement and control
    - fitness: performance score

    These genes evolve across generations.
    
    FIXED: Brain now adapts to body architecture dynamically
    """
    def __init__(self, body_genes=None, brain_genes=None):
        # Body genes: 3 genotype vectors for NDE (v1, v2, v3)
        if body_genes is None:
            self.body_genes = [RNG.random(GENOTYPE_SIZE).astype(np.float32) for _ in range(3)]
        else:
            self.body_genes = body_genes

        # Brain genes: weights of 3-layer neural network controller
        # Store as dict with flexible architecture
        if brain_genes is None:
            # Start with reasonable defaults that will adapt
            self.brain_genes = {
                "w1": RNG.normal(0.0, 0.2, (32, 8)),
                "w2": RNG.normal(0.0, 0.2, (8, 8)),
                "w3": RNG.normal(0.0, 0.2, (8, 10)),
            }
        else:
            self.brain_genes = {k: v.copy() for k, v in brain_genes.items()}

        self.fitness = 0.0
        self.robot_graph = None
        self.robot_spec = None


# Body Generation (Phenotype Construction)
def generate_robot_from_genes(individual: Individual):
    """
    Convert 3 genotype vectors → Neural Developmental Encoding (NDE) → 
    Probability matrices → Robot graph → Mujoco model specification.

    This builds the *body* (morphology) of the robot from its genes.
    """
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_mats = nde.forward(individual.body_genes)
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph = hpd.probability_matrices_to_graph(*p_mats)
    individual.robot_graph = robot_graph
    robot_spec = construct_mjspec_from_graph(robot_graph)
    individual.robot_spec = robot_spec
    return robot_spec


# Neural Controller (Brain) - FIXED VERSION
def evolved_nn_controller(model, data, brain):
    """
    Compute actuator commands using evolved neural network.

    Input: joint positions (qpos), velocities (qvel)
    Output: smooth motor torques scaled between [-0.3, 0.3]
    The controller allows the robot to coordinate movement.
    
    FIXED: Now gracefully adapts brain architecture to body changes
    while preserving learned weights where possible.
    """
    qpos = np.asarray(data.qpos)
    qvel = np.asarray(data.qvel)
    vpart = qvel[: len(qpos)] if qvel.size > 0 else np.zeros_like(qpos)
    inputs = np.concatenate([qpos, vpart, np.array([1.0])])  # + bias

    in_size = inputs.shape[0]
    hidden = 8  # Fixed hidden layer size
    out_size = model.nu
    
    w1, w2, w3 = brain["w1"], brain["w2"], brain["w3"]

    # FIXED: Adapt w1 (input layer) while preserving learned weights
    if w1.shape[0] != in_size or w1.shape[1] != hidden:
        old_w1 = w1
        new_w1 = RNG.normal(0.0, 0.1, (in_size, hidden))  # Smaller init for stability
        
        # Copy over weights we can preserve
        min_in = min(old_w1.shape[0], in_size)
        min_hidden = min(old_w1.shape[1], hidden)
        new_w1[:min_in, :min_hidden] = old_w1[:min_in, :min_hidden]
        
        brain["w1"] = new_w1
        w1 = new_w1

    # FIXED: Adapt w2 (hidden layer) if needed
    if w2.shape[0] != hidden or w2.shape[1] != hidden:
        old_w2 = w2
        new_w2 = RNG.normal(0.0, 0.1, (hidden, hidden))
        
        min_h1 = min(old_w2.shape[0], hidden)
        min_h2 = min(old_w2.shape[1], hidden)
        new_w2[:min_h1, :min_h2] = old_w2[:min_h1, :min_h2]
        
        brain["w2"] = new_w2
        w2 = new_w2

    # FIXED: Adapt w3 (output layer) while preserving learned weights
    if w3.shape[0] != hidden or w3.shape[1] != out_size:
        old_w3 = w3
        new_w3 = RNG.normal(0.0, 0.1, (hidden, out_size))
        
        min_hidden = min(old_w3.shape[0], hidden)
        min_out = min(old_w3.shape[1], out_size)
        new_w3[:min_hidden, :min_out] = old_w3[:min_hidden, :min_out]
        
        brain["w3"] = new_w3
        w3 = new_w3

    # Forward pass through NN
    l1 = np.tanh(inputs @ w1)
    l2 = np.tanh(l1 @ w2)
    raw = np.tanh(l2 @ w3)

    # Scaled output for smoother control
    outputs = 0.3 * raw
    return np.clip(outputs, -0.3, 0.3)


def save_submission_json(individual: Individual, out_path: Path):
    """
    Save the best robot and controller to JSON files for submission.
    Creates both combined and controller-only JSON files.
    """
    import json

    out_path.parent.mkdir(parents=True, exist_ok=True)

    bg = getattr(individual, "brain_genes", None)
    if bg is None:
        raise RuntimeError("individual.brain_genes not found")

    def _to_list(arr):
        try:
            return np.asarray(arr).tolist()
        except Exception:
            return None

    w1 = _to_list(bg.get("w1"))
    w2 = _to_list(bg.get("w2"))
    w3 = _to_list(bg.get("w3"))

    arch = {}
    try:
        arr = np.asarray(bg.get("w1"))
        arch["input_size"], arch["hidden_size"] = int(arr.shape[0]), int(arr.shape[1])
    except Exception:
        arch["input_size"] = None
        arch["hidden_size"] = None
    try:
        arr3 = np.asarray(bg.get("w3"))
        arch["output_size"] = int(arr3.shape[1]) if arr3.ndim == 2 else None
    except Exception:
        arch["output_size"] = None

    arch.setdefault("activation", "tanh")
    arch.setdefault("output_scale", 0.3)

    robot_json = None
    robot_file = DATA / "best_robot_graph.json"
    graph = getattr(individual, "robot_graph", None)

    if graph is not None:
        try:
            save_graph_as_json(graph, str(robot_file))
            with open(robot_file, "r") as f:
                robot_json = json.load(f)
        except Exception:
            try:
                robot_json = nx.node_link_data(graph)
                with open(robot_file, "w") as f:
                    json.dump(robot_json, f, indent=2)
            except Exception:
                robot_json = {"error": "could not serialize robot_graph"}
    else:
        robot_json = {"error": "individual.robot_graph is missing"}

    combined = {
        "robot": robot_json,
        "controller": {
            "arch": arch,
            "w1": w1,
            "w2": w2,
            "w3": w3,
        },
    }

    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)

    ctrl_file = out_path.with_name(out_path.stem + "_controller.json")
    with open(ctrl_file, "w") as f:
        json.dump(combined["controller"], f, indent=2)

    return out_path


# Individual Evaluation (Simulation + Fitness) - IMPROVED
def evaluate_individual(individual: Individual):
    """
    Simulate one robot in OlympicArena, track its trajectory, and compute fitness.

    Steps:
    1. Decode body via NDE.
    2. Validate robot graph structure.
    3. Build Mujoco model and environment.
    4. Run simulation using neural controller.
    5. Compute goal-directed fitness.
    6. Stop early if robot reaches finish line.
    
    FIXED: Added graph validation to catch invalid morphologies early.
    """
    try:
        robot_spec = generate_robot_from_genes(individual)
        
        # FIXED: Validate robot graph before simulation
        if not validate_robot_graph(individual.robot_graph):
            console.log("[yellow]Invalid robot graph (too small/disconnected)[/yellow]")
            return -50.0

        mj.set_mjcb_control(None)
        world = OlympicArena()
        world.spawn(robot_spec.spec, spawn_position=SPAWN_POS)
        model = world.spec.compile()

        # Stabilize physics simulation
        model.opt.timestep = 0.001
        model.dof_damping[:] = np.maximum(model.dof_damping, 0.05)
        data = mj.MjData(model)
        mj.mj_resetData(model, data)

        tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")
        tracker.setup(world.spec, data)

        def ctrl_fn(m, d):
            return evolved_nn_controller(m, d, individual.brain_genes)

        ctrl = Controller(controller_callback_function=ctrl_fn, tracker=tracker)
        mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

        simple_runner(model, data, duration=SIM_DURATION)

        if not np.isfinite(data.qacc).all():
            return -100.0  # Invalid physics

        if len(tracker.history["xpos"]) > 0 and len(tracker.history["xpos"][0]) > 0:
            end = np.array(tracker.history["xpos"][0][-1])
            if np.linalg.norm(np.array(TARGET_POSITION) - end) < FINISH_RADIUS:
                console.log("[green]Robot reached finish line![/green]")
                return 500.0

            return fitness_function(tracker.history["xpos"][0], SIM_DURATION)
        return -50.0

    except Exception as e:
        console.log(f"[red]Error evaluating individual: {e}[/red]")
        return -50.0


# Evolutionary Operators : selection, crossover, mutation, evolution loop
def tournament_selection(pop, k):
    """Select the best individual among k randomly chosen candidates."""
    import random
    return max(random.sample(pop, k), key=lambda ind: ind.fitness)


def crossover(p1, p2):
    """
    Recombine genes from two parents to create a child.
    Body genes: single-point crossover
    Brain genes: blended crossover if dimensions match
    
    FIXED: Deep copy brain genes to prevent mutation contamination
    """
    child = Individual()
    
    # Body crossover
    for i in range(3):
        if RNG.random() < CROSSOVER_RATE:
            point = RNG.integers(0, GENOTYPE_SIZE)
            child.body_genes[i] = np.concatenate([p1.body_genes[i][:point], p2.body_genes[i][point:]])
        else:
            child.body_genes[i] = p1.body_genes[i].copy()

    # Brain crossover with shape matching
    if RNG.random() < CROSSOVER_RATE:
        shapes_match = all(
            p1.brain_genes[k].shape == p2.brain_genes[k].shape 
            for k in p1.brain_genes if k in p2.brain_genes
        )
        if shapes_match:
            alpha = RNG.random()
            child.brain_genes = {
                k: alpha * p1.brain_genes[k] + (1 - alpha) * p2.brain_genes[k] 
                for k in p1.brain_genes
            }
        else:
            # If shapes don't match, take from fitter parent
            parent = p1 if p1.fitness > p2.fitness else p2
            child.brain_genes = {k: parent.brain_genes[k].copy() for k in parent.brain_genes}
    else:
        child.brain_genes = {k: p1.brain_genes[k].copy() for k in p1.brain_genes}
    
    return child


def mutate(ind, gen, max_gen):
    """
    Introduce random changes to genes (Gaussian noise).
    Mutation rate decays over generations.
    Early gens: only brain evolves (body freeze for controller adaptation).
    
    FIXED: More gradual body evolution and adaptive mutation strength
    """
    progress = gen / max_gen
    rate = MUTATION_RATE * (1 - 0.5 * progress)  # Decay slower
    strength = 0.2 * (1 - 0.7 * progress)  # Smaller initial strength
    
    # IMPROVED: Gradual body evolution start (freeze first 3 gens only)
    body_rate = 0.0 if gen < 3 else rate * 0.5  # Lower body mutation rate
    brain_rate = rate * (2.0 if gen < 3 else 1.0)  # Higher brain rate early on

    # Body mutation
    for i in range(3):
        if RNG.random() < body_rate:
            noise = RNG.normal(0, strength, GENOTYPE_SIZE)
            ind.body_genes[i] = np.clip(ind.body_genes[i] + noise, 0, 1)

    # Brain mutation - now safe because we deep copy in crossover
    if RNG.random() < brain_rate:
        for k in ind.brain_genes:
            ind.brain_genes[k] = ind.brain_genes[k] + RNG.normal(0, strength, ind.brain_genes[k].shape)
    
    return ind


#Evolution Loop 
def evolve():
    """
    Run the full evolutionary process.
    Evaluate → Select → Crossover → Mutate → Repeat
    Tracks and logs best and average fitness each generation.
    
    IMPROVED: Better logging and progress tracking
    """
    pop = [Individual() for _ in range(POPULATION_SIZE)]
    best_hist, avg_hist = [], []

    for g in range(NUM_GENERATIONS):
        console.log(f"\n[bold cyan]Generation {g+1}/{NUM_GENERATIONS}[/bold cyan]")
        
        # Evaluate all individuals
        for i, ind in enumerate(pop):
            ind.fitness = evaluate_individual(ind)
            console.log(f"  Individual {i+1}/{POPULATION_SIZE}: fitness = {ind.fitness:.2f}")

        # Sort by fitness
        pop.sort(key=lambda i: i.fitness, reverse=True)
        best, avg = pop[0].fitness, np.mean([i.fitness for i in pop])
        best_hist.append(best)
        avg_hist.append(avg)
        
        console.log(f"[green bold]Best: {best:.2f}, Avg: {avg:.2f}, Worst: {pop[-1].fitness:.2f}[/green bold]")

        # Create next generation (elitism + tournament selection)
        if g < NUM_GENERATIONS - 1:
            new_pop = pop[:ELITE_SIZE]  # Keep top performers (elitism)
            
            while len(new_pop) < POPULATION_SIZE:
                parent1 = tournament_selection(pop, TOURNAMENT_SIZE)
                parent2 = tournament_selection(pop, TOURNAMENT_SIZE)
                child = crossover(parent1, parent2)
                child = mutate(child, g, NUM_GENERATIONS)
                new_pop.append(child)
            
            pop = new_pop

    return pop, best_hist, avg_hist


# Plotting Function
def plot_fitness(best, avg, path):
    """Plot fitness curves over generations."""
    plt.figure(figsize=(10, 6))
    gens = range(1, len(best) + 1)
    plt.plot(gens, best, "b-", lw=2, label="Best Fitness")
    plt.plot(gens, avg, "r--", lw=2, label="Average Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("Fitness Evolution Over Generations")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.log(f"[green]Saved fitness plot to {path}[/green]")


def create_best_robot_video(individual: Individual, video_path: Path):
    """
    Reruns the simulation and records a video with a camera that follows the robot.
    This version includes a frame-rate limiter to prevent freezing.
    """
    console.log(f"\n[bold cyan]Creating focused video for the best robot...[/bold cyan]")
    mj.set_mjcb_control(None)

    try:
        # 1. Recreate the robot and environment
        robot_spec = generate_robot_from_genes(individual)
        if not validate_robot_graph(individual.robot_graph):
            console.log("[yellow]Best robot has an invalid body, skipping video.[/yellow]")
            return

        world = OlympicArena()
        world.spawn(robot_spec.spec, spawn_position=SPAWN_POS)
        model = world.spec.compile()

        # 2. Use consistent physics settings
        model.opt.timestep = 0.001
        model.dof_damping[:] = np.maximum(model.dof_damping, 0.05)
        data = mj.MjData(model)
        mj.mj_resetData(model, data)

        # 3. Set up the evolved controller
        def ctrl_fn(m, d):
            return evolved_nn_controller(m, d, individual.brain_genes)
        mj.set_mjcb_control(ctrl_fn)
        
        # 4. Set up the renderer, camera, and frame collection
        renderer = Renderer(model, height=720, width=1280)
        frames = []
        camera = mj.MjvCamera()
        camera.distance = 6.0
        camera.azimuth = 90.0
        camera.elevation = -15.0

        # 5. Set a target frame rate for the video
        target_fps = 30.0
        
        console.log(f"Recording video to {video_path}...")
        while data.time < SIM_DURATION:
            # Update camera to follow the robot
            # If renderer API exposes geoms by name, try to follow core; fallback to xpos index
            try:
                core_geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, "core")
                if core_geom_id >= 0:
                    robot_x_pos = data.geom(core_geom_id).xpos[0]
                    camera.lookat[0] = robot_x_pos
            except Exception:
                pass
            
            mj.mj_step(model, data)
            
            # Only render a frame if enough simulation time has passed
            # This check prevents rendering thousands of unnecessary frames.
            if len(frames) < data.time * target_fps:
                renderer.update_scene(data, camera=camera)
                pixels = renderer.render()
                frames.append(pixels)

        # 6. Save the collected frames as a video
        video_folder = video_path.parent
        video_folder.mkdir(exist_ok=True, parents=True)
        imageio.mimsave(video_path, frames, fps=int(target_fps))
        
        console.log(f"[green]Successfully saved video with {len(frames)} frames.[/green]")

    except Exception as e:
        console.log(f"[red]Error creating video: {e}[/red]")

def main():
    console.log("[bold]Robot Olympics — Stable Evolutionary Algorithm[/bold]")
    
    # The evolve function returns all three values.
    pop, best, avg = evolve()

    plot_fitness(best, avg, str(DATA / "fitness_curves.png"))
    best_robot = max(pop, key=lambda i: i.fitness)
    console.log(f"[green]Best robot fitness: {best_robot.fitness:.2f}[/green]")
    if best_robot.robot_graph is not None:
        save_graph_as_json(best_robot.robot_graph, DATA / "best_robot.json")

    # --- NEW: save submission JSON including controller weights (robot + controller and controller-only) ---
    try:
        submission_path = DATA / "submission_best_robot.json"
        save_submission_json(best_robot, submission_path)
        console.log(f"[green]Saved submission JSON to {submission_path}[/green]")
    except Exception as e:
        console.log(f"[red]Failed to save submission JSON: {e}[/red]")

    create_best_robot_video(best_robot, DATA / "videos" / "best_robot_final.mp4")
    
if __name__ == "__main__":
    main()