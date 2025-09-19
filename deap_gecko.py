# Third-party libraries
import numpy as np
from deap import base, creator, tools

import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko


GENOME_SIZE = 8
#tells DEAP we want to maximize fitness (distance traveled). Define Fitness and Individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Toolbox for genetic operations
"""
attr_float: one random gene = a float between -1 and 1.

individual: builds an Individual by repeating attr_float GENOME_SIZE times.

population: builds a list of Individuals.
"""
toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.rand)  
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=GENOME_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# Keep track of data / history
HISTORY = []

# This takes an individual from DEAP (a list of floats = genome = weights).

#Returns a function (controller) that MuJoCo can use.

def gecko_controller(individual):
    #This is the function MuJoCo calls every timestep.
    def controller(model, data): # to_track
        t = data.time
        #Loop over joints
        for i in range(model.nu): 
            #Pick a weight from the genome (loop if fewer weights than joints)
            w = individual[i % len(individual)] 
            #Generate movement for joint i
            #Here we use a sine wave with frequency based on the weight
            #Scale sine wave to be between -pi/2 and pi/2
            #Set the control value for joint i
            data.ctrl[i] = np.clip(np.sin(t * w) * (np.pi/2), -np.pi/2, np.pi/2)
    return controller

def evaluate(individual):
    """Evaluate the fitness of an individual genome.

    The fitness is defined as the distance traveled in the x-y plane
    from the starting position after a fixed duration of simulation.

    Parameters
    ----------
    individual : list of float
        The genome representing the controller weights.

    Returns
    -------
    tuple
        A tuple containing the fitness value (distance traveled).
    """
    mujoco.set_mjcb_control(None)  # always reset first

    # Setup world + gecko
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0.1])

    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Attach controller built from genome
    mujoco.set_mjcb_control(gecko_controller(individual))

    # Run simulation for a fixed duration
    sim_duration = 10.0  # seconds
    sim_timestep = model.opt.timestep
    n_steps = int(sim_duration / sim_timestep)

    # Initialise history tracking
    HISTORY.clear()
    
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        # Track position of the robot's core geom (assumed to be the first geom)
        HISTORY.append(data.qpos[:3].copy())  # Store x, y, z positions

    # Calculate distance traveled in x-y plane from start to end
    start_pos = np.array(HISTORY[0][:2])  # Initial x, y position
    end_pos = np.array(HISTORY[-1][:2])   # Final x, y position
    distance_traveled = np.linalg.norm(end_pos - start_pos)

    return (distance_traveled,)
# Register EA components
toolbox.register("evaluate", evaluate)         
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)        


def show_qpos_history(history:list):
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')
    
    # Add labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position') 
    plt.title('Robot Path in XY Plane')
    plt.legend()
    plt.grid(True)
    
    # Set equal aspect ratio and center at (0,0)
    plt.axis('equal')
    max_range = max(abs(pos_data).max(), 0.3)  # At least 1.0 to avoid empty plots
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    
    plt.show()

def main():
    mujoco.set_mjcb_control(None)  # reset controller hook

    # World + gecko
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0.1])

    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Hardcode a genome to test
    test_genome = [0.5, 1.0, 2.0, 1.5, 0.8, 1.2, 0.6, 1.8]

    # Attach controller built from genome
    mujoco.set_mjcb_control(gecko_controller(test_genome))

    # Launch interactive viewer to watch movement
    viewer.launch(model=model, data=data)
    # # Initialise controller to controller to None, always in the beginning.
    # mujoco.set_mjcb_control(None) # DO NOT REMOVE
    
    # # Initialise world
    # # Import environments from ariel.simulation.environments
    # world = SimpleFlatWorld()

    # # Initialise robot body
    # # YOU MUST USE THE GECKO BODY
    # gecko_core = gecko()     # DO NOT CHANGE

    # # Spawn robot in the world
    # # Check docstring for spawn conditions
    # world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    
    # # Generate the model and data
    # # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    # model = world.spec.compile()
    # data = mujoco.MjData(model) # type: ignore

    # # Initialise data tracking
    # # to_track is automatically updated every time step
    # # You do not need to touch it.
    # geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    # to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    # # Set the control callback function
    # # This is called every time step to get the next action. 
    # mujoco.set_mjcb_control(lambda m,d: random_move(m, d, to_track))

    # # This opens a viewer window and runs the simulation with the controller you defined
    # # If mujoco.set_mjcb_control(None), then you can control the limbs yourself.
    # viewer.launch(
    #     model=model,  # type: ignore
    #     data=data,
    # )

    # show_qpos_history(HISTORY)
    # # If you want to record a video of your simulation, you can use the video renderer.

    # # # Non-default VideoRecorder options
    # # PATH_TO_VIDEO_FOLDER = "./__videos__"
    # # video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)

    # # # Render with video recorder
    # # video_renderer(
    # #     model,
    # #     data,
    # #     duration=30,
    # #     video_recorder=video_recorder,
    # # )

if __name__ == "__main__":
    main()


