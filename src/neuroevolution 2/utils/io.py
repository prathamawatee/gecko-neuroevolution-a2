import os
import pickle
from datetime import datetime
import matplotlib.pyplot as plt

# --- Paths ---
BASE_RESULTS = "results"
CONTROLLERS_DIR = os.path.join(BASE_RESULTS, "controllers")
PLOTS_DIR = os.path.join(BASE_RESULTS, "plots")
LOGS_DIR = os.path.join(BASE_RESULTS, "logs")

# Ensure directories exist
for d in [BASE_RESULTS, CONTROLLERS_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)


def save_best_controller(best_individual, network_config, experiment_name, run_number=None):
    """Save the best controller (weights + metadata) in results/controllers."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if run_number is not None:
        filename = f"best_controller_{experiment_name}_run{run_number}_{timestamp}.pkl"
    else:
        filename = f"best_controller_{experiment_name}_{timestamp}.pkl"

    filepath = os.path.join(CONTROLLERS_DIR, filename)

    controller_data = {
        'weights': best_individual,
        'network_config': network_config,
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'fitness': (
            best_individual.fitness.values[0]
            if hasattr(best_individual, 'fitness') else None
        )
    }

    with open(filepath, 'wb') as f:
        pickle.dump(controller_data, f)

    print(f"[+] Best controller saved to: {filepath}")
    return filepath


def save_plot(fig, experiment_name, filename=None):
    """Save a matplotlib figure to results/plots."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{experiment_name}_plot_{timestamp}.png"

    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"[+] Plot saved to: {filepath}")
    return filepath


def save_log(data, experiment_name, run_number=None):
    """Save raw log data (like fitness stats per generation) to results/logs."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if run_number is not None:
        filename = f"log_{experiment_name}_run{run_number}_{timestamp}.pkl"
    else:
        filename = f"log_{experiment_name}_{timestamp}.pkl"

    filepath = os.path.join(LOGS_DIR, filename)

    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

    print(f"[+] Log saved to: {filepath}")
    return filepath
