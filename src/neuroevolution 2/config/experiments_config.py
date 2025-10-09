COMMON_EA = {
        "population_size":60,
        "generations": 50,
        "crossover_prob": 0.7,
        "mutation_prob": 0.3,
        "crossover_alpha": 0.5,
        "mutation_sigma": 0.2,
        "mutation_indpb": 0.1,
        "tournament_size": 3,
        "simulation_time": 8.0
}

experiments = {
        "Baseline_Random": {
            "network_config": {"input_size": 10, "hidden_size": 8, "output_size": 8},
            "ea_params": {**COMMON_EA}
        },
        "Exp1_Small_1x8": {
            "network_config": {"input_size": 10, "hidden_size": 8, "output_size": 8},
            "ea_params": {**COMMON_EA}
        },
        "Exp2_Wide_1x32": {
            "network_config": {"input_size": 10, "hidden_size": 32, "output_size": 8},
            "ea_params": {**COMMON_EA}
        }
}

