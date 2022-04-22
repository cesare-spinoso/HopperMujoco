"""
Best hyperparameters found for SAC after hyperparameter tuning
with the additional change in hyperparameters for gaming the
sample efficiency.
"""
import torch
import sklearn.model_selection


grid = [
    {
        "q_architecture": [(64, 64)],
        "q_activation_function": [torch.nn.ReLU],
        "policy_architecture": [(64, 64)],
        "policy_activation_function": [torch.nn.ReLU],
        "q_lr": [1e-3],
        "policy_lr": [1e-3],
        "alpha": [0.1, 0.2, 0.4, 0.8],
        "update_alpha": [False],
        "exploration_timesteps": [100, 500, 1000],
        "update_start_in_episodes": [250, 500, 1000],
        "update_frequency_in_episodes": [1, 5, 15, 25],
    },
    {
        "q_architecture": [(64, 64)],
        "q_activation_function": [torch.nn.ReLU],
        "policy_architecture": [(64, 64)],
        "policy_activation_function": [torch.nn.ReLU],
        "q_lr": [1e-3],
        "policy_lr": [1e-3],
        "update_alpha": [True],
        "exploration_timesteps": [100, 500, 1000],
        "update_start_in_episodes": [250, 500, 1000],
        "update_frequency_in_episodes": [1, 5, 15, 25],
    },
]

hyperparameter_grid = sklearn.model_selection.ParameterGrid(grid)
