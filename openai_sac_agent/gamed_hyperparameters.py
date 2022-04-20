"""
Best hyperparameters found for SAC after hyperparameter tuning
with the additional change in hyperparameters for gaming the
sample efficiency.
"""
import torch
import sklearn.model_selection


grid = {
    "q_architecture": [(64, 64)],
    "q_activation_function": [torch.nn.ReLU],
    "policy_architecture": [(64, 64)],
    "policy_activation_function": [torch.nn.ReLU],
    "exploration_timesteps": [1000, 5000, 10000],
    "update_start_in_episodes": [100, 500, 1000],
    "update_frequency_in_episodes": [25, 50]
}

hyperparameter_grid = sklearn.model_selection.ParameterGrid(grid)