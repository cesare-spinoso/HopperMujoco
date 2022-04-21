"""Hyperparameters for VPG found from this paper: https://arxiv.org/pdf/1709.06560.pdf
"""
import torch
import sklearn.model_selection

grid = {
    "architecture": [(128, 32), (128, 64), (64, 64)],
    "activation": [torch.nn.ReLU],
    "update_start_in_episodes": 100,
    "update_frequency": 25,
    "update_alpha": True
}

grid = sklearn.model_selection.ParameterGrid(grid)

# Provide activations and architectures to both actor and critic
hyperparameter_grid = []

for i, params in enumerate(grid):
    params["q_architecture"] = params["architecture"]
    params["q_activation_function"] = params["activation"]
    params["policy_architecture"] = params["architecture"]
    params["policy_activation_function"] = params["activation"]
    del params["architecture"]
    del params["activation"]
    hyperparameter_grid.append(params)
