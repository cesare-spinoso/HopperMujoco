"""Hyperparameters for VPG found from this paper: https://arxiv.org/pdf/1709.06560.pdf
"""
import torch
import sklearn.model_selection

grid = {
    "reward_scale": [0.2, 2],
    "layer_size": [64, 256],
    "update_frequency_in_episodes": [25, 50]
}
grid = sklearn.model_selection.ParameterGrid(grid)

# Provide activations and architectures to both actor and critic
hyperparameter_grid = []

for i, params in enumerate(grid):
    params["layer1_size"] = params["layer_size"]
    params["layer2_size"] = params["layer_size"]
    del params["layer_size"]
    hyperparameter_grid.append(params)
