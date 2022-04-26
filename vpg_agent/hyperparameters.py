"""
Hyperparameters for tuning VPG found from this paper: https://arxiv.org/pdf/1709.06560.pdf
The train_agent file will pull hyperparameter_grid from here.
"""
import torch
import sklearn.model_selection

grid = {
    "architecture": [(64, 64), (100, 50, 25), (400, 300)],
    "activation": [torch.nn.ReLU, torch.nn.Tanh],
    "number_of_critic_updates_per_actor_update": [100],
    "batch_size_in_time_steps": [5000],
    "actor_lr": [3e-4, 3e-3],
    "critic_lr": [1e-3, 1e-2],
}
grid = sklearn.model_selection.ParameterGrid(grid)

# Provide activations and architectures to both actor *and* critic
hyperparameter_grid = []

for i, params in enumerate(grid):
    params["actor_architecture"] = params["architecture"]
    params["actor_activation_function"] = params["activation"]
    params["critic_architecture"] = params["architecture"]
    params["critic_activation_function"] = params["activation"]
    del params["architecture"]
    del params["activation"]
    hyperparameter_grid.append(params)

