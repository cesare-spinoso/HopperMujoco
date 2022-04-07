"""
Hyperparameters for PPO found from OpenAI and from
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
"""
import torch
import sklearn.model_selection

grid = {
    "architecture": [(64, 64), (100, 50, 25)],
    "activation": [torch.nn.ReLU, torch.nn.Tanh],
    "actor_lr": [3e-4, 3e-3],
    "critic_lr": [1e-3, 1e-2],
}
# gamma = 0.99, lambda = 0.097, KL threshold = 0.015, clip rate = 0.2
grid = sklearn.model_selection.ParameterGrid(grid)

# Provide activations and architectures to both actor and critic
hyperparameter_grid = []

for i, params in enumerate(grid):
    params["actor_architecture"] = params["architecture"]
    params["actor_activation_function"] = params["activation"]
    params["critic_architecture"] = params["architecture"]
    params["critic_activation_function"] = params["activation"]
    del params["architecture"]
    del params["activation"]
    hyperparameter_grid.append(params)

