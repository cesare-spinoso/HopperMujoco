"""
Hyperparameters for PPO found from OpenAI and from
https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py
"""
import torch
import sklearn.model_selection

grid = {
    "architecture": [(64, 64), (100, 50, 25), (400, 300)],
    "activation": [torch.nn.ReLU, torch.nn.Tanh]
}
# gamma = 0.99, lambda = 0.97, KL threshold = 0.015, clip rate = 0.2
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

