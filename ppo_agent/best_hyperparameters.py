"""
Best hyperparameters found for PPO after hyperparameter tuning.
"""
import torch

params = {'actor_architecture': (64, 64), 
        'actor_activation_function': torch.nn.ReLU, 
        'critic_architecture': (64, 64), 
        'critic_activation_function': torch.nn.ReLU}
# gamma = 0.99, lambda = 0.97, KL threshold = 0.015, clip rate = 0.2

