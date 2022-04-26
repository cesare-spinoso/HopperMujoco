"""
Best hyperparameters found for SAC after hyperparameter tuning,
with additional parameters gives 3615 for now
"""
import torch

# Best hyperparameters for SAC found from the tuning experiment. (m_0)

import torch

params = {'q_architecture': (64, 64), 
    'q_activation_function': torch.nn.ReLU, 
    'policy_architecture': (64, 64), 
    'policy_activation_function': torch.nn.ReLU
}

# Hyperparameters used to train the model for longer

params = {
    "q_architecture": (64, 64),
    "q_activation_function": torch.nn.ReLU,
    "q_lr": 1e-3,
    "policy_architecture": (64, 64),
    "policy_activation_function": torch.nn.ReLU,
    "policy_lr": 1e-3,
    "update_alpha": "learned", # Automatic exploration
    "learning_rate_scheduler": "cosine_annealing",
    "exploration_timesteps": 0, # Don't get data that cause you to forget
    "update_start_in_timesteps": 10_000,
    "update_frequency_in_episodes": 1 # Increase update frequency
}

