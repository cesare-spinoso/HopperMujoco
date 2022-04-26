"""
Best hyperparameters found for SAC after hyperparameter tuning,
with additional parameters gives 3615 for now
"""
import torch

# Best hyperparameters found for SAC after hyperparameter tuning

params = {
    "q_architecture": (64, 64),
    "q_activation_function": torch.nn.ReLU,
    "policy_architecture": (64, 64),
    "policy_activation_function": torch.nn.ReLU,
}

# Hyperparameters for SAC to do re-training

params = {
    "q_architecture": (64, 64),
    "q_activation_function": torch.nn.ReLU,
    "q_lr": 1e-3,  # Does a higher lr help? It doesn't look like it does no
    "policy_architecture": (64, 64),
    "policy_activation_function": torch.nn.ReLU,
    "policy_lr": 1e-3,
    "update_alpha": True,  # Automatic exploration
    "exploration_timesteps": 0,  # Don't get data that cause you to forget
    #"update_start_in_timesteps": 5000,
    "update_frequency_in_episodes": 5,  # Increase update frequency
}
