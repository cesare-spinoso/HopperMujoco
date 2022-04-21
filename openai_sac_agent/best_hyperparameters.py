"""
Best hyperparameters found for SAC after hyperparameter tuning,
with additional parameters gives 3615 for now
"""
import torch

params = {
    "q_architecture": (64, 64),
    "q_activation_function": torch.nn.ReLU,
    "policy_architecture": (64, 64),
    "policy_activation_function": torch.nn.ReLU,
    "update_alpha": True, # Automatic exploration
    "exploration_timesteps": 0, # Don't forget
    "update_start_in_episodes": 50,
    "update_frequency_in_episodes": 5
}
# the rest are kept as default
