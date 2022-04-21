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
    "alpha": 0.05, # Force more exploitation,
    "exploration_timesteps": 0 # Don't forget
}
# the rest are kept as default
