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
}
# the rest are kept as default
