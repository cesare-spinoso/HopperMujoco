"""
Best hyperparameters found for SAC after hyperparameter tuning.
"""
import torch

params = {
    "q_architecture": (64, 64),
    "q_activation_function": torch.nn.ReLU,
    "policy_architecture": (64, 64),
    "policy_activation_function": torch.nn.ReLU,
}
# the rest are kept as default
