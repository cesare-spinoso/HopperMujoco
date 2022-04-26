"""
Best hyperparameters from VPG hyperparameter tuning experiment. (m_0)
"""
import torch

params = {
    "actor_architecture": (64, 64),
    "actor_activation_function": torch.nn.ReLU,
    "critic_architecture": (64, 64),
    "critic_activation_function": torch.nn.ReLU,
    "number_of_critic_updates_per_actor_update": 100,
    "batch_size_in_time_steps": 5000,
    "actor_lr": 3e-4,
    "critic_lr": 1e-3
}

