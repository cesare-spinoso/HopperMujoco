"""
Best hyperparameters for SAC found from the tuning experiment. (m_0)
"""
import torch

params = {'q_architecture': (64, 64), 
    'q_activation_function': torch.nn.ReLU, 
    'policy_architecture': (64, 64), 
    'policy_activation_function': torch.nn.ReLU
}
