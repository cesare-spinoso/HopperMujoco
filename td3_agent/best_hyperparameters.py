"""
Best hyperparameters for TD3 found from the tuning experiment. (m_1)
"""
import torch

params = {'q_architecture': (100, 50, 25), 
    'q_activation_function': torch.nn.ReLU, 
    'policy_architecture': (100, 50, 25), 
    'policy_activation_function': torch.nn.ReLU}