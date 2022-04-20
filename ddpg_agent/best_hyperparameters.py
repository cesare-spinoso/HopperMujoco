"""
Best hyperparameters for DDPG found from the tuning experiment. (m_2)
"""
import torch

params = {'q_architecture': (400, 300), 
    'q_activation_function': torch.nn.ReLU, 
    'policy_architecture': (400, 300), 
    'policy_activation_function': torch.nn.ReLU
}
