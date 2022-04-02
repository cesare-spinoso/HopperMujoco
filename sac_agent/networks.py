"""
Implementation of actor, critic, and value networks used by the soft actor-critic agent.

Code adapted from 'Modern Reinforcement Learning: Actor-Critic Algorithms' Udemy course
"""

import os
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class Actor(nn.Module):
  """
  Implementation of policy from (Haarnoja, 2018). This is a Gaussian policy with mean and covariance given by neural
  networks.
  # TODO: change sigma to covariance (I think this implementation only gives standard deviation for each action, not covariance)
  The output layer of the actor network contains mean and standard deviation for each action
  """
  def __init__(self, input_dims: tuple, fc1_dims: int, fc2_dims: int, n_actions: int, learning_rate: float,
               max_action: np.ndarray, name: str, checkpoint_dir: str = 'results/sac'):
    super(Actor, self).__init__()
    self.name = name
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_file = os.path.join(self.checkpoint_dir, name + ".pth.tar")
    self.reparam_noise = 1e-6 # TODO: make hyperparam

    # Define actor network
    self.input_dims = input_dims
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims
    self.max_action = max_action    # Note: max_action is [1.0, 1.0, 1.0] for Mujoco Hopper
    self.n_actions = n_actions

    self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
    self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
    self.mu = nn.Linear(self.fc2_dims, self.n_actions)
    self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

    self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

  def forward(self, state: np.ndarray):
    """Returns the mean an variance of the conditional distribution over the action space: pi(a | s=:state:)"""
    prob = self.fc1(state)
    prob = F.relu(prob)
    prob = self.fc2(prob)
    prob = F.relu(prob)

    mu = self.mu(prob)
    sigma = self.sigma(prob)
    # Use a clamp to avoid exploding values (see og. paper)
    sigma = torch.clamp(sigma, min=self.reparam_noise, max=1)

    return mu, sigma

  def sample_normal(self, state: np.ndarray, reparameterize: bool = True):
    """Takes an action by sampling from policy, which is a normal distribution"""
    mu, sigma = self.forward(state)
    probabilities = Normal(mu, sigma)

    actions = probabilities.rsample() if reparameterize else probabilities.sample()

    action = torch.tanh(actions) * torch.Tensor(self.max_action).to(self.device)
    log_probs = probabilities.log_prob(actions)
    log_probs -= torch.log(1-action.pow(2) + self.reparam_noise)
    log_probs = log_probs.sum(1, keepdim=True)

    return action, log_probs

  def save_checkpoint(self):
    torch.save(self.state_dict(), self.checkpoint_file)

  def load_checkpoint(self):
    self.load_state_dict(torch.load(self.checkpoint_file))


class Critic(nn.Module):
  """
  Implementation of the critic from (Haarnoja, 2018). This is a soft Q-function approximated by a neural network.
  """
  def __init__(self, input_dims: tuple, fc1_dims: int, fc2_dims: int, n_actions: int, learning_rate: float, name: str,
               checkpoint_dir: str = 'results/sac'):
    super(Critic, self).__init__()
    self.name = name
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_file = os.path.join(self.checkpoint_dir, name + ".pth.tar")

    # Define critic network
    self.input_dims = input_dims
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims
    self.n_actions = n_actions

    self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
    self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
    self.q = nn.Linear(self.fc2_dims, 1)

    self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

  def forward(self, state: np.ndarray, action: np.ndarray):
    """Estimate the value of a state-action pair"""
    action_value = self.fc1(torch.cat([state, action], dim=1))
    action_value = F.relu(action_value)
    action_value = self.fc2(action_value)
    action_value = F.relu(action_value)

    q = self.q(action_value)
    return q

  def save_checkpoint(self):
    torch.save(self.state_dict(), self.checkpoint_file)

  def load_checkpoint(self):
    self.load_state_dict(torch.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
  """
  Implementation of the value network from (Haarnoja, 2018). This is a 'typical' neural network.
  """
  def __init__(self, input_dims: tuple, fc1_dims: int, fc2_dims: int, learning_rate: float, name: str,
               checkpoint_dir: str = 'results/sac'):
    super(ValueNetwork, self).__init__()
    self.name = name
    self.checkpoint_dir = checkpoint_dir
    self.checkpoint_file = os.path.join(self.checkpoint_dir, name + ".pth.tar")

    # Define value network
    self.input_dims = input_dims
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims

    self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
    self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
    self.v = nn.Linear(self.fc2_dims, 1)

    self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    self.to(self.device)

  def forward(self, state: np.ndarray):
    """Estimate the value of the state"""
    state_value = self.fc1(state)
    state_value = F.relu(state_value)
    state_value = self.fc2(state_value)
    state_value = F.relu(state_value)

    v = self.v(state_value)
    return v

  def save_checkpoint(self):
    torch.save(self.state_dict(), self.checkpoint_file)

  def load_checkpoint(self):
    self.load_state_dict(torch.load(self.checkpoint_file))

