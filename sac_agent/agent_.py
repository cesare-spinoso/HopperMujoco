"""
Implementation of the Soft Actor-Critic agent proposed in (Haarnoja, 2018)

Code adapted from 'Modern Reinforcement Learning: Actor-Critic Algorithms' Udemy course
"""
import numpy as np
import torch
import torch.nn.functional as F
from .buffer import ReplayBuffer
from .networks import Actor, Critic, ValueNetwork

import logging
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())    # prints to stderr
logger.setLevel(logging.INFO)                 # change to logging.DEBUG or logging.WARNING for more/less messages


class Agent:
  """
  Soft Actor-Critic agent from (Haarnoja, 2018). This is an off-policy maximum entropy agent with a stochastic actor.
  ----------
  Networks:
      :actor:          Gaussian policy with mean and covariance given by neural networks.
      :critic 1 and 2: Two Q functions, trained independently. At each update step, the minimum of the two Q functions
                       is used for the value and actor gradients.
      :value:          Parameterized state-value function which accounts for entropy
      :target_value:   Parameterized state-value function which does not account for entropy (ie. 'true' value function)
  """
  def __init__(self, env_specs: dict, id: str = 'sac_test', actor_lr: float = 3e-4, critic_lr: float = 3e-4,
               tau: float = 5e-3, gamma: float = 0.99, max_buffer_size: int = 10_000_000, layer1_size: int = 256,
               layer2_size: int = 256, batch_size: int = 100, reward_scale: int = 2):

    # General params
    self.tau = tau                # smoothing constant for target network. See section 5.2 of (Haarnoja, 2018)
    self.gamma = gamma            # discount factor
    self.scale = reward_scale     # Important hyperparameter, needs exploring. See section 5.2 of (Haarnoja, 2018)
    self.batch_size = batch_size

    input_dims = env_specs['observation_space'].shape
    n_actions = env_specs['action_space'].shape[0]
    self.memory = ReplayBuffer(max_buffer_size, input_dims, n_actions)

    # Define actor, critic, and value networks
    self.actor = Actor(input_dims, layer1_size, layer2_size, n_actions, actor_lr,
                       max_action=env_specs['action_space'].high, name=id+'_actor')
    self.critic_1 = Critic(input_dims, layer1_size, layer2_size, n_actions, critic_lr, name=id+'_critic_1')
    self.critic_2 = Critic(input_dims, layer1_size, layer2_size, n_actions, critic_lr, name=id+'_critic_2')
    self.value = ValueNetwork(input_dims, layer1_size, layer2_size, critic_lr, name=id+'_value')
    self.target_value = ValueNetwork(input_dims, layer1_size, layer2_size, critic_lr, name=id+'_target_value')

    self.update_target_network_parameters(tau=1)

  def act(self, observation: np.ndarray, mode: str = 'train') -> np.ndarray:
    """Returns a sample from the policy: ie. return ~ pi(a | s=observation)
    TODO: Add a 'mean' version for testing mode instead of just sampling
    """
    state = torch.Tensor([observation]).to(self.actor.device)
    actions, _ = self.actor.sample_normal(state, reparameterize=False)
    return actions.cpu().detach().numpy()[0]

  def update(self, current_state: np.ndarray, action: np.ndarray, reward: np.float64, next_state: np.ndarray,
             done: bool, timestep: int):
    """Stores experience in memory and updates the network parameters"""
    self.memory.store_transition(current_state, action, reward, next_state, done)
    self.train()

  def update_target_network_parameters(self, tau: float = None):
    """
    Updates the target network using exponentially moving average.
    Target network is updated independently since there is no entropy term - see section 5.2 of (Haarnoja, 2018)
    :param tau: smoothing constant
    """
    if tau is None:
      tau = self.tau

    target_value_params = self.target_value.named_parameters()
    value_params = self.value.named_parameters()

    target_value_state_dict = dict(target_value_params)
    value_state_dict = dict(value_params)

    for name in value_state_dict:
      value_state_dict[name] = tau * value_state_dict[name].clone() + \
                               (1-tau)*target_value_state_dict[name].clone()

    self.target_value.load_state_dict(value_state_dict)

  def train(self):
    """Trains actor, critic, and value networks using the equations from (Haarnoja, 2018).
    NOTE THAT the equation and section numbers refer to equations and sections from this paper."""
    if self.memory.mem_counter < self.batch_size:
      return

    # Sample experience and convert it to torch tensors
    state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

    reward = torch.tensor(reward, dtype=torch.float).to(self.critic_1.device)
    done = torch.tensor(done).to(self.critic_1.device)
    next_state = torch.tensor(new_state, dtype=torch.float).to(self.critic_1.device)
    state = torch.tensor(state, dtype=torch.float).to(self.critic_1.device)
    action = torch.tensor(action, dtype=torch.float).to(self.critic_1.device)

    value = self.value(state).view(-1)
    next_value = self.target_value(next_state).view(-1)
    next_value[done] = 0.0

    # Take the min. of both critics to reduce positivity bias in the policy improvement step
    # see last paragraph in section 4.2
    actions, log_probs = self.actor.sample_normal(state, reparameterize=False)
    log_probs = log_probs.view(-1)
    q1_new_policy = self.critic_1.forward(state, actions)
    q2_new_policy = self.critic_2.forward(state, actions)
    critic_value = torch.min(q1_new_policy, q2_new_policy)
    critic_value = critic_value.view(-1)

    # Update value network following Equation 6
    self.value.optimizer.zero_grad()
    value_target = critic_value - log_probs
    value_loss = 0.5 * F.mse_loss(value, value_target)
    value_loss.backward(retain_graph=True)
    self.value.optimizer.step()

    # Update actor following Equation 13
    actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
    log_probs = log_probs.view(-1)
    q1_new_policy = self.critic_1.forward(state, actions)
    q2_new_policy = self.critic_2.forward(state, actions)
    critic_value = torch.min(q1_new_policy, q2_new_policy)
    critic_value = critic_value.view(-1)
    actor_loss = log_probs - critic_value
    actor_loss = torch.mean(actor_loss)
    self.actor.optimizer.zero_grad()
    actor_loss.backward(retain_graph=True)
    self.actor.optimizer.step()

    # Update critics following Equation 9
    q_hat = self.scale * reward + self.gamma*next_value
    q1_old_policy = self.critic_1.forward(state, action).view(-1)
    q2_old_policy = self.critic_2.forward(state, action).view(-1)
    critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
    critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)

    self.critic_1.optimizer.zero_grad()
    self.critic_2.optimizer.zero_grad()
    critic_loss = critic_1_loss + critic_2_loss
    critic_loss.backward()
    self.critic_1.optimizer.step()
    self.critic_2.optimizer.step()

    # Update target value network
    self.update_target_network_parameters()

  def save_checkpoint(self):
    """Saves parameters of all 5 models. Filepaths to weights are created in the constructor of each model"""
    logger.debug('... saving models ...')
    self.actor.save_checkpoint()
    self.value.save_checkpoint()
    self.target_value.save_checkpoint()
    self.critic_1.save_checkpoint()
    self.critic_2.save_checkpoint()

  def load_checkpoint(self):
    """Loads parameters of all 5 models. Filepaths to weights are created in the constructor of each model"""
    logger.debug('... loading models ...')
    self.actor.load_checkpoint()
    self.value.load_checkpoint()
    self.target_value.load_checkpoint()
    self.critic_1.load_checkpoint()
    self.critic_2.load_checkpoint()

