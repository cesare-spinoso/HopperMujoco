"""
Class for handling memory buffer

Code adapted from 'Modern Reinforcement Learning: Actor-Critic Algorithms' Udemy course
"""
import numpy as np


class ReplayBuffer:
  """
  Implementation of a ReplayBuffer for Actor-Critic networks.
  Stores the (state, action, reward, next state, is_terminal) transitions in memory. Experience can be replayed with the
  sample_buffer() method.
  """
  def __init__(self, mem_size: int, input_shape: tuple, n_actions: int):
    self.mem_size = mem_size
    self.mem_counter = 0
    self.state_memory = np.zeros((self.mem_size, *input_shape))
    self.next_state_memory = np.zeros((self.mem_size, *input_shape))
    self.action_memory = np.zeros((self.mem_size, n_actions))
    self.reward_memory = np.zeros(self.mem_size)
    self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

  def store_transition(self, current_state: np.ndarray, action: np.ndarray, reward: np.float64, next_state: np.ndarray,
                       done: bool):
    """Stores a state-action-reward-state' transition into the replay buffer."""

    index = self.mem_counter % self.mem_size    # overwrite oldest entries when we run out of space

    self.state_memory[index] = current_state
    self.action_memory[index] = action
    self.reward_memory[index] = reward
    self.next_state_memory[index] = next_state
    self.terminal_memory[index] = done

    self.mem_counter += 1

  def sample_buffer(self, batch_size: int):
    """Returns a sample of the replay buffer of size :batch:"""
    max_mem = min(self.mem_counter, self.mem_size)

    batch = np.random.choice(max_mem, batch_size)

    current_states = self.state_memory[batch]
    actions = self.action_memory[batch]
    rewards = self.reward_memory[batch]
    next_states = self.next_state_memory[batch]
    dones = self.terminal_memory[batch]

    return current_states, actions, rewards, next_states, dones
