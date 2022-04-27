import os
import torch
from torch import nn
from torch.functional import F
from torch.distributions import Normal
import numpy as np

from typing import Tuple, Optional
from copy import deepcopy


class Agent:
    """
    Soft Actor-Critic Agent, implemented following the OpenAI Spinning Up pseudocode available here:
        https://spinningup.openai.com/en/latest/algorithms/sac.html

    Args:
      alpha:                        Entropy regularization coefficent. Defaults to 0.2.
      exploration_timesteps:        How many timesteps does the agent use at the beginning for uniform exploration.
                                    Defaults to 10_000.
      update_frequency_in_episodes: Frequency (in episodes) of the number of times that the agent takes gradient steps.
                                    Defaults to 50.
      update_start_in_episodes:     Number of episodes required before the agent starts taking gradient steps for its
                                    networks. This is mostly here to ensure that the buffer is full enough to batching.
                                    Defaults to 1_000.
      number_of_batch_updates:      Number of gradient updates to take. Defaults to 1_000.
    """

    def __init__(
        self,
        env_specs,
        gamma: float = 0.99,
        polyak: float = 0.995,
        q_lr: float = 1e-3,
        q_architecture: tuple = (64, 64),
        q_activation_function: F = nn.ReLU,
        policy_lr: float = 1e-3,
        policy_architecture: tuple = (64, 64),
        policy_activation_function: F = nn.ReLU,
        buffer_size: int = 3_000_000,
        alpha: float = 0.2,
        update_alpha: str = None,
        alpha_lr: float = 3e-4,
        learning_rate_scheduler: str = None,
        exploration_timesteps: int = 10_000,
        update_frequency_in_episodes: int = 50,
        update_start_in_episodes: int = 100,
        update_start_in_timesteps: int = None,
        number_of_batch_updates: int = 1_000,
        batch_size: int = 100,
        replay_buffer_type: str = "uniform"
    ):
        # Environment variables
        self.env_specs = env_specs
        # Number of observations (states) and actions
        self.num_obs = env_specs["observation_space"].shape[0]
        self.num_actions = env_specs["action_space"].shape[0]
        self.action_limit = env_specs["action_space"].high[0]
        # Tracking variables
        self.current_timestep = 0
        self.current_episode = 0
        self.episode_of_last_update = None
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.gamma = gamma
        self.polyak = polyak
        # Number of time steps where sample actions randomly
        self.exploration_timesteps = exploration_timesteps
        # Frequency, start and size of updates
        self.update_frequency_in_episodes = update_frequency_in_episodes
        self.update_start_in_episodes = update_start_in_episodes
        self.update_start_in_timesteps = update_start_in_timesteps
        self.number_of_batch_updates = number_of_batch_updates
        self.batch_size = batch_size

        # Q-networks (Q1 and Q1)
        self.q1_network = QNetwork(
            num_obs=self.num_obs,
            num_actions=self.num_actions,
            architecture=q_architecture,
            activation_function=q_activation_function,
        )
        self.q1_optimizer = torch.optim.Adam(self.q1_network.parameters(), lr=q_lr)
        self.q2_network = QNetwork(
            num_obs=self.num_obs,
            num_actions=self.num_actions,
            architecture=q_architecture,
            activation_function=q_activation_function,
        )
        self.q2_optimizer = torch.optim.Adam(self.q2_network.parameters(), lr=q_lr)
        # Create q target networks (for both 1 and 2) and freeze gradients
        self.q1_network_target = deepcopy(self.q1_network)
        self._freeze_network(self.q1_network_target)
        self.q2_network_target = deepcopy(self.q2_network)
        self._freeze_network(self.q2_network_target)

        # Policy network
        self.policy_network = PolicyNetwork(
            num_obs=self.num_obs,
            num_actions=self.num_actions,
            action_limit=self.action_limit,
            architecture=policy_architecture,
            activation_function=policy_activation_function,
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy_network.parameters(), lr=policy_lr
        )

        # Learning rate scheduler
        self.learning_rate_scheduler = learning_rate_scheduler
        if self.learning_rate_scheduler is not None:
            assert self.learning_rate_scheduler in {
                "exponential_decay",
                "cosine_annealing",
            }
            self.update_counter = 0
            if self.learning_rate_scheduler == "cosine_annealing":
                self.learning_rate_scheduler_frequency = 100
                self.q1_scheduler = (
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        self.q1_optimizer, T_0=self.learning_rate_scheduler_frequency
                    )
                )
                self.q2_scheduler = (
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        self.q2_optimizer, T_0=self.learning_rate_scheduler_frequency
                    )
                )
                self.policy_scheduler = (
                    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        self.policy_optimizer,
                        T_0=self.learning_rate_scheduler_frequency,
                    )
                )
            else:
                self.learning_rate_scheduler_frequency_timesteps = 100_000
                self.decay_rate = 0.9
                self.q1_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.q1_optimizer, gamma=self.decay_rate
                )
                self.q2_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.q2_optimizer, gamma=self.decay_rate
                )
                self.policy_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.policy_optimizer, gamma=self.decay_rate
                )
        # Adaptable alpha
        self.alpha = alpha  # entropy parameter
        self.update_alpha = update_alpha
        if self.update_alpha is not None:
            assert self.update_alpha in {"learned", "exponential_decay"}
            if self.update_alpha == "learned":
                # Use a heuristic for the entropy target (ADD SOURCE)
                self.entropy_target = -np.prod(self.env_specs["action_space"].shape)
                # Set the initial alpha to 1
                self.log_alpha = torch.tensor(0.0, requires_grad=True)
                self.alpha = torch.exp(self.log_alpha)
                # Optimizer
                self.alpha_lr = alpha_lr
                self.alpha_optimizer = torch.optim.Adam(
                    [self.log_alpha], lr=self.alpha_lr
                )
            else:
                self.alpha_decaying_frequency = 100_000
                self.alpha_decay_rate = 0.9
                self.alpha_update_counter = 0
        # Replay Buffer
        self.buffer = SACBuffer(
            number_obs=self.num_obs,
            number_actions=self.num_actions,
            size=buffer_size,
            batch_size=self.batch_size,
            buffer_type=replay_buffer_type
        )

    def load_weights(self, root_path: str, pretrained_model_name: str = None) -> None:
        """Load the weights of the actor and the critic into the agent. If pretrained_model_name is None,
        then use default name of "model" which is assumed to be in the same directory as load_weights.

        Args:
          root_path (str): Root path
          pretrained_model_name (str, optional): Model name e.g. td3_ckpt_98.888. Defaults to None.
        """
        if pretrained_model_name is None:
            pretrained_model_path = os.path.join(root_path, "model.pth.tar")
        else:
            pretrained_model_path = os.path.join(
                root_path, str(pretrained_model_name) + ".pth.tar"
            )

        try:
            pretrained_model = torch.load(
                pretrained_model_path, map_location=torch.device(self.device)
            )
        except FileNotFoundError:
            raise Exception(
                "Invalid location for loading pretrained model. You need folder/filename in results folder (without .pth.tar). \
                \nE.g. python3 train_agent.py --group vpg_agent --load 2022-03-31_12h46m44/td3_ckpt_98.888"
            )

        # load state dict for the 4 networks
        self.q1_network.load_state_dict(pretrained_model["q1_network"])
        self.q2_network.load_state_dict(pretrained_model["q2_network"])
        self.policy_network.load_state_dict(pretrained_model["policy_network"])
        self.q1_network_target.load_state_dict(pretrained_model["q1_network_target"])
        self.q2_network_target.load_state_dict(pretrained_model["q2_network_target"])

        print("Loaded {} OK".format(pretrained_model_name))

    def save_checkpoint(
        self, score_avg: float, ckpt_path: str, name: str = None
    ) -> str:
        """Save the weights of the critic and the actor as well as its score. If name is None,
        then use its score as the name.
        """
        # path for current version you're saving (only need ckpt_xxx, not ckpt_xxx.pth.tar)
        if name is None:
            ckpt_path = os.path.join(
                ckpt_path, "sac_ckpt_" + str(round(score_avg, 3)) + ".pth.tar"
            )
        else:
            ckpt_path = os.path.join(
                ckpt_path,
                "sac_ckpt_" + name + "_" + str(round(score_avg, 3)) + ".pth.tar",
            )

        torch.save(
            {
                "q1_network": self.q1_network.state_dict(),
                "q2_network": self.q2_network.state_dict(),
                "policy_network": self.policy_network.state_dict(),
                "q1_network_target": self.q1_network_target.state_dict(),
                "q2_network_target": self.q2_network_target.state_dict(),
                "score": score_avg,
            },
            ckpt_path,
        )

        return ckpt_path

    def act(self, curr_obs: np.ndarray, mode: Optional[str] = "eval") -> np.ndarray:
        """Returns an action following observation of :curr_obs:.
        If mode is 'train', the agent waits for :self.exploration_timesteps: before sampling actions from the policy.
        If mode is 'eval', agent samples from the policy right away.
        """
        with torch.no_grad():
            curr_obs = torch.from_numpy(curr_obs).float().to(self.device)
            if mode == "train":
                if self.current_timestep < self.exploration_timesteps:
                    # Do exploration at the beginning of training
                    action = self.env_specs["action_space"].sample()
                else:
                    action, _ = self.policy_network(
                        curr_obs, deterministic=False, return_logprob=False
                    )
                    action = action.data.cpu().numpy()
            else:
                action, _ = self.policy_network(
                    curr_obs, deterministic=True, return_logprob=False
                )
                action = action.data.cpu().numpy()
            return action

    def update(self, curr_obs, action, reward, next_obs, done, timestep, logger=None):
        """After each timestep, send experience to the SACBuffer. Train the agent if it is ready to train."""
        # Store experience in buffer
        curr_obs = torch.from_numpy(curr_obs).float()
        action = torch.from_numpy(action).float()
        next_obs = torch.from_numpy(next_obs).float()
        self.buffer.store_experience(curr_obs, action, reward, next_obs, done)
        # Track the current timestep
        self.current_timestep = timestep
        if done:
            self.current_episode += 1
            # print(f"Current episode: {self.current_episode}")
        if self.is_ready_to_train():
            self.train()
            print(self.current_episode)
            print(f"Alpha: {self.alpha}")
            self.episode_of_last_update = self.current_episode
            if logger:
                logger.log(f"Timestep: {timestep}")
                logger.log(f"Current episode: {self.current_episode}")
                logger.log(f"Alpha: {self.alpha}")
                if self.learning_rate_scheduler is not None:
                    logger.log(f"LR: {self.q1_scheduler.get_last_lr()[0]}")

    def is_ready_to_train(self) -> bool:
        """Returns True if enough exploration timesteps/episodes have passed and we are at the end of an episode.
           Otherwise, returns False."""
        if self.episode_of_last_update is None:
            return self.current_episode > self.update_start_in_episodes or (
                self.update_start_in_timesteps is not None
                and self.current_timestep > self.update_start_in_timesteps
            )
        else:
            return (
                self.current_episode > self.episode_of_last_update
                and self.current_episode % self.update_frequency_in_episodes == 0
            )

    def train(self):
        """Trains agents following OpenAI's Spinning Up pseudocode:
            https://spinningup.openai.com/en/latest/algorithms/sac.html
        """
        for j in range(self.number_of_batch_updates):
            # Get training batch
            (
                obs_data,
                action_data,
                reward_data,
                next_obs_data,
                done_data,
            ) = self.buffer.get_training_batch()
            # Get the training targets (Line 12 of the OpenAI pseudocode)
            y = self._compute_targets(reward_data, next_obs_data, done_data)

            self._train_q_networks(obs_data, action_data, y)
            self._train_policy_network(obs_data)

            # Learn entropy regularization coefficient, if applicable
            self._train_alpha(obs_data)

            self._update_target_networks()

        # Update learning rate scheduler, if applicable
        self.update_learning_rate()

    def _compute_targets(
        self,
        reward_data: torch.Tensor,
        next_obs_data: torch.Tensor,
        done_data: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the targets for the Q network."""
        with torch.no_grad():
            target_actions, log_proba = self.policy_network(next_obs_data)
            min_q_network_target = torch.min(
                self.q1_network_target(next_obs_data, target_actions).squeeze(-1),
                self.q2_network_target(next_obs_data, target_actions).squeeze(-1),
            )
            return reward_data + self.gamma * (1 - done_data) * (
                min_q_network_target - self.alpha * log_proba
            )

    def _train_q_networks(self, obs_data, action_data, y):
        """Train the Q-network (Line 13 of the OpenAI pseudocode)"""
        for q_network, q_optimizer in [
            (self.q1_network, self.q1_optimizer),
            (self.q2_network, self.q2_optimizer),
        ]:
            # Train the Q-network
            q_optimizer.zero_grad()
            # Compute q-values
            q_values = q_network(obs_data, action_data).squeeze(-1)
            # Compute loss
            loss = F.mse_loss(q_values, y)
            # Backpropagate
            loss.backward()
            # Take a step
            q_optimizer.step()

    def _train_policy_network(self, obs_data):
        """Train the policy network (Line 14 of the OpenAI pseudocode)"""
        # Freeze the Q-networks
        self._freeze_network(self.q1_network)
        self._freeze_network(self.q2_network)
        self._freeze_alpha()
        # Train the policy network
        self.policy_optimizer.zero_grad()
        # Compute the policy target
        actions, log_proba = self.policy_network(obs_data)
        q1_values = self.q1_network(obs_data, actions).squeeze(-1)
        q2_values = self.q2_network(obs_data, actions).squeeze(-1)
        q_values = torch.min(q1_values, q2_values)
        # Want to maximize the the output of pi and thus minimize its negative output
        loss = -(q_values - self.alpha * log_proba).mean()
        # Backpropagate
        loss.backward()
        # Take a step
        self.policy_optimizer.step()
        # Unfreeze the Q-network
        self._unfreeze_network(self.q1_network)
        self._unfreeze_network(self.q2_network)
        self._unfreeze_alpha()

    def _train_alpha(self, obs_data):
        """Learn the entropy regularization coefficient."""
        if self.update_alpha is not None:
            if self.update_alpha == "learned":
                # Zero grad
                self.alpha_optimizer.zero_grad()
                # Get the alpha targets
                with torch.no_grad():
                    _, log_proba = self.policy_network(obs_data)
                targets = -torch.exp(self.log_alpha) * (log_proba + self.entropy_target)
                alpha_loss = targets.mean()
                # Backpropagate
                alpha_loss.backward()
                # Take a step
                self.alpha_optimizer.step()
                # Update the alpha, is this line necessary?
                self.alpha = torch.exp(self.log_alpha)
            else:
                if (
                    self.current_timestep - self.alpha_update_counter
                ) / self.alpha_decaying_frequency > 1:
                    self.alpha_update_counter = self.current_timestep
                    self.alpha *= self.alpha_decay_rate

    def update_learning_rate(self):
        """Learning rate scheduler update."""
        if self.learning_rate_scheduler is not None:
            if self.learning_rate_scheduler == "cosine_annealing":
                self.q1_scheduler.step()
                self.q2_scheduler.step()
                self.policy_scheduler.step()
            else:
                if (
                    self.current_timestep - self.update_counter
                ) / self.learning_rate_scheduler_frequency_timesteps > 1:
                    self.update_counter = self.current_timestep
                    self.q1_scheduler.step()
                    self.q2_scheduler.step()
                    self.policy_scheduler.step()

    def _update_target_networks(self):
        """Update the target networks for the q-network and the policy-network via
        a moving average (Line 15 of the OpenAI pseudocode).
        """
        with torch.no_grad():
            self._polyak_average_update(self.q1_network, self.q1_network_target)
            self._polyak_average_update(self.q2_network, self.q2_network_target)

    def _polyak_average_update(self, network, target_network):
        """Polyak averaging of network."""
        for target_param, param in zip(
            target_network.parameters(), network.parameters()
        ):
            # Use OpenAI's in-place trick
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)


    def _freeze_network(self, network):
        """Freeze the gradients of the network so that loss cannot backprop
        through it."""
        for param in network.parameters():
            param.requires_grad = False

    def _unfreeze_network(self, network):
        """Unfreeze the network gradients."""
        for param in network.parameters():
            param.requires_grad = True

    def _freeze_alpha(self):
        if self.update_alpha == "learned":
            self.log_alpha.requires_grad = False

    def _unfreeze_alpha(self):
        if self.update_alpha == "learned":
            self.log_alpha.requires_grad = True


class QNetwork(nn.Module):
    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        architecture: tuple,
        activation_function: F,
    ) -> None:
        """Standard MLP for the Q-Network takes in observations and actions and
        outputs estimate q-value."""
        super(QNetwork, self).__init__()
        layers = [
            nn.Linear(num_obs + num_actions, architecture[0]),
            activation_function(),
        ]
        for input_dimension, output_dimension in zip(
            architecture[0:], architecture[1:]
        ):
            layers.append(nn.Linear(input_dimension, output_dimension))
            layers.append(activation_function())
        layers.append(nn.Linear(architecture[-1], 1))
        self.network = nn.Sequential(*layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        # Expecte batch_size x (num_obs + num_actions)
        return self.network(torch.hstack((s, a)))


class PolicyNetwork(nn.Module):
    # Spin-up uses a lower and upper bound for the log_std calculation
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20

    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        action_limit: float,
        architecture: tuple,
        activation_function: F,
    ) -> None:
        super().__init__()
        """Standard MLP for the policy takes in an observation and outputs an action."""
        super(PolicyNetwork, self).__init__()
        # Architecture before mean and std output
        layers = [nn.Linear(num_obs, architecture[0]), activation_function()]
        for input_dimension, output_dimension in zip(
            architecture[0:], architecture[1:]
        ):
            layers.append(nn.Linear(input_dimension, output_dimension))
            layers.append(activation_function())
        # Add a layer for the mean and the std
        self.network = nn.Sequential(*layers)
        self.mu_layer = nn.Linear(architecture[-1], num_actions)
        # Output the log for numerical stability
        self.log_std_layer = nn.Linear(architecture[-1], num_actions)

        self.action_limit = action_limit
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(
        self, x: torch.Tensor, deterministic=False, return_logprob=True
    ) -> torch.Tensor:
        network_output = self.network(x)
        mu = self.mu_layer(network_output)
        log_std = self.log_std_layer(network_output)
        log_std = torch.clamp(
            log_std, PolicyNetwork.LOG_STD_MIN, PolicyNetwork.LOG_STD_MAX
        )
        std = torch.exp(log_std)

        # This is directly from the spin-up implementation
        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            # Use the re-parametrization trick so that gradients go into the actions,
            # rsample allows you to do mu + std * noise so that can backprop
            # through the mu and std layers
            pi_action = pi_distribution.rsample()

        if return_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(
                axis=1
            )
        else:
            logp_pi = None

        # Squash the sampled action and then rescale based on environment
        pi_action = torch.tanh(pi_action)
        pi_action = self.action_limit * pi_action

        return pi_action, logp_pi


class SACBuffer:
    def __init__(
        self,
        number_obs: int,
        number_actions: int,
        size: int = 3_000_000,
        batch_size: int = 100,
        gamma: float = 0.99,
        priority_threshold: float = 0.5,
        buffer_type: str = "uniform",
    ) -> None:
        """Buffer responsible for storing the experience and the Q target.

        If :buffer_type: is 'prioritized' then prioritized experience replay (PER) and on-policy mixing is used when
        sampling the replay buffer, as described in (Banerjee, 2021): https://arxiv.org/pdf/2109.11767v1.pdf
        """
        # Number of states, actions and total number of time steps
        self.number_obs = number_obs
        self.number_actions = number_actions
        self.size = size

        # Create state, action, next_state, reward, done buffers
        self.action_buffer = torch.zeros((self.size, self.number_actions))
        self.obs_buffer = torch.zeros((self.size, self.number_obs))
        self.next_obs_buffer = torch.zeros((self.size, self.number_obs))
        self.reward_buffer = torch.zeros(self.size)
        self.priority_buffer = torch.zeros(self.size)  # for PER
        self.done_buffer = torch.zeros(self.size)
        self.experience_pointer = 0
        self.effective_size = 0
        self.current_episode_reward = 0

        # PER Cache
        self.action_cache = torch.zeros((self.size, self.number_actions))
        self.obs_cache = torch.zeros((self.size, self.number_obs))
        self.next_obs_cache = torch.zeros((self.size, self.number_obs))
        self.reward_cache = torch.zeros(self.size)
        self.priority_cache = torch.zeros(self.size)  # for PER
        self.done_cache = torch.zeros(self.size)
        self.cache_pointer = 0
        self.effective_cache_size = 0
        self.previous_effective_cache_size = 0

        # Temporary buffer
        self.action_temp_buffer = torch.zeros((self.size, self.number_actions))
        self.obs_temp_buffer = torch.zeros((self.size, self.number_obs))
        self.next_obs_temp_buffer = torch.zeros((self.size, self.number_obs))
        self.reward_temp_buffer = torch.zeros(self.size)
        self.priority_temp_buffer = torch.zeros(self.size)  # for PER
        self.done_temp_buffer = torch.zeros(self.size)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.batch_size = batch_size
        self.gamma = gamma
        self.priority_threshold = priority_threshold
        self.buffer_type = buffer_type  # uniform or prioritized

    def store_experience(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_obs: torch.Tensor,
        done: bool,
    ) -> None:
        """
        When called during an episode, experience is stored in the cache. At the end of episode, calculate the episodic
        return (used for the 'priority' of each transition in that episode) and transfer the cache into the buffer.
        """

        # Store experience in cache
        self.action_cache[self.cache_pointer, :] = action
        self.obs_cache[self.cache_pointer, :] = obs
        self.next_obs_cache[self.cache_pointer, :] = next_obs
        self.reward_cache[self.cache_pointer] = reward
        self.done_cache[self.cache_pointer] = int(done)

        # Update total reward
        self.current_episode_reward += (
            np.power(self.gamma, self.effective_cache_size) * reward
        )
        self.cache_pointer += 1
        self.effective_cache_size += 1

        if done:
            # Copy cache into buffer
            self.action_buffer[
                self.experience_pointer: self.experience_pointer
                + self.effective_cache_size,
                :,
            ] = self.action_cache[0: self.effective_cache_size, :]
            self.obs_buffer[
                self.experience_pointer: self.experience_pointer
                + self.effective_cache_size,
                :,
            ] = self.obs_cache[0: self.effective_cache_size, :]
            self.next_obs_buffer[
                self.experience_pointer: self.experience_pointer
                + self.effective_cache_size,
                :,
            ] = self.next_obs_cache[0: self.effective_cache_size, :]
            self.reward_buffer[
                self.experience_pointer: self.experience_pointer
                + self.effective_cache_size
            ] = self.reward_cache[0: self.effective_cache_size]
            self.done_buffer[
                self.experience_pointer: self.experience_pointer
                + self.effective_cache_size
            ] = self.done_cache[0: self.effective_cache_size]

            self.priority_buffer[
                self.experience_pointer: self.experience_pointer
                + self.effective_cache_size
            ] = self.current_episode_reward

            self.effective_size += self.effective_cache_size
            self.experience_pointer += self.effective_cache_size
            self.previous_effective_cache_size = self.effective_cache_size
            self.effective_cache_size = 0
            self.cache_pointer = 0
            self.current_episode_reward = 0

    def get_training_batch(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample :self.batch_size: number of data points from the replay buffer and returns tensors
        for s, a, s', r, and done. Experience is sampled on the buffer based on the :buffer_type:
        """
        if self.buffer_type == "prioritized":
            # Sample indices
            sample_index1 = np.random.choice(
                np.arange(self.effective_size), self.batch_size
            )
            sample_index2 = np.random.choice(
                np.arange(self.effective_size), self.batch_size
            )
            # Sample data prioritization - Section 3.A
            priority_sample_index1 = self.priority_buffer[sample_index1]
            priority_sample_index2 = self.priority_buffer[sample_index2]

            # Compute the similarity between the samples
            priority_sample_similarity = np.dot(
                priority_sample_index1, priority_sample_index2
            ) / (
                np.linalg.norm(priority_sample_index1)
                * np.linalg.norm(priority_sample_index2)
            )

            # Only use prioritized samples if they are different enough, otherwise use random samples
            if priority_sample_similarity < self.priority_threshold:
                sample_index = np.concatenate((sample_index1, sample_index2))
                indexing_of_sample_index = np.argsort(
                    self.priority_buffer[sample_index]
                )
                sample_index = sample_index[indexing_of_sample_index][
                    -self.batch_size:
                ]

                # To implement Section 3.B - To mix on and off-policy
                # On-policy experience is stored in the previous written cache
                swap_index = np.random.choice(self.batch_size)

                return_obs_buffer = self.obs_buffer[sample_index, :]
                return_action_buffer = self.action_buffer[sample_index, :]
                return_reward_buffer = self.reward_buffer[sample_index]
                return_next_obs_buffer = self.next_obs_buffer[sample_index, :]
                return_done_buffer = self.done_buffer[sample_index]

                return_obs_buffer[swap_index, :] = self.obs_cache[
                    np.random.choice(self.previous_effective_cache_size), :
                ]
                return_action_buffer[swap_index, :] = self.action_cache[
                    np.random.choice(self.previous_effective_cache_size), :
                ]
                return_reward_buffer[swap_index] = self.reward_cache[
                    np.random.choice(self.previous_effective_cache_size)
                ]
                return_next_obs_buffer[swap_index, :] = self.next_obs_cache[
                    np.random.choice(self.previous_effective_cache_size), :
                ]
                return_done_buffer[swap_index] = self.done_cache[
                    np.random.choice(self.previous_effective_cache_size)
                ]

                return (
                    return_obs_buffer.to(self.device),
                    return_action_buffer.to(self.device),
                    return_reward_buffer.to(self.device),
                    return_next_obs_buffer.to(self.device),
                    return_done_buffer.to(self.device),
                )
            else:
                sample_index = sample_index1
                return (
                    self.obs_buffer[sample_index, :].to(self.device),
                    self.action_buffer[sample_index, :].to(self.device),
                    self.reward_buffer[sample_index].to(self.device),
                    self.next_obs_buffer[sample_index, :].to(self.device),
                    self.done_buffer[sample_index].to(self.device),
                )
        else:
            sample_index = np.random.choice(
                np.arange(self.effective_size), self.batch_size, replace=False
            )
            return (
                self.obs_buffer[sample_index, :].to(self.device),
                self.action_buffer[sample_index, :].to(self.device),
                self.reward_buffer[sample_index].to(self.device),
                self.next_obs_buffer[sample_index, :].to(self.device),
                self.done_buffer[sample_index].to(self.device),
            )
