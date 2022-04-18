import os
import time
import torch
from torch import nn
from torch.functional import F
import numpy as np

from typing import Tuple
from copy import deepcopy


class Agent:
    """The agent class that is to be filled.
    You are allowed to add any method you
    want to this class.
    """

    def __init__(
        self,
        env_specs,
        gamma: float = 0.99,
        polyak: float = 0.995,
        q_lr: float = 1e-3,
        q_architecture: tuple = (64, 64),
        q_activation_function: F = nn.Tanh,
        policy_lr: float = 1e-3,
        policy_architecture: tuple = (64, 64),
        policy_activation_function: F = nn.Tanh,
        buffer_size: int = 1_000_000,
        action_noise_scale: float = 0.1,
        action_noise_rescaling: int = 1,
        action_noise_rescaling_frequency: int = 500_000,
        target_noise_scale: float = 0.2,
        target_noise_clip: float = 0.5,
        policy_delay: int = 2,
        exploration_timesteps: int = 10_000,
        update_frequency_in_episodes: int = 50,
        update_start_in_episodes: int = 1_000,
        number_of_batch_updates: int = 1_000,
        batch_size: int = 100,
    ):
        ### ENVIRONMENT VARIABLES ###
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
        ### HYPERPARAMETERS ###
        self.gamma = gamma
        self.polyak = polyak
        # For the scale (i.e. std) and rescaling of noise in the exploration phase
        self.action_noise_scale = action_noise_scale
        self.action_noise_rescaling = action_noise_rescaling
        self.action_noise_rescaling_frequency = action_noise_rescaling_frequency
        # For the scale in the computation of the target actions
        self.target_noise_scale = target_noise_scale
        self.target_noise_clip = target_noise_clip
        # For the delay in the computation of target network updates
        self.policy_delay = policy_delay
        # Number of time steps where sample actions randomly
        self.exploration_timesteps = exploration_timesteps
        # Frequency, start and size of updates
        self.update_frequency_in_episodes = update_frequency_in_episodes
        self.update_start_in_episodes = update_start_in_episodes
        self.number_of_batch_updates = number_of_batch_updates
        self.batch_size = batch_size
        ### Q-NETWORKS (Q1 and Q1) ###
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
        ### POLICY NETWORK ###
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
        # Create the target network and freeze its gradients
        self.policy_network_target = deepcopy(self.policy_network)
        self._freeze_network(self.policy_network_target)
        ### BUFFER ###
        self.buffer = TD3Buffer(
            number_obs=self.num_obs,
            number_actions=self.num_actions,
            size=buffer_size,
            gamma=self.gamma,
            batch_size=self.batch_size,
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
        except:
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
        self.policy_network_target.load_state_dict(pretrained_model["policy_network"])

        print("Loaded {} OK".format(pretrained_model_name))

    def save_checkpoint(
        self, score_avg: float, ckpt_path: str, name: str = None
    ) -> str:
        """Save the weights of the critic and the actor as well as its score. If name is None,
        then use its score as the name.
        """
        # path for current version you're saving (only need ckpt_xxx, not ckpt_xxx.pth.tar)
        if name == None:
            ckpt_path = os.path.join(
                ckpt_path, "td3_ckpt_" + str(round(score_avg, 3)) + ".pth.tar"
            )
        else:
            ckpt_path = os.path.join(
                ckpt_path,
                "td3_ckpt_" + name + "_" + str(round(score_avg, 3)) + ".pth.tar",
            )

        torch.save(
            {
                "q1_network": self.q1_network.state_dict(),
                "q2_network": self.q2_network.state_dict(),
                "policy_network": self.policy_network.state_dict(),
                "q1_network_target": self.q1_network_target.state_dict(),
                "q2_network_target": self.q2_network_target.state_dict(),
                "policy_network_target": self.policy_network_target.state_dict(),
                "score": score_avg,
            },
            ckpt_path,
        )

        return ckpt_path

    def act(self, curr_obs, mode="eval"):
        with torch.no_grad():
            curr_obs = torch.from_numpy(curr_obs).float().to(self.device)
            if mode == "train":
                if self.current_timestep < self.exploration_timesteps:
                    # Do exploration at the beginning of training
                    action = self.env_specs["action_space"].sample()
                else:
                    if (
                        self.current_timestep % self.action_noise_rescaling_frequency
                        == 0
                    ):
                        self.action_noise_scale = (
                            self.action_noise_scale / self.action_noise_rescaling
                        )
                    noise = (
                        self.action_noise_scale * torch.randn(self.num_actions)
                    ).to(self.device)
                    noised_action = self.policy_network(curr_obs) + noise
                    action = torch.clip(
                        noised_action, -self.action_limit, self.action_limit
                    )
                    action = action.data.cpu().numpy()
            else:
                action = torch.clip(
                    self.policy_network(curr_obs), -self.action_limit, self.action_limit
                )
                action = action.data.cpu().numpy()
            return action

    def update(self, curr_obs, action, reward, next_obs, done, timestep):
        # Store experience in buffer
        curr_obs = torch.from_numpy(curr_obs).float()
        action = torch.from_numpy(action).float()
        next_obs = torch.from_numpy(next_obs).float()
        self.buffer.store_experience(curr_obs, action, reward, next_obs, done)
        # Track the current timestep
        self.current_timestep = timestep
        if done:
            self.current_episode += 1
        if self.is_ready_to_train():
            self.train()
            self.episode_of_last_update = self.current_episode

    def is_ready_to_train(self):
        if self.episode_of_last_update is None:
            return self.current_episode > self.update_start_in_episodes
        else:
            return (
                self.current_episode > self.episode_of_last_update
                and self.current_episode % self.update_frequency_in_episodes == 0
            )

    def train(self):
        for j in range(self.number_of_batch_updates):
            # Get training batch
            (
                obs_data,
                action_data,
                reward_data,
                next_obs_data,
                done_data,
            ) = self.buffer.get_training_batch()
            # Get the training targets
            y = self.compute_targets(reward_data, next_obs_data, done_data)
            # Train the Q-network
            self.train_q_networks(obs_data, action_data, y)
            if j % self.policy_delay == 0:
                # Train the policy network
                self.train_policy_network(obs_data)
                # Update the target networks
                self.update_target_networks()

    def _compute_target_actions(self, next_obs_data: torch.Tensor) -> torch.Tensor:
        """Compute the actions produces by the target network"""
        with torch.no_grad():
            raw_target_actions = self.policy_network_target(next_obs_data).squeeze(-1)
            noise = torch.clip(
                torch.rand_like(raw_target_actions) * self.target_noise_scale,
                -self.target_noise_clip,
                self.target_noise_clip,
            )
            noised_target_actions = torch.clip(
                raw_target_actions + noise, -self.action_limit, self.action_limit
            )
            return noised_target_actions

    def compute_targets(
        self,
        reward_data: torch.Tensor,
        next_obs_data: torch.Tensor,
        done_data: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the targets for the Q network."""
        with torch.no_grad():
            target_actions = self._compute_target_actions(next_obs_data)
            min_q_network_target = torch.min(
                self.q1_network_target(next_obs_data, target_actions).squeeze(-1),
                self.q2_network_target(next_obs_data, target_actions).squeeze(-1),
            )
            return reward_data + self.gamma * (1 - done_data) * min_q_network_target

    def train_q_networks(self, obs_data, action_data, y):
        for q_network, q_optimizer in [
            (self.q1_network, self.q1_optimizer),
            (self.q2_network, self.q2_optimizer),
        ]:
            # Train the Q-network
            q_network.zero_grad()
            # Compute q-values
            q_values = q_network(obs_data, action_data).squeeze(-1)
            # Compute loss
            loss = F.mse_loss(q_values, y)
            # Backpropagate
            loss.backward()
            # Take a step
            q_optimizer.step()

    def train_policy_network(self, obs_data):
        # Freeze the Q-network
        self._freeze_network(self.q1_network)
        # Train the policy network
        self.policy_network.zero_grad()
        # Want the policy network to maximize the q-network
        # so minimize the negative of the q-network
        q_values = self.q1_network(obs_data, self.policy_network(obs_data)).squeeze(-1)
        # Compute loss
        loss = -torch.mean(q_values)
        # Backpropagate
        loss.backward()
        # Take a step
        self.policy_optimizer.step()
        # Unfreeze the Q-network
        self._unfreeze_network(self.q1_network)

    def _freeze_network(self, network):
        """Freeze the gradients of the network so that loss cannot backprop
        through it."""
        for param in network.parameters():
            param.requires_grad = False

    def _unfreeze_network(self, network):
        """Unfreeze the network gradients."""
        for param in network.parameters():
            param.requires_grad = True

    def update_target_networks(self):
        """Update the target networks for the q-network and the policy-network via
        a moving average."""
        with torch.no_grad():
            self._polyak_average_update(self.q1_network, self.q1_network_target)
            self._polyak_average_update(self.q2_network, self.q2_network_target)
            self._polyak_average_update(self.policy_network, self.policy_network_target)

    def _polyak_average_update(self, network, target_network):
        """Polyak averaging of network."""
        for target_param, param in zip(
            target_network.parameters(), network.parameters()
        ):
            # Use OpenAI's in-place trick
            target_param.data.mul_(self.polyak)
            target_param.data.add_((1 - self.polyak) * param.data)


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
        layers = [nn.Linear(num_obs, architecture[0]), activation_function()]
        for input_dimension, output_dimension in zip(
            architecture[0:], architecture[1:]
        ):
            layers.append(nn.Linear(input_dimension, output_dimension))
            layers.append(activation_function())
        layers.append(nn.Linear(architecture[-1], num_actions))
        # Specific to Mujoco, add tanh activation as the output activation
        # and rescale it by the action limit
        self.action_limit = action_limit
        layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.action_limit * self.network(x)


class TD3Buffer:
    def __init__(
        self,
        number_obs: int,
        number_actions: int,
        gamma: float,
        size: int = 1_000_000,
        batch_size: int = 100,
    ) -> None:
        """Buffer responsible for storing the experience and the Q target.
        Unlike the VPG and PPO buffer, this buffer is static because random sampling
        is used to train the agent.
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
        self.done_buffer = torch.zeros(self.size)
        self.experience_pointer = 0
        self.effective_size = 0
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Hyperparameters
        self.gamma = gamma
        self.batch_size = batch_size

    def store_experience(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_obs: torch.Tensor,
        done: bool,
    ) -> None:
        """Store the experience and increment pointer. Once the pointer reaches the
        end of the buffer reset it to 0. In this way the buffer acts as a queue."""
        self.action_buffer[self.experience_pointer, :] = action
        self.obs_buffer[self.experience_pointer, :] = obs
        self.next_obs_buffer[self.experience_pointer, :] = next_obs
        self.reward_buffer[self.experience_pointer] = reward
        self.done_buffer[self.experience_pointer] = int(done)
        self.experience_pointer = (self.experience_pointer + 1) % self.size
        self.effective_size = min(self.effective_size + 1, self.size)

    def get_training_batch(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample (batch_size) number of data points from (0, self.experience_pointer - 1)
        and returns tensors for s, a, s', r, and done."""
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
