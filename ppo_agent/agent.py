import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple
import os


class Agent:
    """The agent class that is to be filled.
    You are allowed to add any method you
    want to this class.
    """

    def __init__(
        self,
        env_specs,
        gamma: float = 0.99,
        lambda_: float = 0.97,
        actor_lr: float = 3e-4,
        actor_architecture: tuple = (64, 64),
        actor_activation_function: F = nn.ReLU,
        critic_lr: float = 1e-3,
        critic_architecture: tuple = (64, 64),
        critic_activation_function: F = nn.ReLU,
        number_of_critic_updates: int = 80,
        number_of_actor_updates: int = 80,
        kl_threshold: float = 0.015,
        clip_rate: float = 0.2,
        batch_size_in_time_steps: int = 5000,
        advantage_computation_method: str = "generalized-advantage-estimation",
        normalize_advantage: bool = False,
    ):
        ### ENVIRONMENT VARIABLES ###
        self.env_specs = env_specs
        # Number of observations (states) and actions
        self.num_obs = env_specs["observation_space"].shape[0]
        self.num_actions = env_specs["action_space"].shape[0]
        # Keep track of the timestep of the last episode (for the return and advantage computation)
        self.timestep_of_last_episode = 0
        self.time_since_last_update = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ### GENERAL HYPERPARAMETERS ###
        self.gamma = gamma
        self.lambda_ = lambda_
        self.kl_threshold = kl_threshold
        self.clip_rate = clip_rate
        ### ACTOR ###
        self.actor_model = Actor(
            num_obs=self.num_obs,
            num_actions=self.num_actions,
            architecture=actor_architecture,
            activation_function=actor_activation_function,
        )
        self.actor_learning_rate = actor_lr
        self.actor_optimizer = torch.optim.Adam(
            self.actor_model.parameters(), lr=self.actor_learning_rate
        )
        ### CRITIC ###
        self.critic_model = Critic(
            num_obs=self.num_obs,
            architecture=critic_architecture,
            activation_function=critic_activation_function,
        )
        self.critic_learning_rate = critic_lr  # Critic should stabilize faster
        self.critic_optimizer = torch.optim.Adam(
            self.critic_model.parameters(), lr=self.critic_learning_rate
        )
        # OpenAI Uses 80, another hyperparameter
        self.number_of_critic_updates = number_of_critic_updates
        self.number_of_actor_updates = number_of_actor_updates
        ### BUFFER ###
        self.buffer = PPOBuffer(
            number_obs=self.num_obs,
            number_actions=self.num_actions,
            gamma=self.gamma,
            lambda_=self.lambda_,
            batch_size_in_time_steps=batch_size_in_time_steps,
            advantage_computation_method=advantage_computation_method,
            normalize_advantage=normalize_advantage,
        )

    def load_weights(self, root_path: str, pretrained_model_name: str = None) -> None:
        # get pretrained model path and load it
        if pretrained_model_name is None:
            pretrained_model_path = os.path.join(root_path, "model.pth.tar")
        else:
            pretrained_model_path = os.path.join(
                root_path, str(pretrained_model_name) + ".pth.tar"
            )

        try:
            pretrained_model = torch.load(pretrained_model_path, map_location=torch.device(self.device))
        except:
            raise Exception(
                "Invalid location for loading pretrained model. You need folder/filename in results folder (without .pth.tar). \
                \nE.g. python3 train_agent.py --group vpg_agent --load 2022-03-31_12h46m44/vpg_ckpt_98.888"
            )

        # load state dict for actor and critic
        self.actor_model.load_state_dict(pretrained_model["actor"])
        self.critic_model.load_state_dict(pretrained_model["critic"])

        print("Loaded {} OK".format(pretrained_model_name))

    def save_checkpoint(
        self, score_avg: float, ckpt_path: str, name: str = None
    ) -> str:
        # path for current version you're saving (only need ckpt_xxx, not ckpt_xxx.pth.tar)
        if name == None:
            ckpt_path = os.path.join(
                ckpt_path, "ppo_ckpt_" + str(round(score_avg, 3)) + ".pth.tar"
            )
        else:
            ckpt_path = os.path.join(
                ckpt_path,
                "ppo_ckpt_" + name + ".pth.tar",
            )

        torch.save(
            {
                "actor": self.actor_model.state_dict(),
                "critic": self.critic_model.state_dict(),
                "score": score_avg,
            },
            ckpt_path,
        )

        return ckpt_path

    def act(self, curr_obs, mode="eval"):
        sample_action_as_array = torch.zeros(self.num_actions)
        with torch.no_grad():
            # We can use no grad for both train and eval because we're not
            # keeping track of gradients here so this should make things
            # run faster
            # Do this computation on the cpu
            self.actor_model.cpu()
            curr_obs = torch.from_numpy(curr_obs).float()
            action_distribution = self.actor_model(curr_obs)
            if mode == "train":
                sample_action = action_distribution.sample()
                self.buffer.compute_and_store_log_proba(
                    action_distribution, sample_action
                )
                sample_action_as_array = sample_action.data.cpu().numpy()
            else:
                sample_action = action_distribution.mean
                sample_action_as_array = sample_action.data.cpu().numpy()
        # Place back on gpu if there is one
        self.actor_model.to(self.device)
        return sample_action_as_array

    def update(
        self,
        curr_obs: np.array,
        action: np.array,
        reward: np.array,
        next_obs: np.array,
        done: bool,
        timestep: int,
    ) -> None:
        # Convert the observations to tensors
        curr_obs = torch.from_numpy(curr_obs).float()
        action = torch.from_numpy(action).float()
        next_obs = torch.from_numpy(next_obs).float()
        # Store the latest observation, action and reward
        self.buffer.store_experience(obs=curr_obs, action=action, reward=reward)
        if done:
            start = (
                0
                if self.timestep_of_last_episode == 0
                else self.timestep_of_last_episode + 1
            )
            end = timestep + 1
            reward_data = self.buffer.get_reward_data(start=start, end=end)
            obs_data = self.buffer.get_obs_data(start=start, end=end)
            # Compute and store the return and advantage
            self.buffer.compute_and_store_return(
                rewards=reward_data, start=start, end=end
            )
            self.buffer.compute_and_store_advantage(
                critic=self.critic_model,
                obs_data=obs_data,
                reward_data=reward_data,
            )
            if self.is_ready_to_train():
                # Train the actor
                (
                    action_data,
                    obs_data,
                    advantage_data,
                    log_proba_data,
                ) = self.buffer.get_data_for_training_actor()
                self.train_actor(
                    action_data=action_data,
                    obs_data=obs_data,
                    advantage_data=advantage_data,
                    log_proba_data=log_proba_data,
                    iterations=self.number_of_actor_updates,
                )
                # Train the critic
                obs_data, return_data = self.buffer.get_data_for_training_critic()
                self.train_critic(
                    obs_data=obs_data,
                    return_data=return_data,
                    iterations=self.number_of_critic_updates,
                )
                # Reset last time you trained a batch
                self.time_since_last_update = 0
                # Reset buffer
                self.buffer.reset()
            # Move episode pointer
            self.timestep_of_last_episode = timestep
        self.time_since_last_update += 1

    def is_ready_to_train(self) -> bool:
        return (
            int(self.time_since_last_update / self.buffer.batch_size_in_time_steps) >= 1
        )

    def train_actor(
        self,
        action_data: torch.Tensor,
        obs_data: torch.Tensor,
        advantage_data: torch.Tensor,
        log_proba_data: torch.Tensor,
        iterations: int,
    ) -> None:
        # Torch device moving
        advantage_data = advantage_data.to(self.device)
        action_data = action_data.to(self.device)
        obs_data = obs_data.to(self.device)
        log_proba_old = log_proba_data.to(self.device)
        # Apply forward passes with the batched data with early stopping (OpenAI)
        for _ in range(iterations):
            self.actor_optimizer.zero_grad()
            # Recompute pi(a|s)
            distribution_of_action = self.actor_model(obs_data)
            # Compute the log_proba of the distribution using the actions
            log_proba = distribution_of_action.log_prob(action_data).sum(axis=-1)
            # Compute pi(a|s)/pi_old(a|s) by using exp(log) trick
            ratio = torch.exp(log_proba - log_proba_old)
            # PPO clipping function
            clipped_ratio = torch.clamp(
                ratio, 1.0 - self.clip_rate, 1.0 + self.clip_rate
            )
            # Compute the per time step PPO loss
            per_time_step_loss = -(torch.min(ratio * advantage_data, clipped_ratio))
            # Compute the mean loss
            mean_loss = per_time_step_loss.mean()
            # Compute the approximate kl divergence for early stopping
            approx_kl_divergence = (log_proba_old - log_proba).mean().item()
            if approx_kl_divergence > self.kl_threshold:
                break
            # Backpropagate the loss
            mean_loss.backward()
            # Update the parameters
            self.actor_optimizer.step()

    def train_critic(
        self, obs_data: torch.Tensor, return_data: torch.Tensor, iterations: int
    ) -> None:
        obs_data = obs_data.to(self.device)
        return_data = return_data.to(self.device)
        for _ in range(iterations):
            self.critic_model.zero_grad()
            # Apply a new forward pass with the batched data
            estimated_value = self.critic_model(obs_data)
            # Compute the loss
            loss = F.mse_loss(estimated_value.squeeze(-1), return_data)
            # Backpropagate the loss
            loss.backward()
            # Update the parameters
            self.critic_optimizer.step()


class Actor(nn.Module):
    def __init__(
        self,
        num_obs: int,
        num_actions: int,
        architecture: tuple,
        activation_function: F,
    ) -> None:
        super(Actor, self).__init__()
        layers = [nn.Linear(num_obs, architecture[0]), activation_function()]
        for input_dimension, output_dimension in zip(
            architecture[0:], architecture[1:]
        ):
            layers.append(nn.Linear(input_dimension, output_dimension))
            layers.append(activation_function())
        layers.append(nn.Linear(architecture[-1], num_actions))
        self.network = nn.Sequential(*layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Normal:
        # TODO: Should allow for different distributions (e.g. learnable std, VAEs, etc.)
        mu = self.network(x)
        # NOTE: The forward returns a distribution
        return Normal(mu, torch.ones_like(mu))


class Critic(nn.Module):
    def __init__(
        self, num_obs: int, architecture: tuple, activation_function: F
    ) -> None:
        super(Critic, self).__init__()
        layers = [nn.Linear(num_obs, architecture[0]), activation_function()]
        for input_dimension, output_dimension in zip(
            architecture[0:], architecture[1:]
        ):
            layers.append(nn.Linear(input_dimension, output_dimension))
            layers.append(activation_function())
        layers.append(nn.Linear(architecture[-1], 1))
        self.network = nn.Sequential(*layers)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PPOBuffer:
    ADVANTAGE_COMPUTATION_METHODS = ["td-error", "generalized-advantage-estimation"]

    def __init__(
        self,
        number_obs: int,
        number_actions: int,
        gamma: float,
        lambda_: float,
        batch_size_in_time_steps: int,
        advantage_computation_method: str = "generalized-advantage-estimation",
        normalize_advantage: bool = True,
    ) -> None:
        # Number of states, actions and total number of time steps
        self.number_obs = number_obs
        self.number_actions = number_actions
        self.gamma = gamma
        self.lambda_ = lambda_
        # Only consider dynamic buffer for simplicity (and thus most-recent batching only)
        self.action_buffer = []
        self.obs_buffer = []
        self.reward_buffer = []
        self.return_buffer = []
        self.advantage_buffer = []
        self.log_proba_buffer = []
        # Hyperparameters
        self.advantage_computation_method = advantage_computation_method
        self.normalize_advantage = normalize_advantage
        self.batch_size_in_time_steps = batch_size_in_time_steps

    def store_experience(
        self, obs: torch.Tensor, action: torch.Tensor, reward: float
    ) -> None:
        """Store the experience differently based on the type of buffer"""
        self.obs_buffer.append(obs)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)

    def reset(self):
        self.action_buffer = []
        self.obs_buffer = []
        self.reward_buffer = []
        self.return_buffer = []
        self.advantage_buffer = []
        self.log_proba_buffer = []

    def compute_and_store_log_proba(
        self, action_distribution: Normal, sample_action: torch.Tensor
    ):
        with torch.no_grad():
            log_proba = (
                action_distribution.log_prob(sample_action).sum(axis=-1).cpu().detach()
            )
            self.log_proba_buffer.append(log_proba)

    def compute_and_store_return(
        self, rewards: torch.Tensor, start: int, end: int
    ) -> None:
        t = end - start
        assert t == len(
            rewards
        ), f"The length of the rewards tensor ({len(rewards)}) does not match end - start = {end - start}"
        running_returns = 0
        temporary_return_buffer = [0] * t

        for i in reversed(range(0, t)):
            running_returns = rewards[i] + self.gamma * running_returns
            temporary_return_buffer[i] = running_returns

        self.return_buffer.extend(temporary_return_buffer)

    def compute_and_store_advantage(
        self,
        critic: nn.Module,
        obs_data: torch.Tensor,
        reward_data: torch.Tensor,
    ) -> None:
        if self.advantage_computation_method == "td-error":
            advantage = self._compute_td_error_advantage(critic, obs_data, reward_data)
        elif self.advantage_computation_method == "generalized-advantage-estimation":
            advantage = self._compute_generalized_advantage_estimation_advantage(
                critic, obs_data, reward_data
            )
        else:
            raise ValueError("Unknown advantage computation method")
        self.advantage_buffer.append(advantage)

    def _compute_td_error_advantage(
        self, critic: nn.Module, obs_data: torch.Tensor, reward_data: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            # Place the critic on the cpu
            critic.cpu()
            # Compute the TD error
            rewards = torch.cat((reward_data, torch.tensor([0.0])))
            estimated_values = torch.cat(
                (critic(obs_data).squeeze(-1), torch.tensor([0.0]))
            )
            td_error = (
                rewards[:-1] + self.gamma * estimated_values[1:] - estimated_values[:-1]
            )
            critic.to(critic.device)
            return td_error

    def _compute_generalized_advantage_estimation_advantage(
        self, critic: nn.Module, obs_data: torch.Tensor, reward_data: torch.Tensor
    ) -> torch.Tensor:
        td_error = self._compute_td_error_advantage(critic, obs_data, reward_data)
        generalized_advantage = torch.zeros_like(td_error)
        running_advantage = 0
        for i in reversed(range(0, len(generalized_advantage))):
            running_advantage = (
                td_error[i] + self.gamma * self.lambda_ * running_advantage
            )
            generalized_advantage[i] = running_advantage
        return generalized_advantage

    def get_reward_data(self, start: int, end: int) -> torch.Tensor:
        # The reward buffer is a list of floats
        return torch.tensor(self.reward_buffer[-(end - start) :]).float()

    def get_obs_data(self, start: int, end: int) -> torch.Tensor:
        # The obs buffer is a list of (1, 11) tensors
        return torch.stack(self.obs_buffer[-(end - start) :]).float()

    def get_data_for_training_actor(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # FIXME: OpenAI standardizes the advantage
        # The obs and action buffers are lists of (1, 11) tensors
        # The advantage buffer is a list of 1-D tensors
        action_buffer, obs_buffer, advantage_buffer, log_proba_buffer = (
            torch.stack(self.action_buffer).float(),
            torch.stack(self.obs_buffer).float(),
            torch.cat(tuple(self.advantage_buffer)).float(),
            torch.tensor(self.log_proba_buffer).float(),
        )
        if self.normalize_advantage:
            advantage_buffer = (advantage_buffer - advantage_buffer.mean()) / (
                advantage_buffer.std() + 1e-8
            )
        return action_buffer, obs_buffer, advantage_buffer, log_proba_buffer

    def get_data_for_training_critic(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.stack(self.obs_buffer).float(),
            torch.tensor(self.return_buffer).float(),
        )
