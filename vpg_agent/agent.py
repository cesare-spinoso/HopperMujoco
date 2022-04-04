import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from typing import Tuple
import numpy as np


class Agent:
    """The agent class that is to be filled.
    You are allowed to add any method you
    want to this class.
    """

    TOTAL_TIMESTEPS = 2100000

    def __init__(
        self,
        env_specs: dict,
        gamma: float = 0.99,
        lambda_: float = 0.97,
        actor_lr: float = 3e-4,
        actor_architecture: tuple = (64, 64),
        actor_activation_function: F = nn.ReLU,
        critic_lr: float = 1e-3,
        critic_architecture: tuple = (64, 64),
        critic_activation_function: F = nn.ReLU,
        number_of_critic_updates_per_actor_update: int = 80,
        buffer_type: str = "dynamic",
        batch_size_in_time_steps: int = 5000,
        advantage_computation_method: str = "generalized-advantage-estimation",
        normalize_advantage: bool = False,
        batching_method: str = "most-recent",
    ):
        # TODO: Add hyperparameter tuning for the actor e.g. activation function, learning rate, architecture etc.
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
        self.number_of_critic_updates_per_actor_update = (
            number_of_critic_updates_per_actor_update
        )
        ### BUFFER ###
        self.buffer = Buffer(
            number_obs=self.num_obs,
            number_actions=self.num_actions,
            gamma=self.gamma,
            lambda_=self.lambda_,
            total_timesteps=Agent.TOTAL_TIMESTEPS,
            buffer_type=buffer_type,
            batch_size_in_time_steps=batch_size_in_time_steps,
            advantage_computation_method=advantage_computation_method,
            normalize_advantage=normalize_advantage,
            batching_method=batching_method,
        )

    def load_weights(self, root_path: str, pretrained_model_name: str) -> None:
        # get pretrained model path and load it
        pretrained_model_path = os.path.join(
            root_path, "results", str(pretrained_model_name) + ".pth.tar"
        )

        try:
            pretrained_model = torch.load(pretrained_model_path)
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
                ckpt_path, "vpg_ckpt_" + str(round(score_avg, 3)) + ".pth.tar"
            )
        else:
            ckpt_path = os.path.join(
                ckpt_path,
                "vpg_ckpt_" + name + "_" + str(round(score_avg, 3)) + ".pth.tar",
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

    def act(self, curr_obs: np.array, mode: str = "eval") -> np.array:
        sample_action_as_array = torch.zeros(self.num_actions)
        with torch.no_grad():
            # We can use no grad for both train and eval because we're not
            # keeping track of gradients here so this should make things
            # run faster
            self.actor_model.to(self.device)
            curr_obs = torch.from_numpy(curr_obs).float().to(self.device)
            action_distribution = self.actor_model(curr_obs)
            if mode == "train":
                sample_action = action_distribution.sample()
                sample_action_as_array = sample_action.data.cpu().numpy()
            else:
                sample_action = action_distribution.mean
                sample_action_as_array = sample_action.data.cpu().numpy()

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
                start=start,
                end=end,
            )
            if self.is_ready_to_train():
                start = timestep - self.time_since_last_update
                end = timestep + 1
                # Train the actor
                (
                    action_data,
                    obs_data,
                    advantage_data,
                ) = self.buffer.get_data_for_training_actor(start=start, end=end)
                self.train_actor(
                    action_data=action_data,
                    obs_data=obs_data,
                    advantage_data=advantage_data,
                )
                # Train the critic
                obs_data, return_data = self.buffer.get_data_for_training_critic(
                    start=start, end=end
                )
                self.train_critic(
                    obs_data=obs_data,
                    return_data=return_data,
                    iterations=self.number_of_critic_updates_per_actor_update,
                )
                # Reset last time you trained a batch
                self.time_since_last_update = 0
                # Reset buffer
                self.buffer.reset()
            # Move episode pointer
            self.timestep_of_last_episode = timestep
        self.time_since_last_update += 1

    def is_ready_to_train(self) -> bool:
        # FIXME: This is not the correct condition for being ready to train see Buffer.get_data_for_training
        # for a better explanation
        return (
            int(self.time_since_last_update / self.buffer.batch_size_in_time_steps) >= 1
        )

    def train_actor(
        self,
        action_data: torch.Tensor,
        obs_data: torch.Tensor,
        advantage_data: torch.Tensor,
    ) -> None:
        print("Updating actor")
        self.actor_optimizer.zero_grad()
        # Torch device moving
        self.actor_model.to(self.device)
        advantage_data = advantage_data.to(self.device)
        action_data = action_data.to(self.device)
        obs_data = obs_data.to(self.device)
        # Apply a new forward pass with the batched data
        distribution_of_obs = self.actor_model(obs_data)
        # Compute the log_proba of the distribution using the actions
        log_proba = distribution_of_obs.log_prob(action_data).sum(axis=-1)
        # Compute the per time step loss
        per_time_step_loss = -log_proba * advantage_data
        # Compute the mean loss
        mean_loss = per_time_step_loss.mean()
        # Backpropagate the loss
        mean_loss.backward()
        # Update the parameters
        self.actor_optimizer.step()
        # Put back on cpu
        self.actor_model.to("cpu")

    def train_critic(
        self, obs_data: torch.Tensor, return_data: torch.Tensor, iterations: int
    ) -> None:
        print("Updating critic")
        self.critic_model.to(self.device)
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
        # Put back on cpu
        self.critic_model.to("cpu")


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

    def forward(self, x: torch.Tensor) -> Normal:
        # TODO: Should allow for different distributions (e.g. learnable std, VAEs, etc.)
        mu = self.network(x)
        # NOTE: The forward returns a distribution
        return Normal(mu, torch.ones_like(mu))


class Critic(nn.Module):
    def __init__(
        self, num_obs: int, architecture: tuple, activation_function: F
    ) -> None:
        # TODO: Make the architecture hyper-parameterizable
        super(Critic, self).__init__()
        layers = [nn.Linear(num_obs, architecture[0]), activation_function()]
        for input_dimension, output_dimension in zip(
            architecture[0:], architecture[1:]
        ):
            layers.append(nn.Linear(input_dimension, output_dimension))
            layers.append(activation_function())
        layers.append(nn.Linear(architecture[-1], 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class Buffer:
    ADVANTAGE_COMPUTATION_METHODS = ["td-error", "generalized-advantage-estimation"]
    BATCHING_METHOD = ["most-recent", "random"]
    BUFFER_TYPES = ["static", "dynamic"]

    def __init__(
        self,
        number_obs: int,
        number_actions: int,
        gamma: float,
        lambda_: float,
        total_timesteps: int,
        buffer_type: str = "dynamic",
        batch_size_in_time_steps: int = 4000,
        advantage_computation_method: str = "generalized-advantage-estimation",
        normalize_advantage: bool = True,
        batching_method: str = "most-recent",
    ) -> None:
        # Number of states, actions and total number of time steps
        self.number_obs = number_obs
        self.number_actions = number_actions
        self.total_timesteps = total_timesteps
        self.gamma = gamma
        self.lambda_ = lambda_
        self.buffer_type = buffer_type
        # NOTE: Unlike OpenAI, we do not include state values and log_p in the buffer
        if self.buffer_type == "static":
            self.action_buffer = torch.zeros(
                (self.total_timesteps, self.number_actions)
            )
            self.obs_buffer = torch.zeros((self.total_timesteps, self.number_obs))
            self.reward_buffer = torch.zeros(self.total_timesteps)
            self.experience_pointer = 0
            self.return_buffer = torch.zeros(self.total_timesteps)
            self.advantage_buffer = torch.zeros(self.total_timesteps)
        else:
            self.action_buffer = []
            self.obs_buffer = []
            self.reward_buffer = []
            self.return_buffer = []
            self.advantage_buffer = []
        # Hyperparameters
        self.advantage_computation_method = advantage_computation_method
        self.normalize_advantage = normalize_advantage
        self.batch_size_in_time_steps = batch_size_in_time_steps
        self.batching_method = batching_method

    def store_experience(
        self, obs: torch.Tensor, action: torch.Tensor, reward: float
    ) -> None:
        """Store the experience differently based on the type of buffer"""
        if self.buffer_type == "static":
            self.obs_buffer[self.experience_pointer, :] = obs
            self.action_buffer[self.experience_pointer, :] = action
            self.reward_buffer[self.experience_pointer] = reward
            self.experience_pointer += 1
        else:
            self.obs_buffer.append(obs)
            self.action_buffer.append(action)
            self.reward_buffer.append(reward)

    def reset(self):
        if self.buffer_type == "dynamic":
            self.action_buffer = []
            self.obs_buffer = []
            self.reward_buffer = []
            self.return_buffer = []
            self.advantage_buffer = []

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

        if self.buffer_type == "static":
            self.return_buffer[start:end] = torch.tensor(
                temporary_return_buffer
            ).float()
        else:
            self.return_buffer.extend(temporary_return_buffer)

    def compute_and_store_advantage(
        self,
        critic: nn.Module,
        obs_data: torch.Tensor,
        reward_data: torch.Tensor,
        start: int,
        end: int,
    ) -> None:
        if self.advantage_computation_method == "td-error":
            advantage = self._compute_td_error_advantage(critic, obs_data, reward_data)
        elif self.advantage_computation_method == "generalized-advantage-estimation":
            advantage = self._compute_generalized_advantage_estimation_advantage(
                critic, obs_data, reward_data
            )
        else:
            raise ValueError("Unknown advantage computation method")
        if self.buffer_type == "static":
            self.advantage_buffer[start:end] = advantage
        else:
            self.advantage_buffer.append(advantage)

    def _compute_td_error_advantage(
        self, critic: nn.Module, obs_data: torch.Tensor, reward_data: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            # Compute the TD error
            # FIXME: OpenAI's implementation suggests that the value of the final state
            # should be 0 (line 298 of vpg.py)
            # print(reward_data)
            rewards = torch.cat((reward_data, torch.tensor([0.0])))
            estimated_values = torch.cat(
                (critic(obs_data).squeeze(-1), torch.tensor([0.0]))
            )
            td_error = (
                rewards[:-1] + self.gamma * estimated_values[1:] - estimated_values[:-1]
            )
            return td_error

    def _compute_generalized_advantage_estimation_advantage(
        self, critic: nn.Module, obs_data: torch.Tensor, reward_data: torch.Tensor
    ) -> torch.Tensor:
        # TODO: OpenAI's implementation of the advantage uses something like a lambda-return
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
        if self.buffer_type == "static":
            return self.reward_buffer[start:end]
        else:
            return torch.tensor(self.reward_buffer[-(end - start) :]).float()

    def get_obs_data(self, start: int, end: int) -> torch.Tensor:
        if self.buffer_type == "static":
            return self.obs_buffer[start:end]
        else:
            return torch.stack(self.obs_buffer[-(end - start) :]).float()

    def get_data_for_training_actor(
        self, start: int, end: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # FIXME: OpenAI standardizes the advantage
        if self.buffer_type == "static":
            action_buffer, obs_buffer, advantage_buffer = (
                self.action_buffer[start:end, :],
                self.obs_buffer[start:end, :],
                self.advantage_buffer[start:end],
            )
        else:
            action_buffer, obs_buffer, advantage_buffer = (
                torch.stack(self.action_buffer).float(),
                torch.stack(self.obs_buffer).float(),
                torch.cat(tuple(self.advantage_buffer)).float(),
            )
        if self.normalize_advantage:
            advantage_buffer = (advantage_buffer - advantage_buffer.mean()) / (
                advantage_buffer.std() + 1e-8
            )
        return action_buffer, obs_buffer, advantage_buffer

    def get_data_for_training_critic(
        self, start: int, end: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.buffer_type == "static":
            return (
                self.obs_buffer[start:end, :],
                self.return_buffer[start:end],
            )
        else:
            return (
                torch.stack(self.obs_buffer).float(),
                torch.tensor(self.return_buffer).float(),
            )
