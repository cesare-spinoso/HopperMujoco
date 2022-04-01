from cgi import print_arguments
from re import S
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class Agent:
    """The agent class that is to be filled.
    You are allowed to add any method you
    want to this class.
    """

    TOTAL_TIMESTEPS = 2000000

    def __init__(self, env_specs):
        # TODO: Add hyperparameter tuning for the actor e.g. activation function, learning rate, architecture etc.
        ### ENVIRONMENT VARIABLES ###
        self.env_specs = env_specs
        # Number of observations (states) and actions
        self.num_obs = env_specs["observation_space"].shape[0]
        self.num_actions = env_specs["action_space"].shape[0]
        # Keep track of the timestep of the last episode (for the return computation)
        self.timestep_of_last_episode = 0
        self.time_since_last_update = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ### GENERAL HYPERPARAMETERS ###
        self.gamma = 0.99
        self.lambda_ = 0.97
        ### ACTOR ###
        self.actor_model = Actor(
            num_obs=self.num_obs,
            num_actions=self.num_actions,
        )
        self.actor_learning_rate = 0.001
        self.actor_optimizer = torch.optim.Adam(
            self.actor_model.parameters(), lr=self.actor_learning_rate
        )
        ### CRITIC ###
        self.critic_model = Critic(num_obs=self.num_obs)
        self.critic_learning_rate = 0.005  # Critic should stabilize faster
        self.critic_optimizer = torch.optim.Adam(
            self.critic_model.parameters(), lr=self.critic_learning_rate
        )
        # OpenAI Uses 80, another hyperparameter
        self.number_of_critic_updates_per_actor_update = 80
        ### BUFFER ###
        self.buffer = Buffer(
            number_obs=self.num_obs,
            number_actions=self.num_actions,
            gamma=self.gamma,
            lambda_=self.lambda_,
            total_timesteps=Agent.TOTAL_TIMESTEPS,
        )

    def load_weights(self, root_path, pretrained_model_name):
        # get pretrined model path and load it
        pretrained_model_path = os.path.join(root_path, 'results', str(pretrained_model_name)+'.pth.tar')

        try:
            pretrained_model = torch.load(pretrained_model_path)
        except:
            raise Exception("Invalid location for loading pretrained model. You need folder/filename in results folder (without .pth.tar). \
                \nE.g. python3 train_agent.py --group vpg_agent --load 2022-03-31_12h46m44/vpg_ckpt_98.888")

        # load state dict for actor and critic
        self.actor_model.load_state_dict(pretrained_model['actor'])
        self.critic_model.load_state_dict(pretrained_model['critic'])
        
        print("Loaded {} OK".format(pretrained_model_name))
    
    def save_checkpoint(self, actor, critic, score_avg, ckpt_path, name=None):
        # path for current version you're saving (only need ckpt_xxx, not ckpt_xxx.pth.tar)
        if name == None:
            ckpt_path = os.path.join(ckpt_path, 'vpg_ckpt_' + str(round(score_avg,3)) + '.pth.tar')
        else:
            ckpt_path = os.path.join(ckpt_path, 'vpg_ckpt_' + name + '_' + str(round(score_avg,3)) + '.pth.tar')

        torch.save({'actor': actor.state_dict(), 'critic': critic.state_dict(), 'score': score_avg}, ckpt_path)
        
        return ckpt_path

    def act(self, curr_obs, mode="eval"):
        sample_action_as_array = torch.zeros(self.num_actions)
        with torch.no_grad():
            # We can use no grad for both train and eval because we're not
            # keeping track of gradients here so this should make things
            # run faster
            if mode == "train":
                curr_obs = torch.from_numpy(curr_obs).float()
                action_distribution = self.actor_model(curr_obs)
                sample_action = action_distribution.sample()
                sample_action_as_array = sample_action.data.numpy()
            else:
                curr_obs = torch.from_numpy(curr_obs).float()
                action_distribution = self.actor_model(curr_obs)
                sample_action = action_distribution.mean
                sample_action_as_array = sample_action.data.numpy()

        return sample_action_as_array

    def update(self, curr_obs, action, reward, next_obs, done, timestep):
        # Convert the observations to tensors
        curr_obs = torch.from_numpy(curr_obs).float()
        action = torch.from_numpy(action).float()
        next_obs = torch.from_numpy(next_obs).float()
        # Store the latest observation, action and reward
        self.buffer.store_experience(obs=curr_obs, action=action, reward=reward)
        if done:
            reward_data = self.buffer.get_reward_data(
                start=self.timestep_of_last_episode, end=timestep
            )
            obs_data = self.buffer.get_obs_data(
                start=self.timestep_of_last_episode, end=timestep
            )
            # Compute and store the return and advantage
            self.buffer.compute_and_store_return(
                rewards=reward_data, start=self.timestep_of_last_episode, end=timestep
            )
            self.buffer.compute_and_store_advantage(
                critic=self.critic_model,
                obs_data=obs_data,
                reward_data=reward_data,
                start=self.timestep_of_last_episode,
                end=timestep,
            )
            if self.is_ready_to_train():
                # Train the actor
                (
                    action_data,
                    obs_data,
                    advantage_data,
                ) = self.buffer.get_data_for_training_actor(
                    timestep, self.time_since_last_update
                )
                self.train_actor(
                    action_data=action_data,
                    obs_data=obs_data,
                    advantage_data=advantage_data,
                )
                # Train the critic
                # import pdb; pdb.set_trace()
                obs_data, return_data = self.buffer.get_data_for_training_critic(
                    timestep, self.time_since_last_update
                )
                self.train_critic(
                    obs_data=obs_data,
                    return_data=return_data,
                    iterations=self.number_of_critic_updates_per_actor_update,
                )
                # Reset last time you trained a batch
                self.time_since_last_update = 0
            # Move episode pointer
            self.timestep_of_last_episode = timestep
        self.time_since_last_update += 1

    def is_ready_to_train(self):
        # FIXME: This is not the correct condition for being ready to train see Buffer.get_data_for_training
        # for a better explanation
        return (
            int(self.time_since_last_update / self.buffer.batch_size_in_time_steps) >= 1
        )

    def train_actor(self, action_data, obs_data, advantage_data):
        # NOTE: The idea is that all the other models would mostly only change here
        # Compute and store the advantage
        self.actor_optimizer.zero_grad()
        # Apply a new forward pass with the batched data
        self.actor_model.to(self.device)
        obs_data = obs_data.to(self.device)
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

    def train_critic(self, obs_data, return_data, iterations):
        # NOTE: The idea is that all the other models would mostly only change here
        # Train the critic just like a regression model
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


class Actor(nn.Module):
    def __init__(self, num_obs, num_actions):
        # TODO: Make the architecture hyper-parameterizable
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_obs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_actions)

    def forward(self, x):
        # TODO: Should allow for different distributions (e.g. learnable std, VAEs, etc.)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc3(x)
        # NOTE: The forward returns a distribution
        return Normal(mu, torch.ones_like(mu))


class Critic(nn.Module):
    def __init__(self, num_obs):
        # TODO: Make the architecture hyper-parameterizable
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_obs, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Buffer:
    ADVANTAGE_COMPUTATION_METHODS = ["td-error", "generalized-advantage-estimation"]
    BACTHING_METHODS = ["most-recent", "random"]

    def __init__(self, number_obs, number_actions, gamma, lambda_, total_timesteps):
        # Number of states, actions and total number of time steps
        self.number_obs = number_obs
        self.number_actions = number_actions
        self.total_timesteps = total_timesteps
        self.gamma = gamma
        self.lambda_ = lambda_
        # Experience includes action and observation
        # NOTE: Unlike OpenAI, we do not include rewards, state values and log_p in the buffer
        # but we can add this if we want
        self.action_buffer = torch.zeros((total_timesteps, number_actions))
        self.obs_buffer = torch.zeros((total_timesteps, number_obs))
        self.reward_buffer = torch.zeros(total_timesteps)
        self.experience_pointer = 0
        # Return buffer
        self.return_buffer = torch.zeros(total_timesteps)
        # Advantage buffer
        self.advantage_buffer = torch.zeros(total_timesteps)
        # Hyperparameters
        self.advantage_computation_method = "generalized-advantage-estimation"
        self.batch_size_in_time_steps = 5000
        self.batching_method = "most-recent"

    def store_experience(self, obs, action, reward):
        self.obs_buffer[self.experience_pointer, :] = obs
        self.action_buffer[self.experience_pointer, :] = action
        self.reward_buffer[self.experience_pointer] = reward
        self.experience_pointer += 1

    def compute_and_store_return(self, rewards, start, end):
        running_returns = 0
        t = end - start

        for i in range(0, t):
            running_returns = rewards[t - 1 - i] + self.gamma * running_returns
            self.return_buffer[end - 1 - i] = running_returns

    def compute_and_store_advantage(self, critic, obs_data, reward_data, start, end):
        if self.advantage_computation_method == "td-error":
            advantage = self._compute_td_error_advantage(critic, obs_data, reward_data)
        elif self.advantage_computation_method == "generalized-advantage-estimation":
            advantage = self._compute_generalized_advantage_estimation_advantage(
                critic, obs_data, reward_data
            )
        else:
            raise ValueError("Unknown advantage computation method")
        self.advantage_buffer[start:end] = advantage

    def _compute_td_error_advantage(self, critic, obs_data, reward_data):
        with torch.no_grad():
            # Compute the TD error
            # FIXME: OpenAI's implementation suggests that the value of the final state
            # should be 0 (line 298 of vpg.py)
            rewards = torch.cat((reward_data, torch.tensor([0.])))
            estimated_values = torch.cat((critic(obs_data).squeeze(-1), torch.tensor([0.])))
            td_error = (
                rewards[:-1] + self.gamma * estimated_values[1:] - estimated_values[:-1]
            )
            return td_error

    def _compute_generalized_advantage_estimation_advantage(
        self, critic, obs_data, reward_data
    ):
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

    def get_reward_data(self, start, end):
        return self.reward_buffer[start:end]

    def get_obs_data(self, start, end):
            return self.obs_buffer[start:end]

    def get_data_for_training_actor(self, timestep, time_since_last_update):
        # FIXME: For now this retrieves the last self.batch_size_in_time_steps amount of data
        # but this is *incorrect* because it may be chopping an episode
        # FIXME: OpenAI standardizes the advantage
        return (
            self.action_buffer[timestep - time_since_last_update : timestep, :],
            self.obs_buffer[timestep - time_since_last_update : timestep, :],
            self.advantage_buffer[timestep - time_since_last_update : timestep],
        )

    def get_data_for_training_critic(self, timestep, time_since_last_update):
        # NOTE: The return is computed on the fly for efficiency
        return (
            self.obs_buffer[timestep - time_since_last_update : timestep, :],
            self.return_buffer[timestep - time_since_last_update : timestep],
        )
