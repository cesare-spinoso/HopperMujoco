import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent:
    """The agent class that is to be filled.
    You are allowed to add any method you
    want to this class.
    """

    def __init__(self, env_specs):
        self.env_specs = env_specs
        self.actor_model = Actor(
            num_inputs=env_specs["observation_space"].shape[0],
            num_outputs=env_specs["action_space"].shape[0],
        )
        # TODO: Create an optimizer and loss

    def load_weights(self, root_path):
        # Add root_path in front of the path of the saved network parameters
        # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters
        pass

    def act(self, curr_obs, mode="eval"):
        if mode == "train":
            self.actor_model.train()
        else:
            self.actor_model.eval()
        curr_obs = torch.from_numpy(curr_obs).float()
        actor_forward = self.actor_model(curr_obs)
        sample_action = torch.normal(actor_forward)
        sample_action_as_array = sample_action.data.numpy()
        return sample_action_as_array

    def update(self, curr_obs, action, reward, next_obs, done, timestep):
        # TODO: If doing not doing an update, just collect otherwise do an update
        # TODO: An update looks like
        # TODO: What is 1/|D_k| in the spinup version
        # 1. Computing the returns and advantages with the critic
        # 2. Update with policy grad
        # 3. Update with critic grad
        pass

    def create_optimizer():
        pass

    def create_loss():
        pass

    def get_returns():
        pass


class Actor(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_inputs, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, num_outputs)

    def forward(self, x):
        # TODO: Do we want to add standard deviation as parametrized?
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.fc3(x)
        return mu
    
    def _distribution():
        # Get the action distribution
        pass


class Critic(nn.Module):
    pass


class Buffer:
    def __init__(number_obs, number_actions, total_timesteps):
        pass

    def store(obs, action, reward):
        pass

    def get():
        pass
