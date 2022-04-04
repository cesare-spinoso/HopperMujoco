import torch
import time
import random

n_runs = 100
n_episodes = 100
update_at_every_n_episodes = 10
n_obs = 11

start_time = time.time()
obs_list = []

for run in range(n_runs):
    for episode in range(n_episodes):
        episode_length_time_steps = random.randint(1, 200)
        for t in range(episode_length_time_steps):
            obs = torch.rand(n_obs)
            obs_list.append(obs)
        if episode % update_at_every_n_episodes == 0:
            obs_tensor = torch.stack(obs_list)
            # print(f"Final obs tensor shape: {obs_tensor.shape}")
            obs_list = []

end_time = time.time()
print("Time taken: {}".format(end_time - start_time))

start_time = time.time()
obs_tensor = torch.empty((0, n_obs))

for run in range(n_runs):
    for episode in range(n_episodes):
        episode_length_time_steps = random.randint(1, 500)
        for t in range(episode_length_time_steps):
            obs = torch.rand((1, n_obs))
            obs_tensor = torch.cat((obs_tensor, obs))
        if episode % update_at_every_n_episodes == 0:
            # print(f"Final obs tensor shape: {obs_tensor.shape}")
            obs_tensor = torch.empty((0, n_obs))

end_time = time.time()
print("Time taken: {}".format(end_time - start_time))