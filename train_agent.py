import gym

import argparse
import importlib
import time
import random
import numpy as np

import tensorflow as tf
import torch

import os
from os import listdir, makedirs
from os.path import isfile, join

from environments import JellyBeanEnv, MujocoEnv


def evaluate_agent(agent, env, n_episodes_to_evaluate):
    """Evaluates the agent for a provided number of episodes."""
    array_of_acc_rewards = []
    for _ in range(n_episodes_to_evaluate):
        acc_reward = 0
        done = False
        curr_obs = env.reset()
        while not done:
            action = agent.act(curr_obs, mode="eval")
            next_obs, reward, done, _ = env.step(action)
            acc_reward += reward
            curr_obs = next_obs
        array_of_acc_rewards.append(acc_reward)
    return np.mean(np.array(array_of_acc_rewards))


def get_environment(env_type):
    """Generates an environment specific to the agent type."""
    if "jellybean" in env_type:
        env = JellyBeanEnv(gym.make("JBW-COMP579-obj-v1"))
    elif "mujoco" in env_type:
        env = MujocoEnv(gym.make("Hopper-v2"))
    else:
        raise Exception(
            "ERROR: Please define your env_type to be either 'jellybean' or 'mujoco'!"
        )
    return env


def train_agent(
    agent, env, env_eval, total_timesteps, evaluation_freq, n_episodes_to_evaluate, save_frequency
):

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # tf.random.set_seed(seed)
    env.seed(seed)
    env_eval.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    timestep = 0
    array_of_mean_acc_rewards = []

    while timestep < total_timesteps:
        ## uncomment this to visualize the training
        env.render()

        done = False
        curr_obs = env.reset()
        while not done:
            action = agent.act(curr_obs, mode="train")
            next_obs, reward, done, _ = env.step(action)
            agent.update(curr_obs, action, reward, next_obs, done, timestep)
            curr_obs = next_obs

            timestep += 1
            if timestep % evaluation_freq == 0:
                mean_acc_rewards = evaluate_agent(
                    agent, env_eval, n_episodes_to_evaluate
                )
                print(
                    "timestep: {ts}, acc_reward: {acr:.2f}".format(
                        ts=timestep, acr=mean_acc_rewards
                    )
                )
                array_of_mean_acc_rewards.append(mean_acc_rewards)

            if timestep % save_frequency == 0:
              # find average rewards
              mean_acc_rewards = evaluate_agent(agent, env_eval, n_episodes_to_evaluate)
              # call the save_checkpoint model from agent.py
              agent.save_checkpoint(agent.actor_model, agent.critic_model, mean_acc_rewards)

    return array_of_mean_acc_rewards


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--group", type=str, default="GROUP1", help="group directory")
    parser.add_argument("--load", type=str, default="None", help="need filename in save_model folder (without .pth.tar)")

    args = parser.parse_args()

    path = "./" + args.group + "/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    if ("agent.py" not in files) or ("env_info.txt" not in files):
        print("Your GROUP folder does not contain agent.py or env_info.txt!")
        exit()

    with open(path + "env_info.txt") as f:
        lines = f.readlines()
    env_type = lines[0].lower()

    env = get_environment(env_type)
    env_eval = get_environment(env_type)
    if "jellybean" in env_type:
        env_specs = {
            "scent_space": env.scent_space,
            "vision_space": env.vision_space,
            "feature_space": env.feature_space,
            "action_space": env.action_space,
        }
    if "mujoco" in env_type:
        env_specs = {
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        }
    agent_module = importlib.import_module(args.group + ".agent")
    agent = agent_module.Agent(env_specs)

    # load in the save_model if one is provided
    if args.load != "None":
      agent.load_weights(os.getcwd(), args.load)

    # Note these can be environment specific and you are free to experiment with what works best for you
    total_timesteps = 2000000
    evaluation_freq = 1000
    n_episodes_to_evaluate = 20
    save_frequency = 100000

    learning_curve = train_agent(
        agent, env, env_eval, total_timesteps, evaluation_freq, n_episodes_to_evaluate, save_frequency
    )
