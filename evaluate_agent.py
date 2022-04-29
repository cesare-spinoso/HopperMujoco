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


def evaluate_agent(agent, env, num_seeds, n_episodes_to_evaluate):
    """Evaluates the agent for a provided number of episodes."""
    rewards_per_seed = []
    for seed in range(num_seeds):
        array_of_acc_rewards = []
        if num_seeds > 1:
            env.seed(seed)
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
        rewards_per_seed.append(np.mean(np.array(array_of_acc_rewards)))
    return np.mean(np.array(rewards_per_seed))


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--group", type=str, default="GROUP1", help="group directory")
    parser.add_argument("--root_path", type=str, help="root path that should go into load_weights", required=True)
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
    agent.load_weights(args.root_path)

    # Note these can be environment specific and you are free to experiment with what works best for you
    n_episodes_to_evaluate = 50
    num_seeds = 5

    mean_reward = evaluate_agent(agent, env_eval, num_seeds, n_episodes_to_evaluate)
    print(mean_reward)