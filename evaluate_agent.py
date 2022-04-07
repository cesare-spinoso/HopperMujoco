import gym

import argparse
import importlib
import numpy as np

import os
from os import listdir, makedirs
from os.path import isfile, join

from environments import JellyBeanEnv, MujocoEnv

from utils.logging import start_logging


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--group", type=str, default="GROUP1", help="group directory")
    parser.add_argument("--load", type=str, default="None", help="need folder/filename in results folder (without .pth.tar)")

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

    # starting a logger - results stored in folder labeled w/ date+time
    logger = start_logging()

    # load in the pretrained model if one is provided
    if args.load == "None":
        raise ValueError("You must load in a pretrained model for the evaluation script.")
    else:
        agent.load_weights(os.getcwd(), args.load)
        logger.log(f'Pretrained model loaded: {args.load}')

    # "Out-of-sample" Evaluation 
    n_episodes_to_evaluate = 100

    # TODO: Add Sabina's sample efficiency calculation here?

    ########################################## evaluate a single model ##########################################
    logger.log("Evaluation starting ... ")
    # Calculate the average (out-of-sample) reward
    average_reward_list = []
    average_reward_per_episode = evaluate_agent(agent, env_eval, n_episodes_to_evaluate)

    logger.log(f"Average reward per episode: {average_reward_per_episode} +/-")
