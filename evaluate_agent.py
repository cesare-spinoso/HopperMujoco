import gym

import argparse
import importlib
import random
import numpy as np
from time import time
from datetime import datetime

import tensorflow as tf
import torch

import os
from os import listdir, makedirs
from os.path import isfile, join

import matplotlib.pyplot as plt
from sklearn.metrics import auc

from environments import JellyBeanEnv, MujocoEnv

class Logger():
    def __init__(self, path):
        self.logfile = open(path+'/log.txt',"w+")
        self.location = path
    def log(self, msg):
        print(msg)
        self.logfile.write(f'\n{msg}')


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


def start_logging():
    '''
    Starts an instance of the Logger class to log the training results. 
    :return logger: the Logger class instance for this training session
    :return results_path: the location of the log files
    '''
    # making sure there is a results folder
    results_folder = os.path.join(os.getcwd(),'results')
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    # making a folder for this current training
    results_path = os.path.join(os.getcwd(),'results/' + datetime.now().strftime('%Y-%m-%d_%Hh%Mm%S'))
    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    # start an instance of the Logger class :)
    logger = Logger(results_path)

    return logger


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
    n_evaluations = 1000
    n_episodes_to_evaluate = 50

    ########################################## evaluate a single model ##########################################
    logger.log("Evaluation starting ... ")
    start_time = time()
    average_reward_list = []
    for i in range(n_evaluations):
        average_evaluation_reward = evaluate_agent(agent, env_eval, n_episodes_to_evaluate)
        if i % 100 == 0:
            logger.log(f"Average reward for {args.load}: {np.mean(np.array(average_reward_list))}")
        average_reward_list.append(average_evaluation_reward)
    final_average_reward = np.mean(np.array(average_reward_list))
    logger.log(f"Evaluation complete. Final average reward: {final_average_reward}")
