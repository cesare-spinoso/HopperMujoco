import argparse
import importlib
import os
from os import listdir
from os.path import isfile, join
from time import time
from copy import deepcopy
import random

import torch
import numpy as np
from sklearn.metrics import auc


from utils.json_utils import log_training_experiment_to_json
from utils.plotting import plot_rewards
from utils.logging_utils import start_logging
from utils.environment import get_environment
from utils.training import train_agent


def record_trained_agent(
    agent,
    env,
    env_eval,
    total_timesteps
):
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    env.seed(0)
    env_eval.seed(0)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    timestep = 0

    while timestep < total_timesteps:

        done = False
        curr_obs = env.reset()

        while not done:
            env.render()

            action = agent.act(curr_obs, mode="eval")
            next_obs, reward, done, _ = env.step(action)
            agent.update(curr_obs, action, reward, next_obs, done, timestep)
            curr_obs = next_obs
            timestep += 1



if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--group", type=str, default="GROUP1", help="group directory")
    parser.add_argument(
        "--load",
        type=str,
        default="None",
        help="need folder/filename in results folder (without .pth.tar)",
    )
    args = parser.parse_args()

    # Process input
    path = "./" + args.group + "/"
    files = [f for f in listdir(path) if isfile(join(path, f))]
    if ("agent.py" not in files) or ("env_info.txt" not in files):
        print("Your GROUP folder does not contain agent.py or env_info.txt!")
        exit()

    # Get environment
    with open(path + "env_info.txt") as f:
        lines = f.readlines()
    env_type = lines[0].lower()

    # Get environment for training and evaluation
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

    # Training and evaluation variables
    total_timesteps = 20_000
    evaluation_freq = 1000
    n_episodes_to_evaluate = 20

    # starting a logger - results stored in folder labeled w/ date+time
    logger = start_logging()
    # Load the agent and try to load hyperparameters
    agent_module = importlib.import_module(args.group + ".agent")
    agent = agent_module.Agent(env_specs)

    # if model provided, we're recording the trained model
    if args.load != "None":
        agent_pretrained = deepcopy(agent)
        agent_pretrained.load_weights(os.getcwd(), args.load)
        
        record_trained_agent(
            agent_pretrained,
            env,
            env_eval,
            total_timesteps
        )
        
    else:
        print("PROVIDE --load ARGUMENT")