import gym

import argparse
import importlib
import numpy as np

import os
from os import listdir
from os.path import isfile, join
from pathlib import Path

from environments import JellyBeanEnv, MujocoEnv

from utils.logging import start_logging
from utils.json import get_json_data


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
    parser.add_argument(
        "--model_path",
        type=str,
        default="None",
        help="Relative path to model (without .pth.tar) with respect GROUP_13 directory",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default="None",
        help="Path to json file containing the paths of the models",
    )

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

    # starting a logger - results stored in folder labeled w/ date+time
    logger = start_logging()

    # load in the pretrained model if one is provided
    agent = None
    if args.model_path == "None" and args.json_path == "None":
        raise ValueError(
            "You must load in a pretrained model or the log.json file for the evaluation script."
        )
    elif args.model_path != "None":
        agent_module = importlib.import_module(args.group + ".agent")
        # Load single agent (requires manually changing the parameters to the agent's constructor in the
        # next line)
        agent = agent_module.Agent(env_specs)
        agent.load_weights(os.getcwd(), args.model_path)
        logger.log(f"Pretrained model loaded: {args.model_path}")
    else:
        # Load json file containing the paths of the models
        json_data = get_json_data(args.json_path)
        # Load the hyperparameters corresponding to the json file
        hyperparameter_module = importlib.import_module(args.group + ".hyperparameters")
        grid = hyperparameter_module.hyperparameter_grid
        # Load the different agents
        agent_module = importlib.import_module(args.group + ".agent")
        agents = [agent_module.Agent(env_specs, **params) for params in grid]
        for model, json in zip(agents, json_data):
            model.load_weights(
                os.path.join(os.getcwd(), args.model_path, "results"),
                Path(Path(json["path_to_best_model"]).stem).stem,
            )

    # "Out-of-sample" Evaluation
    n_episodes_to_evaluate = 100

    ########################################## evaluate a single/multiple model(s) ##########################################
    if agent is not None:
        logger.log("Evaluation starting ... ")
        # Calculate the average (out-of-sample) reward
        average_reward_per_episode = evaluate_agent(
            agent, env_eval, n_episodes_to_evaluate
        )
        # TODO: Add Sabina's sample efficiency calculation here
        logger.log(f"Average reward per episode: {average_reward_per_episode}")
    else:
        for agent, json in zip(agents, json_data):
            logger.log(f"Evaluation starting for ... {json['model_name']}")
            # Calculate the average (out-of-sample) reward
            average_reward_per_episode = evaluate_agent(
                agent, env_eval, n_episodes_to_evaluate
            )
            logger.log(f"Average reward per episode: {average_reward_per_episode}")
            # TODO: Add Sabina's sample efficiency calculation here
