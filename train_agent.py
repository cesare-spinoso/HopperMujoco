import gym

import argparse
import importlib
import random
import numpy as np
from time import time
from datetime import datetime
from pytest import param

import tensorflow as tf
import torch

import logging

import os
from os import listdir, makedirs
from os.path import isfile, join

import matplotlib.pyplot as plt
from sklearn.metrics import auc
import sklearn.model_selection

from environments import JellyBeanEnv, MujocoEnv


class Logger:
    def __init__(self, path):
        logging.basicConfig(filename=path + "/log.txt",
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S',
                            level=logging.DEBUG)
        self.logger = logging.getLogger()
        self.location = path

    def log(self, msg):
        print(msg)
        self.logger.info(f"{msg}")


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
    agent,
    env,
    env_eval,
    total_timesteps,
    evaluation_freq,
    n_episodes_to_evaluate,
    save_frequency,
    logger,
    name=None,
    visualize=False,
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
    current_mean_acc_rewards = 0

    while timestep < total_timesteps:

        done = False
        curr_obs = env.reset()

        while not done:
            if visualize:
                env.render()

            action = agent.act(curr_obs, mode="train")
            next_obs, reward, done, _ = env.step(action)
            agent.update(curr_obs, action, reward, next_obs, done, timestep)
            curr_obs = next_obs
            timestep += 1
            if timestep % evaluation_freq == 0:
                mean_acc_rewards = evaluate_agent(
                    agent, env_eval, n_episodes_to_evaluate
                )
                logger.log(
                    "timestep: {ts}, acc_reward: {acr:.2f}".format(
                        ts=timestep, acr=mean_acc_rewards
                    )
                )
                array_of_mean_acc_rewards.append(mean_acc_rewards)

                # if we have improvement, save a checkpoint
                if mean_acc_rewards > current_mean_acc_rewards:
                    if name != None:
                        save_path = agent.save_checkpoint(
                            mean_acc_rewards, logger.location, name
                        )
                    else:
                        save_path = agent.save_checkpoint(
                            mean_acc_rewards, logger.location
                        )
                    logger.log("checkpoint saved: {}".format(save_path))
                    current_mean_acc_rewards = mean_acc_rewards

            # if timestep % save_frequency == 0:
            #   # find average rewards
            #   mean_acc_rewards = evaluate_agent(agent, env_eval, n_episodes_to_evaluate)
            #   # call the save_checkpoint model from agent.py
            #   save_path = agent.save_checkpoint(agent.actor_model, agent.critic_model, mean_acc_rewards, logger.location)
            #   logger.log("checkpoint saved: {}".format(save_path))

    return array_of_mean_acc_rewards


def plot_rewards(rewards, location, model_names=None):
    """
    Graphs the average and cumulative reward plots.
    :param rewards: a list, or list of lists, of rewards gained by the model
    :param location: the location in which you want to store the generated .png files
    :param model_names: str of a single model name, or a list of names (if not provided, will use final reward as label)
    """

    if any(isinstance(r, list) for r in rewards):

        # average reward plot
        _ = plt.figure()
        for i in range(len(rewards)):
            if model_names != None:
                reward_label = model_names[i]
            else:
                reward_label = str(round(rewards[i][-1], 2))

            plt.plot(range(len(rewards[i])), rewards[i], label=reward_label)

        plt.ylabel("Average Reward")
        plt.xlabel("Time Step")
        plt.title("Average Reward Over Time")
        plt.legend()
        filename = location + "/avg_rewards_vpg.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

        # cumulative reward plot
        _ = plt.figure()
        for i in range(len(rewards)):
            if model_names != None:
                reward_label = model_names[i]
            else:
                reward_label = str(round(rewards[i][-1], 2))

            cumulative = np.cumsum(rewards[i])
            plt.plot(range(len(cumulative)), cumulative, label=reward_label)

        plt.ylabel("Cumulative Reward")
        plt.xlabel("Time Step")
        plt.title("Cumulative Reward Over Time")
        plt.legend()
        filename = location + "/cum_rewards_vpg.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

    else:  # we have only one reward list, graph only the one

        last_reward = str(round(rewards[-1], 2))
        reward_label = ""

        # average reward plot
        _ = plt.figure()

        if model_names != None:
            if isinstance(model_names, list):
                # select the first element as the model name
                reward_label = model_names[0]
            elif isinstance(model_names, str):
                # use single name
                reward_label = model_names
            else:
                reward_label = last_reward
        else:
            reward_label = last_reward

        plt.plot(range(len(rewards)), rewards, label=reward_label)
        plt.ylabel("Average Reward")
        plt.xlabel("Time Step")
        plt.title("Average Reward Over Time")
        plt.legend()
        filename = location + "/avg_rewards_vpg_{}.png".format(last_reward)
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

        # cumulative reward plot
        _ = plt.figure()
        cumulative = np.cumsum(rewards)
        plt.plot(range(len(cumulative)), cumulative, label=reward_label)
        plt.ylabel("Cumulative Reward")
        plt.xlabel("Time Step")
        plt.title("Cumulative Reward Over Time")
        plt.legend()
        filename = location + "/cum_rewards_vpg_{}.png".format(last_reward)
        plt.savefig(filename, bbox_inches="tight")
        plt.close()


def start_logging():
    """
    Starts an instance of the Logger class to log the training results.
    :return logger: the Logger class instance for this training session
    :return results_path: the location of the log files
    """
    # making sure there is a results folder
    results_folder = os.path.join(os.getcwd(), "results")
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)

    # making a folder for this current training
    results_path = os.path.join(
        os.getcwd(), "results/" + datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S")
    )
    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    # start an instance of the Logger class :)
    logger = Logger(results_path)

    return logger


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--group", type=str, default="GROUP1", help="group directory")
    parser.add_argument(
        "--load",
        type=str,
        default="None",
        help="need folder/filename in results folder (without .pth.tar)",
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
    agent_module = importlib.import_module(args.group + ".agent")

    # Note these can be environment specific and you are free to experiment with what works best for you
    total_timesteps = 2000000
    evaluation_freq = 1000
    n_episodes_to_evaluate = 20
    save_frequency = 100000

    ########################################## training a single agent ##########################################
    # starting a logger - results stored in folder labeled w/ date+time
    logger = start_logging()

    grid = {
        # Found in this paper: https://arxiv.org/pdf/1709.06560.pdf
        "architecture": [(64, 64), (100, 50, 25), (400, 300)],
        "activation": [torch.nn.ReLU, torch.nn.Tanh],
        "number_of_critic_updates_per_actor_update": [100],
        "batch_size_in_time_steps": [5000],
        "actor_lr": [3e-4, 3e-3],
        "critic_lr": [1e-3, 1e-2]
    }
    grid = sklearn.model_selection.ParameterGrid(grid)

    for i, params in enumerate(grid):
        params["actor_architecture"] = params["architecture"]
        params["actor_activation_function"] = params["activation"]
        params["critic_architecture"] = params["architecture"]
        params["critic_activation_function"] = params["activation"]
        del params["architecture"]
        del params["activation"]
        agent = agent_module.Agent(env_specs, **params)

        # load in the pretrained model if one is provided
        if args.load != "None":
            agent.load_weights(os.getcwd(), args.load)
            logger.log(f"Pretrained model loaded: {args.load}")

        logger.log(f"Training start for the following model {i} ... ")
        logger.log(f"{agent.__dict__}")
        start_time = time()
        # you can feed names to train_agent or not --> will change the saved file names/graph labels
        learning_curve = train_agent(
            agent,
            env,
            env_eval,
            total_timesteps,
            evaluation_freq,
            n_episodes_to_evaluate,
            save_frequency,
            logger,
            name=f"m_{i}",
            visualize=False,
        )
        logger.log("Training complete.")

        # log some details of the training
        logger.log(f"\n\nFinal Mean Reward: {round(learning_curve[-1],5)}")
        logger.log(f"Best average reward: {round(max(learning_curve),5)}")
        logger.log(f"Final Cumulative Reward: {round(np.sum(learning_curve),5)}")
        logger.log(
            f"AUC for Mean Reward: {round(auc(range(len(learning_curve)), learning_curve),5)}"
        )
        elapsed_time = time() - start_time
        logger.log(f"Time Elapsed During Training: {elapsed_time}\n")

        # plot learning curves - average reward and cumulative reward
        # you can feed names to plot_rewards or not --> will change the graph labels
        plot_rewards(learning_curve, logger.location, model_names=f"m_{i}")
        logger.log("\nRewards graphed successfully. See {}".format(logger.location))

    ########################################## training multiple models! ##########################################
    # TODO: add hyperparameter changes into this loop
    # model_names = ['v1', 'v2']
    # learning_curves = []
    # for name in model_names:
    #     logger.log("Training start ... ")
    #     start_time = time()
    #     # you can feed names to train_agent or not --> will change the saved file names/graph labels
    #     learning_curve = train_agent(agent, env, env_eval, total_timesteps,
    #         evaluation_freq, n_episodes_to_evaluate, save_frequency, logger, name, visualize=False)
    #     learning_curves.append(learning_curve)
    #     logger.log("Training complete.")

    #     # log some details of the training
    #     logger.log(f"\n\nFinal Mean Reward: {round(learning_curve[-1],5)}")
    #     logger.log(f"Final Cumulative Reward: {round(np.sum(learning_curve),5)}")
    #     logger.log(f"AUC for Mean Reward: {round(auc(range(len(learning_curve)), learning_curve),5)}")
    #     elapsed_time = time() - start_time
    #     logger.log(f'Time Elapsed During Training: {elapsed_time}\n')

    # # plot learning curves - average reward and cumulative reward
    # # you can feed names to plot_rewards or not --> will change the graph labels
    # plot_rewards(learning_curves, logger.location, model_names)
    # logger.log('\nRewards graphed successfully. See {}'.format(logger.location))
