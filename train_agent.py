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


def calc_sample_efficiency(agent_module, env_specs, env, env_eval):
    start_time = time()

    eval_performances = []
    total_timesteps = 100000
    evaluation_freq = 1000
    n_episodes_to_evaluate = 20
    save_frequency = 100000

    for seed in range(5):
        print(f'starting training on {seed+1} of 5...')

        # set a random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.seed(seed)
        env_eval.seed(seed)

        # randomly initialize the weights for the agent
        # TODO: ensure this random initialization actually leads to different things... --> try printing first reward for all 5
        agent = agent_module.Agent(env_specs)

        # train for 100K steps, evaluating every 1000
        seed_performance = train_agent(agent, env, env_eval, total_timesteps, evaluation_freq, n_episodes_to_evaluate, save_frequency, logger=None)
        eval_performances.append(seed_performance)

    # calculate the AUC for the mean performance
    mean_performance = np.mean(eval_performances, axis=0)
    sample_efficiency = auc(range(len(mean_performance)), mean_performance)

    time_elapsed = time() - start_time
    
    return sample_efficiency, time_elapsed


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


def train_agent(agent, env, env_eval, total_timesteps, evaluation_freq, n_episodes_to_evaluate, 
    save_frequency, logger=None, name=None, visualize=False):
    # seed = 0
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # # tf.random.set_seed(seed)
    # env.seed(seed)
    # env_eval.seed(seed)

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
                mean_acc_rewards = evaluate_agent(agent, env_eval, n_episodes_to_evaluate)
                if logger:
                    logger.log("timestep: {ts}, acc_reward: {acr:.2f}".format(ts=timestep, acr=mean_acc_rewards))
                array_of_mean_acc_rewards.append(mean_acc_rewards)

                # if we have improvement, and we're logging, save a checkpoint
                if mean_acc_rewards > current_mean_acc_rewards:
                    if name != None and logger != None:
                        save_path = agent.save_checkpoint(mean_acc_rewards, logger.location, name)
                    elif logger != None:
                        save_path = agent.save_checkpoint(mean_acc_rewards, logger.location)

                    if logger:
                        logger.log("checkpoint saved: {}".format(save_path))
                    current_mean_acc_rewards = mean_acc_rewards

            # if timestep % save_frequency == 0:
            #   # find average rewards
            #   mean_acc_rewards = evaluate_agent(agent, env_eval, n_episodes_to_evaluate)
            #   # call the save_checkpoint model from agent.py
            #   save_path = agent.save_checkpoint(agent.actor_model, agent.critic_model, mean_acc_rewards, logger.location)
            #   if logger:
            #       logger.log("checkpoint saved: {}".format(save_path))

    return array_of_mean_acc_rewards


def plot_rewards(rewards, location, names=None, time_step=None):
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
            if names != None:
                reward_label = names[i]
            else:
                reward_label = str(round(rewards[i][-1], 2))
            if time_step: # range(start, stop, step)
                plt.plot(range(0, len(rewards[i])*time_step, time_step), rewards[i], label=reward_label)
            else: 
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
            if names != None:
                reward_label = names[i]
            else:
                reward_label = str(round(rewards[i][-1], 2))

            cumulative = np.cumsum(rewards[i])
            if time_step: # range(start, stop, step)
                plt.plot(range(0, len(cumulative)*time_step, time_step), cumulative, label=reward_label)
            else: 
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

        if names != None:
            if isinstance(names, list):
                # select the first element as the model name
                reward_label = names[0]
            elif isinstance(names, str):
                # use single name
                reward_label = names
            else:
                reward_label = last_reward
        else:
            reward_label = last_reward

        if time_step: # range(start, stop, step)
            plt.plot(range(0, len(rewards)*time_step, time_step), rewards, label=reward_label)
        else: 
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
        if time_step: # range(start, stop, step)
            plt.plot(range(0, len(cumulative)*time_step, time_step), cumulative, label=reward_label)
        else: 
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


def train_and_evaluate(agent, env, env_eval, total_timesteps, evaluation_freq, n_episodes_to_evaluate, save_frequency, 
    logger=None, name=None, visualize=False):
    
    if logger: logger.log("Training start ... ")
    start_time = time()

    # you can feed names to train_agent or not --> will change the saved file names/graph labels
    learning_curve = train_agent(agent, env, env_eval, total_timesteps, evaluation_freq, n_episodes_to_evaluate, save_frequency,
        logger=logger, name=name, visualize=False)
    if logger: logger.log("Training complete.")

    # log some details of the training
    elapsed_time = time() - start_time
    if logger:
        logger.log(f"\n\nFinal Mean Reward: {round(learning_curve[-1],5)}")
        logger.log(f"Final Cumulative Reward: {round(np.sum(learning_curve),5)}")
        logger.log(f"AUC for Mean Reward: {round(auc(range(len(learning_curve)), learning_curve),5)}")
        logger.log(f'Time Elapsed During Training: {elapsed_time}\n')

    # plot learning curves - average reward and cumulative reward
    # you can feed names to plot_rewards or not --> will change the graph labels
    if logger: plot_rewards(learning_curve, logger.location, names=name, time_step=evaluation_freq)
    if name:
        if logger: logger.log('\n{} rewards graphed successfully. See {}'.format(name, logger.location))
    else:
        if logger: logger.log('\nRewards graphed successfully. See {}'.format(logger.location))

    return learning_curve


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
    logger = start_logging()

    # Note these can be environment specific and you are free to experiment with what works best for you
    total_timesteps = 2000000
    evaluation_freq = 1000
    n_episodes_to_evaluate = 20
    save_frequency = 100000

    # setting a seed --> removed from train_agent so that calc_sample_efficiency can work
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    env_eval.seed(seed)

    ########################################## training a single model ##########################################
    # load in the pretrained model if one is provided
    agent = agent_module.Agent(env_specs) # creates agent with our default parameters
    if args.load != "None":
        agent.load_weights(os.getcwd(), args.load)
        logger.log(f"Pretrained model loaded: {args.load}")
    train_and_evaluate(agent, env, env_eval, total_timesteps, evaluation_freq, n_episodes_to_evaluate, save_frequency,
        logger, visualize=False)

    ########################################## calculating sample efficiency ##########################################
    logger.log('Starting calculation of sample efficiency...\n')
    sample_efficiency, time_elapsed = calc_sample_efficiency(agent_module, env_specs, env, env_eval)
    logger.log(f'Sample Efficiency: {sample_efficiency}')
    logger.log(f'Time to Calculate Sample Efficiency: {time_elapsed}')
