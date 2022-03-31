import gym

import argparse
import importlib
import time
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


def train_agent(agent, env, env_eval, total_timesteps, evaluation_freq, n_episodes_to_evaluate, save_frequency, logger):
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

        done = False
        curr_obs = env.reset()
        while not done:
            # uncomment this to visualize the training
            env.render()

            action = agent.act(curr_obs, mode="train")
            next_obs, reward, done, _ = env.step(action)
            agent.update(curr_obs, action, reward, next_obs, done, timestep)
            curr_obs = next_obs

            timestep += 1
            if timestep % evaluation_freq == 0:
                mean_acc_rewards = evaluate_agent(agent, env_eval, n_episodes_to_evaluate)
                logger.log("timestep: {ts}, acc_reward: {acr:.2f}".format(ts=timestep, acr=mean_acc_rewards))
                array_of_mean_acc_rewards.append(mean_acc_rewards)

            if timestep % save_frequency == 0:
              # find average rewards
              mean_acc_rewards = evaluate_agent(agent, env_eval, n_episodes_to_evaluate)
              # call the save_checkpoint model from agent.py
              save_path = agent.save_checkpoint(agent.actor_model, agent.critic_model, mean_acc_rewards, logger.location)
              logger.log("checkpoint saved: {}".format(save_path))

    return array_of_mean_acc_rewards


def plot_rewards(rewards, location):
    '''
    Graphs the average and cumulative reward plots.
    :param rewards: a list of rewards gained by the model
    :param location: the location in which you want to store the generated .png files
    '''

    last_reward = str(round(rewards[-1],2))

    # average reward plot
    _ = plt.figure()
    plt.plot(range(len(rewards)), rewards)
    plt.ylabel('Average Reward')
    plt.xlabel('Time Step')
    plt.title("Average Reward Over Time")
    filename = location + '/avg_rewards_vpg_{}.png'.format(last_reward)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

    # cumulative reward plot
    _ = plt.figure()
    cumulative = np.cumsum(rewards)
    plt.plot(range(len(cumulative)), cumulative)
    plt.ylabel('Cumulative Reward')
    plt.xlabel('Time Step')
    plt.title("Cumulative Reward Over Time")
    filename = location + '/cum_rewards_vpg_{}.png'.format(last_reward)
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


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
    parser.add_argument("--load", type=str, default="None", help="need filename in results folder (without .pth.tar)")

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
    if args.load != "None":
      agent.load_weights(os.getcwd(), args.load)
      logger.log(f'Pretrained model loaded: {args.load}')

    # Note these can be environment specific and you are free to experiment with what works best for you
    total_timesteps = 2000000
    evaluation_freq = 1000
    n_episodes_to_evaluate = 20
    save_frequency = 100000

    # TODO: save the learning_curve to file --> will be useful to plot comparison graphs when hyp. tuning
    logger.log("Training start ... ")
    start_time = time()
    learning_curve = train_agent(agent, env, env_eval, total_timesteps, 
        evaluation_freq, n_episodes_to_evaluate, save_frequency, logger)
    logger.log("Training complete.")

    # log some details of the training - TODO: add more stats here?
    logger.log(f"\n\nFinal Mean Reward: {round(learning_curve[-1],5)}")
    logger.log(f"Final Cumulative Reward: {round(np.sum(learning_curve),5)}")
    elapsed_time = time() - start_time
    logger.log(f'Time Elapsed During Training: {elapsed_time}')

    # plot learning curves - average reward and cumulative reward
    plot_rewards(learning_curve, logger.location)
    logger.log('\nRewards graphed successfully. See {}'.format(logger.location))

