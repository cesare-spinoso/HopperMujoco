"""
A script to train the best hyperparameter set 5 times.
"""

import argparse
import importlib
import os
from os import listdir
from os.path import isfile, join
from time import time

import numpy as np
from sklearn.metrics import auc
import torch

from utils.json_utils import log_training_experiment_to_json
from utils.plotting import plot_best_model_rewards
from utils.logging_utils import start_logging
from utils.environment import get_environment
from utils.training import train_agent


def train_and_evaluate(
    agent,
    env,
    env_eval,
    total_timesteps,
    evaluation_freq,
    n_episodes_to_evaluate,
    save_checkpoints,
    logger=None,
    name=None,
    visualize=False,
):

    if logger:
        logger.log("Training start ... ")
    start_time = time()

    # you can feed names to train_agent or not --> will change the saved file names/graph labels
    learning_curve = train_agent(
        agent,
        env,
        env_eval,
        total_timesteps,
        evaluation_freq,
        n_episodes_to_evaluate,
        save_checkpoints,
        logger=logger,
        name=name,
        visualize=visualize,
    )
    if logger:
        logger.log("Training complete.")

    # log some details of the training
    elapsed_time = time() - start_time
    if logger:
        logger.log(f"\n\nFinal Mean Reward: {round(learning_curve[-1],5)}")
        logger.log(f"Final Cumulative Reward: {round(np.sum(learning_curve),5)}")
        logger.log(
            f"AUC for Mean Reward: {round(auc(range(len(learning_curve)), learning_curve),5)}"
        )
        logger.log(f"Time Elapsed During Training: {elapsed_time}")

    # plot learning curves - average reward and cumulative reward
    # you can feed names to plot_rewards or not --> will change the graph labels
    if logger:
        plot_rewards(
            learning_curve, logger.location, names=name, time_step=evaluation_freq
        )
    if name:
        if logger:
            logger.log(
                "{} rewards graphed successfully. See {}".format(name, logger.location)
            )
    else:
        if logger:
            logger.log("Rewards graphed successfully. See {}".format(logger.location))

    return learning_curve


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
    total_timesteps = 5000 #2_000_000
    evaluation_freq = 1000
    n_episodes_to_evaluate = 20

    # starting a logger - results stored in folder labeled w/ date+time
    logger = start_logging()

    # Load the agent and try to load hyperparameters
    agent_module = importlib.import_module(args.group + ".agent")

    # NOTE: change these dependent on what model you want to train & its best hyperparams
    # These are currently the best hyperparameters for: PPO
    params = {'actor_architecture': (64, 64), 
        'actor_activation_function': torch.nn.ReLU, 
        'critic_architecture': (64, 64), 
        'critic_activation_function': torch.nn.ReLU}

    for i in range(5):
        # Create the agent
        agent = agent_module.Agent(env_specs, **params)

        # Load in the pretrained model if one is provided
        if args.load != "None":
            agent.load_weights(os.getcwd(), args.load)
            logger.log(f"Pretrained model loaded: {args.load}")

        logger.log(f"Training start for the following model run {i} ... ")
        logger.log(f"{agent.__dict__}")
        start_time = time()
        # You can feed names to train_agent or not --> will change the saved file names/graph labels
        learning_curve, path_to_best_model = train_agent(
            agent,
            env,
            env_eval,
            total_timesteps,
            evaluation_freq,
            n_episodes_to_evaluate,
            logger,
            name=f"run_{i}",
            visualize=False,
            save_checkpoint=True
        )
        logger.log("Training complete.")

        # Compute some details about the training experiment
        final_mean_reward = learning_curve[-1]
        average_mean_reward = np.mean(learning_curve)
        median_mean_reward = np.median(learning_curve)
        best_mean_reward = np.max(learning_curve)
        cumulative_reward = np.sum(learning_curve)
        auc_mean_reward = auc(range(len(learning_curve)), learning_curve)
        # Log to file
        logger.log(f"\n\nFinal Mean Reward: {round(final_mean_reward, 5)}")
        logger.log(f"\n\Average Mean Reward: {round(average_mean_reward, 5)}")
        logger.log(f"\n\Median Mean Reward: {round(median_mean_reward, 5)}")
        logger.log(f"Best Mean Reward: {round(best_mean_reward, 5)}")
        logger.log(f"Final Cumulative Reward: {round(cumulative_reward, 5)}")
        logger.log(f"AUC for Mean Reward: {round(auc_mean_reward, 5)}")
        elapsed_time = time() - start_time
        logger.log(f"Time Elapsed During Training: {elapsed_time}\n")
        # Log to json
        log_training_experiment_to_json(
            path_to_json=os.path.join(logger.location, "log.json"),
            model_name=f"run_{i}",
            hyperparameters=f"{params}",
            final_mean_reward=final_mean_reward,
            average_mean_reward=average_mean_reward,
            median_mean_reward=median_mean_reward,
            best_mean_reward=best_mean_reward,
            cumulative_reward=cumulative_reward,
            auc_mean_reward=auc_mean_reward,
            path_to_best_model=path_to_best_model,
            list_of_rewards=learning_curve,
        )

    # Plot learning curves - average reward and cumulative reward averages with standard deviation
    plot_best_model_rewards(os.path.join(logger.location, "log.json"), logger.location, time_step=evaluation_freq)
    logger.log("\nRewards graphed successfully. See {}".format(logger.location))
