import argparse
import importlib
import os
import random
from os import listdir
from os.path import isfile, join
from time import time

import gym
import numpy as np
import torch
from sklearn.metrics import auc

from environments import JellyBeanEnv, MujocoEnv

from utils.json import log_training_experiment_to_json
from utils.plotting import plot_rewards
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


def calc_sample_efficiency(agent_module, env_specs, env, env_eval):
    start_time = time()

    eval_performances = []
    total_timesteps = 100000
    evaluation_freq = 1000
    n_episodes_to_evaluate = 20
    save_checkpoints = False

    for seed in range(5):
        print(f"starting training on {seed+1} of 5...")

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
        seed_performance = train_agent(
            agent,
            env,
            env_eval,
            total_timesteps,
            evaluation_freq,
            n_episodes_to_evaluate,
            save_checkpoints,
            logger=None,
        )
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


def train_agent(
    agent,
    env,
    env_eval,
    total_timesteps,
    evaluation_freq,
    n_episodes_to_evaluate,
    logger,
    name=None,
    visualize=False,
):
    """Train the agent and return the average rewards and the path to the best model."""
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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

    return array_of_mean_acc_rewards, save_path


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
    total_timesteps = 2_000_000
    evaluation_freq = 1000
    n_episodes_to_evaluate = 20

    ########################################## training a single/multiple agent(s) ##########################################
    # starting a logger - results stored in folder labeled w/ date+time
    logger = start_logging()

    # Load the agent and try to load hyperparameters
    agent_module = importlib.import_module(args.group + ".agent")
    try:
        hyperparameter_module = importlib.import_module(args.group + ".hyperparameters")
        grid = hyperparameter_module.hyperparameter_grid
        logger.log("Loaded the hyperparameter grid")
    except:
        # Use the default hyperparameters
        logger.log("Could not find the hyperparameters.py module, using default agent.")
        grid = [{}]

    for i, params in enumerate(grid):
        # Create the agent
        agent = agent_module.Agent(env_specs, **params)

        # Load in the pretrained model if one is provided
        if args.load != "None":
            agent.load_weights(os.getcwd(), args.load)
            logger.log(f"Pretrained model loaded: {args.load}")

        logger.log(f"Training start for the following model {i} ... ")
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
            name=f"m_{i}",
            visualize=False,
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
            model_name=f"m_{i}",
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

        # Plot learning curves - average reward and cumulative reward
        # You can feed names to plot_rewards or not --> will change the graph labels
        plot_rewards(learning_curve, logger.location, names=f"m_{i}", time_step=evaluation_freq)
        logger.log("\nRewards graphed successfully. See {}".format(logger.location))
