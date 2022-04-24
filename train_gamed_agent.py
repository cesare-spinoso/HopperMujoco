import argparse
from copy import deepcopy
import importlib
import os
from os import listdir
from os.path import isfile, join
from time import time
import json
import numpy as np

from utils.logging_utils import start_logging
from utils.environment import get_environment
from utils.training import train_agent
from utils.metrics import calc_sample_efficiency


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--group", type=str, default="GROUP1", help="group directory")
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

    # Leaderboard sample efficiency = average reward (we don't need to publish these results)
    num_seeds = 1  # Reduce time this takes
    total_timesteps = 100_000
    evaluation_freq = 5_000
    n_episodes_to_evaluate = 10

    ########################################## training a single/multiple agent(s) ##########################################
    # starting a logger - results stored in folder labeled w/ date+time
    logger = start_logging()

    # Load the agent and try to load hyperparameters
    agent_module = importlib.import_module(args.group + ".agent")
    try:
        hyperparameter_module = importlib.import_module(
            args.group + ".gamed_hyperparameters"
        )
        grid = hyperparameter_module.hyperparameter_grid
        logger.log("Loaded the hyperparameter grid")
    except:
        # Use the default hyperparameters
        raise ValueError(
            "You cannot game without gamed_hyperparameters.py in the group folder."
        )


    for i, params in enumerate(grid):
        # Create the agent
        agent = agent_module.Agent(env_specs, **params)

        logger.log(f"(Gamed) training start for the following model {i} ... ")
        logger.log(f"{agent.__dict__}")

        mean_sample_efficiency, mean_time_to_train = calc_sample_efficiency(
            agent_module,
            env_specs,
            params,
            env,
            env_eval,
            total_timesteps,
            evaluation_freq,
            n_episodes_to_evaluate,
            num_seeds,
            logger=logger
        )

        logger.log(
            f"Average sample efficiency for agent {i} with hyperparams {params} is {mean_sample_efficiency}"
        )
        # Modify learning curves such that it's a dictionary
        dict_to_write = {
            "id": i,
            "mean_sample_efficiency": mean_sample_efficiency,
            "mean_time_to_train": mean_time_to_train,
            "hyperparams": f"{params}",
        }
        with open(os.path.join(logger.location, "gamed_evaluation.json"), "a") as f:
            json.dump(dict_to_write, f)
            f.write("\n")


# start_time = time()
# cum_average_reward = 0

# learning_curves_array = np.zeros(
#     (num_seeds, int(total_timesteps / evaluation_freq))
# )
# for seed in range(num_seeds):
#     logger.log(f"Training agent {i} with seed {seed}")
#     # Prevent memory leak
#     agent_copy = deepcopy(agent)
#     learning_curve, path_to_best_model = train_agent(
#         agent_copy,
#         env,
#         env_eval,
#         total_timesteps,
#         evaluation_freq,
#         n_episodes_to_evaluate,
#         logger,
#         save_checkpoint=False,
#     )
#     average_reward = np.array(learning_curve).mean()
#     logger.log(
#         f"Average reward for agent {i} with seed {seed} is {average_reward}"
#     )
#     cum_average_reward += average_reward
#     learning_curves_array[seed, :] = learning_curve
# average_reward = cum_average_reward / num_seeds
# logger.log(f"Average reward for agent {i} is {average_reward}")
# average_rewards.append(average_reward)
# logger.log(f"Saving the mean learning curve for agent {i}")
# learning_curves.append(learning_curves_array.mean(axis=0).tolist())
# logger.log("Training complete.")

# learning_curves = []
# average_rewards = []