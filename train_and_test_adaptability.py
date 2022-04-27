import argparse
import importlib
import os
from os import listdir
from os.path import isfile, join
from time import time

import numpy as np
from sklearn.metrics import auc


from utils.json_utils import log_training_experiment_to_json
from utils.plotting import plot_rewards
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
        "--gravity",
        type=float,
        default=1.0,
        help="gravity multiplier, only works with env type mujoco"
    )
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
        env.model.opt.gravity[-1] = env.model.opt.gravity[-1] * args.gravity
        env_eval.model.opt.gravity[-1] = env_eval.model.opt.gravity[-1] * args.gravity
        env_specs = {
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        }

    # Training and evaluation variables
    total_timesteps = 20_000
    evaluation_freq = 10000
    n_episodes_to_evaluate = 20
    sample_efficiency_num_seeds = 5
    sample_efficiency_total_timesteps = 100_000
    sample_efficiency_evaluation_freq = 5000

    ########################################## training a single/multiple agent(s) ##########################################
    # starting a logger - results stored in folder labeled w/ date+time
    logger = start_logging()

    # Load the agent and try to load hyperparameters
    agent_module = importlib.import_module(args.group + ".agent")
    try:
        # If no .pth file is provided try to do grid search otherwise run default
        if args.load == "None":
            hyperparameter_module = importlib.import_module(args.group + ".hyperparameters")
            grid = hyperparameter_module.hyperparameter_grid
            logger.log("Loaded the hyperparameter grid")
        # If a path is provided, the assumption is that the hyperparameters of the agent are in best_hyperparameters.py
        else:
            hyperparameter_module = importlib.import_module(args.group + ".best_hyperparameters")
            grid = [hyperparameter_module.params]
            logger.log("Loaded the best hyperparameters")
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
        
        for j in range(sample_efficiency_num_seeds):
            # Create the agent
            agent = agent_module.Agent(env_specs, **params)

            # Load in the pretrained model if one is provided
            if args.load != "None":
                agent.load_weights(os.getcwd(), args.load)
                logger.log(f"Pretrained model loaded: {args.load}")

            logger.log(f"Training start for the following model {i}_{j}  ... ")
            logger.log(f"{agent.__dict__}")
            start_time = time()
            # You can feed names to train_agent or not --> will change the saved file names/graph labels
            learning_curve, path_to_best_model = train_agent(
                agent,
                env,
                env_eval,
                sample_efficiency_total_timesteps,
                sample_efficiency_evaluation_freq,
                n_episodes_to_evaluate,
                logger,
                name=f"m_{i}",
                visualize=False,
                seed=j,
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
                model_name=f"m_{i}_{j}",
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
