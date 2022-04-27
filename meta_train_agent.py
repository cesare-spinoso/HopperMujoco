"""
A script to train a single agent OR perform hyperparameter tuning.
"""

import argparse
import importlib
import os
from os import listdir
from os.path import isfile, join
from time import time
from pathlib import Path

import numpy as np
from sklearn.metrics import auc

from utils.json_utils import log_training_experiment_to_json
from utils.evaluation import evaluate_agent
from utils.logging_utils import start_logging
from utils.environment import get_environment
from utils.training import train_agent

if __name__ == "__main__":
    """Example:
    python meta_train_agent.py --group openai_sac_agent 
    --root_path /home/c_spino/comp_597/erasec/GROUP_013/openai_sac_agent/results 
    --pretrained_model_name sac_ckpt_m_0_3698.841
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--group", type=str, default="GROUP1", help="group directory")
    parser.add_argument(
        "--root_path",
        type=str,
        default="None",
        help="need folder/filename in results folder (without .pth.tar)",
        required=True,
    )
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default="None",
        help="need folder/filename in results folder (without .pth.tar)",
        required=True,
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

    n_episodes_to_evaluate_leaderboard = 50
    num_seeds = 5

    ########################################## hyperparameter tuning ##########################################

    # Load the agent and try to load hyperparameters
    agent_module = importlib.import_module(args.group + ".agent")
    try:
        # A path must be provided, the assumption is that the hyperparameters of the agent are in best_hyperparameters.py
        hyperparameter_module = importlib.import_module(
            args.group + ".best_hyperparameters"
        )
        params = hyperparameter_module.params
        assert params["update_start_in_timesteps"] is not None
    except:
        raise ValueError(
            "Could not load best_hyperparameters.py, this is required to do meta learning."
        )

    # Path to best agent
    path_to_best_model = os.path.join(
        args.root_path, args.pretrained_model_name + ".pth.tar"
    )
    seed = 0 # also change the training randomness from one iteration to the other

    while True:
        # starting a logger - results stored in folder labeled w/ date+time
        # placed inside loop so that create new folder everytime you restart
        logger = start_logging()

        # Create the agent
        agent = agent_module.Agent(env_specs, **params)

        # Load the best agent
        agent.load_weights(
            root_path=Path(path_to_best_model).parent,
            pretrained_model_name=Path(Path(path_to_best_model).stem).stem,
        )
        # Evaluate the current agent
        current_agent_performance = evaluate_agent(
            agent, env_eval, n_episodes_to_evaluate_leaderboard, num_seeds
        )

        logger.log(
            f"Training start for the model with path {path_to_best_model} which currently has performance {current_agent_performance:.3f}"
        )
        logger.log(f"{agent.__dict__}")
        start_time = time()
        # You can feed names to train_agent or not --> will change the saved file names/graph labels
        learning_curve, new_path_to_best_model = train_agent(
            agent,
            env,
            env_eval,
            total_timesteps,
            evaluation_freq,
            n_episodes_to_evaluate,
            logger,
            name=f"m_0",
            visualize=False,
            seed=seed,
            # Don't resave the model!
            save_checkpoint_start_timestep=agent.update_start_in_timesteps + 1,
        )
        logger.log(
            f"Training complete. The best agent that was found is {new_path_to_best_model}"
        )

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
            model_name=f"m_0",
            hyperparameters=f"{params}",
            final_mean_reward=final_mean_reward,
            average_mean_reward=average_mean_reward,
            median_mean_reward=median_mean_reward,
            best_mean_reward=best_mean_reward,
            cumulative_reward=cumulative_reward,
            auc_mean_reward=auc_mean_reward,
            path_to_best_model=new_path_to_best_model,
            list_of_rewards=learning_curve,
        )

        # Evaluate the agent as it would be on the leaderboard
        logger.log("Evaluating the new 'best agent' via the leaderboard metric")
        new_best_agent = agent_module.Agent(env_specs, **params)
        new_best_agent.load_weights(
            root_path=Path(new_path_to_best_model).parent,
            pretrained_model_name=Path(Path(new_path_to_best_model).stem).stem,
        )
        new_best_performance = evaluate_agent(
            new_best_agent, env_eval, n_episodes_to_evaluate_leaderboard, num_seeds
        )
        seed += 1
        if new_best_performance > current_agent_performance:
            logger.log(f"The new agent ({new_best_performance:.3f}) is indeed better than the old one ({current_agent_performance:.3f})")
            logger.log("Setting the best path to the new model")
            path_to_best_model = new_path_to_best_model
        else:
            logger.log(f"The new agent ({new_best_performance:.3f}) is not better than the old one ({current_agent_performance:.3f})")
            logger.log("Keeping the old model, decreasing the initial amount of exploration by 10%")
            params["alpha"] = params["alpha"] * 0.9
