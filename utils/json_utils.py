import os
import json
from typing import List
import numpy as np
from sklearn.metrics import auc

def log_training_experiment_to_json(
    path_to_json: str,
    model_name: str,
    hyperparameters: dict,
    final_mean_reward: float,
    average_mean_reward: float,
    median_mean_reward: float,
    best_mean_reward: float,
    cumulative_reward: float,
    auc_mean_reward: float,
    path_to_best_model: str,
    list_of_rewards: List[float],
):
    """Save training information to json file"""
    dict_to_write = {
        "model_name": model_name,
        "hyperparameters": hyperparameters,
        "final_mean_reward": final_mean_reward,
        "average_mean_reward": average_mean_reward,
        "median_mean_reward": median_mean_reward,
        "best_mean_reward": best_mean_reward,
        "cumulative_reward": cumulative_reward,
        "auc_mean_reward": auc_mean_reward,
        "path_to_best_model": path_to_best_model,
        "list_of_rewards": list_of_rewards,
    }
    if not os.path.exists(path_to_json):
        with open(path_to_json, "w") as f:
            json.dump(dict_to_write, f)
            f.write("\n")
    else:
        with open(path_to_json, "a") as f:
            json.dump(dict_to_write, f)
            f.write("\n")

def get_json_data(path_to_json: str):
    """Load information from json file"""
    json_data = []
    with open(path_to_json, "r") as f:
        for line in f:
            json_data.append(json.loads(line))
    return json_data

def clip_to_n_train_iterations_json(path_to_json_old: str, path_to_json_new: str, n: int):
    """
    Load information from json file, keep only n training steps, and save to a new file.
    """
    json_data = []
    with open(path_to_json_old, "r") as f:
        for line in f:
            json_data.append(json.loads(line))

    for run in json_data:
        print("old list length:", len(run['list_of_rewards']))
        new_rewards_list = run['list_of_rewards'][0:n]

        final_mean_reward = new_rewards_list[-1]
        average_mean_reward = np.mean(new_rewards_list)
        median_mean_reward = np.median(new_rewards_list)
        best_mean_reward = np.max(new_rewards_list)
        cumulative_reward = np.sum(new_rewards_list)
        auc_mean_reward = auc(range(len(new_rewards_list)), new_rewards_list)

        log_training_experiment_to_json(path_to_json_new, run['model_name'], run['hyperparameters'],
            final_mean_reward, average_mean_reward, median_mean_reward, best_mean_reward, cumulative_reward,
            auc_mean_reward, "", new_rewards_list)

if __name__ == '__main__':
    sac_json_old = "results/sac_variants/bottleneck_varying_alpha_sac.json"
    sac_json_new = "results/sac_variants/bottleneck_varying_alpha_sac_clipped.json"
    n_iterations = int(2_000_000/1000)

    clip_to_n_train_iterations_json(sac_json_old, sac_json_new, n_iterations)