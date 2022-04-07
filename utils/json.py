import os
import json
from typing import List


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