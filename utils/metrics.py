import time
import numpy as np

from sklearn.metrics import auc
from copy import deepcopy
from .json_utils import get_json_data
from .training import train_agent

def calc_sample_efficiency(
    agent_untrained,
    env,
    env_eval,
    total_timesteps,
    evaluation_freq,
    n_episodes_to_evaluate,
    num_seeds,
    logger,
    save_checkpoint=False,
):
    """
    Calculates the sample efficiency (as only AUC of rewards) by training len(num_seeds) new models.
    """
    auc_performances = []
    start_time = time.time()
    for i in range(num_seeds):
        print(f"starting training on {i+1} of {num_seeds}...")

        # train for 100K steps, evaluating every 1000
        agent_to_train = deepcopy(agent_untrained)
        seed_performance, _ = train_agent(
            agent_to_train,
            env,
            env_eval,
            total_timesteps,
            evaluation_freq,
            n_episodes_to_evaluate,
            logger,
            seed=i,
            save_checkpoint=save_checkpoint
        )
        auc_performances.append(auc(range(len(seed_performance)), seed_performance))
    end_time = time.time()

    mean_sample_efficiency = np.mean(np.array(auc_performances))
    mean_time_to_train = (end_time - start_time) / num_seeds

    return mean_sample_efficiency, mean_time_to_train

def calc_sample_efficiency_from_json(json_list, model_names):
    """
    Calculates the sample efficiency from a json file (as described by Nishanth (TA)).
    """
    for i in range(len(json_list)): 
        loaded_json = get_json_data(json_list[i])

        sample_efficiencies =[]

        for m in loaded_json:
            # collect every 5k eval for 100_000 steps
            rewards = m['list_of_rewards'][0:100] # i.e. first 100_000 timesteps if we measure every 1000
            rewards_5k = [rewards[j] for j in range(len(rewards)) if j % 5 == 0]
            # calc AUC
            auc_run = auc(range(0, 5000*len(rewards_5k), 5000), rewards_5k)
            # divide by 100_000
            sample_efficiencies.append(auc_run/100_000)
        
        print("SAMPLE EFFICIENCY FOR", model_names[i])
        print(sample_efficiencies)
        print(np.mean(sample_efficiencies))
        print()

if __name__ == '__main__':
    json_list = ["ppo_agent/results/best_model/log_best_model.json", "vpg_agent/results/best_model/log_best_model.json",
        "ddpg_agent/results/best_model/log_best_model.json", "td3_agent/results/best_model/log_best_model.json",
        "openai_sac_agent/results/best_model/log_best_model.json"]
    calc_sample_efficiency_from_json(json_list, model_names=['PPO', 'VPG', 'DDPG', 'TD3', 'SAC'])