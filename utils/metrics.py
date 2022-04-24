import time
import numpy as np

from sklearn.metrics import auc
from copy import deepcopy

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
        auc_performances.append(auc(range(0, total_timesteps, evaluation_freq), seed_performance))
    end_time = time.time()

    mean_sample_efficiency = np.mean(np.array(auc_performances))/total_timesteps
    mean_time_to_train = (end_time - start_time) / num_seeds

    return mean_sample_efficiency, mean_time_to_train