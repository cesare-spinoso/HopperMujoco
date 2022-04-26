import numpy as np
from utils.json_utils import get_json_data

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

def find_mean_metrics(json_list, model_names):
    for i in range(len(json_list)): 
        loaded_json = get_json_data(json_list[i])

        final_mean_reward, average_mean_reward, median_mean_reward = [], [], []
        best_mean_reward, cumulative_reward, auc_mean_reward = [], [], []
        for m in loaded_json:
            final_mean_reward.append(m['final_mean_reward'])
            average_mean_reward.append(m['average_mean_reward'])
            median_mean_reward.append(m['median_mean_reward'])
            best_mean_reward.append(m['best_mean_reward'])
            cumulative_reward.append(m['cumulative_reward'])
            auc_mean_reward.append(m['auc_mean_reward'])

        print("-"*50)
        print("MODEL:", model_names[i])
        print("final_mean_reward:", np.mean(final_mean_reward))
        print("average_mean_reward:", np.mean(average_mean_reward))
        print("median_mean_reward:", np.mean(median_mean_reward))
        print("best_mean_reward:", max(best_mean_reward)) #!!!!!!!!!!!!
        print("cumulative_reward:", np.mean(cumulative_reward))
        print("auc_mean_reward:", np.mean(auc_mean_reward))
        print("-"*50)


if __name__ == '__main__':
    json_list = ["ppo_agent/results/best_model/log_best_model.json", "vpg_agent/results/best_model/log_best_model.json",
        "ddpg_agent/results/best_model/log_best_model.json", "td3_agent/results/best_model/log_best_model.json",
        "openai_sac_agent/results/best_model/log_best_model.json"]
    find_mean_metrics(json_list, model_names=['PPO', 'VPG', 'DDPG', 'TD3', 'SAC'])
