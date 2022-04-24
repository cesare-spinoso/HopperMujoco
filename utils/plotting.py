import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch import ge
from json_utils import get_json_data

font = {'family' : 'monospace',
        'size'   : 18}

matplotlib.rc('font', **font)

def plot_rewards(rewards, location, names=None, time_step=None):
    """
    Graphs the average and cumulative reward plots.
    :param rewards: a list, or list of lists, of rewards gained by the model
    :param location: the location in which you want to store the generated .png files
    :param model_names: str of a single model name, or a list of names (if not provided, will use final reward as label)
    """

    if any(isinstance(r, list) for r in rewards):

        # average reward plot
        _ = plt.figure(figsize=(20, 10))
        for i in range(len(rewards)):
            if names != None:
                reward_label = names[i]
            else:
                reward_label = str(round(rewards[i][-1], 2))
            if time_step: # range(start, stop, step)
                plt.plot(range(0, len(rewards[i])*time_step, time_step), rewards[i], label=reward_label)
            else: 
                plt.plot(range(len(rewards[i])), rewards[i], label=reward_label)

        plt.ylabel("Average Reward")
        plt.xlabel("Time Step")
        plt.title("Average Reward Over Time")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        filename = location + "/avg_rewards.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

        # cumulative reward plot
        _ = plt.figure(figsize=(20, 10))
        for i in range(len(rewards)):
            if names != None:
                reward_label = names[i]
            else:
                reward_label = str(round(rewards[i][-1], 2))

            cumulative = np.cumsum(rewards[i])
            if time_step: # range(start, stop, step)
                plt.plot(range(0, len(cumulative)*time_step, time_step), cumulative, label=reward_label)
            else: 
                plt.plot(range(len(cumulative)), cumulative, label=reward_label)

        plt.ylabel("Cumulative Reward")
        plt.xlabel("Time Step")
        plt.title("Cumulative Reward Over Time")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        filename = location + "/cum_rewards.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

    else:  # we have only one reward list, graph only the one

        last_reward = str(round(rewards[-1], 2))
        reward_label = ""

        # average reward plot
        _ = plt.figure()

        if names != None:
            if isinstance(names, list):
                # select the first element as the model name
                reward_label = names[0]
            elif isinstance(names, str):
                # use single name
                reward_label = names
            else:
                reward_label = last_reward
        else:
            reward_label = last_reward

        if time_step: # range(start, stop, step)
            plt.plot(range(0, len(rewards)*time_step, time_step), rewards, label=reward_label)
        else: 
            plt.plot(range(len(rewards)), rewards, label=reward_label)
        plt.ylabel("Average Reward")
        plt.xlabel("Time Step")
        plt.title("Average Reward Over Time")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        filename = location + "/avg_rewards_{}.png".format(last_reward)
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

        # cumulative reward plot
        _ = plt.figure()
        cumulative = np.cumsum(rewards)
        if time_step: # range(start, stop, step)
            plt.plot(range(0, len(cumulative)*time_step, time_step), cumulative, label=reward_label)
        else: 
            plt.plot(range(len(cumulative)), cumulative, label=reward_label)
        plt.ylabel("Cumulative Reward")
        plt.xlabel("Time Step")
        plt.title("Cumulative Reward Over Time")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        filename = location + "/cum_rewards_{}.png".format(last_reward)
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

def plot_best_model_rewards(json_location, save_location, model_names, time_step=None):
    """
    Graphs the average and cumulative reward plots for 5 runs of the best model.
    """
    matplotlib_colors = ['b','r','green','purple','orange','black','pink','yellow']
    # collect rewards from json file
    if type(json_location) == list:
        # average reward plot
        _ = plt.figure(figsize=(20, 10))

        for i in range(len(json_location)): 
            loaded_json = get_json_data(json_location[i])

            rewards, names = [], []
            for m in loaded_json:
                names.append(m['model_name'])
                rewards.append(m['list_of_rewards'])

            mean_rewards = np.mean(rewards, axis=0)
            std_rewards = np.std(rewards, axis=0)

            if time_step: # range(start, stop, step)
                x = range(0, len(mean_rewards)*time_step, time_step)
                plt.plot(x, mean_rewards, color=matplotlib_colors[i], label=model_names[i])
                plt.fill_between(x=x, y1=mean_rewards-std_rewards, y2=mean_rewards+std_rewards, 
                    alpha=0.1, color=matplotlib_colors[i])
            else: 
                x = range(len(mean_rewards))
                plt.plot(x, mean_rewards, color=matplotlib_colors[i], label=model_names[i])
                plt.fill_between(x=x, y1=mean_rewards-std_rewards, y2=mean_rewards+std_rewards, 
                    alpha=0.1, color=matplotlib_colors[i])

        plt.ylabel("Reward")
        plt.xlabel("Time Step")
        plt.title("Average Reward Over Time")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        filename = save_location + "/avg_rewards_best_model.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

        # cumulative reward plot
        _ = plt.figure(figsize=(20, 10))
        for j in range(len(json_location)): 
            loaded_json = get_json_data(json_location[j])

            rewards, names = [], []
            for m in loaded_json:
                names.append(m['model_name'])
                rewards.append(m['list_of_rewards'])
            cumulative_rewards = []
            for i in range(len(rewards)): cumulative_rewards.append(np.cumsum(rewards[i]))
            
            mean_cumulative_rewards = np.mean(cumulative_rewards, axis=0)
            std_cumulative_rewards = np.std(cumulative_rewards, axis=0)

            if time_step: # range(start, stop, step)
                x = range(0, len(mean_cumulative_rewards)*time_step, time_step)
                plt.plot(x, mean_cumulative_rewards, color=matplotlib_colors[j], label=model_names[j])
                plt.fill_between(x=x, y1=mean_cumulative_rewards-std_cumulative_rewards, 
                    y2=mean_cumulative_rewards+std_cumulative_rewards, alpha=0.1, color=matplotlib_colors[j])
            else: 
                x = range(len(mean_cumulative_rewards))
                plt.plot(x, mean_cumulative_rewards, color=matplotlib_colors[j], label=model_names[j])
                plt.fill_between(x=x, y1=mean_cumulative_rewards-std_cumulative_rewards, 
                    y2=mean_cumulative_rewards+std_cumulative_rewards, alpha=0.1, color=matplotlib_colors[j])

        plt.ylabel("Cumulative Reward")
        plt.xlabel("Time Step")
        plt.title("Cumulative Reward Over Time")
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        filename = save_location + "/cum_rewards_best_model.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

    else: 
        loaded_json = get_json_data(json_location)

        rewards, names = [], []
        for m in loaded_json:
            names.append(m['model_name'])
            rewards.append(m['list_of_rewards'])

        # average reward plot
        _ = plt.figure(figsize=(20, 10))
        mean_rewards = np.mean(rewards, axis=0)
        std_rewards = np.std(rewards, axis=0)

        if time_step: # range(start, stop, step)
            x = range(0, len(mean_rewards)*time_step, time_step)
            plt.plot(x, mean_rewards, color='b', label=model_names[0])
            plt.fill_between(x=x, y1=mean_rewards-std_rewards, y2=mean_rewards+std_rewards, alpha=0.3, color='b')
        else: 
            x = range(len(mean_rewards))
            plt.plot(x, mean_rewards, color='b', label=model_names[0])
            plt.fill_between(x=x, y1=mean_rewards-std_rewards, y2=mean_rewards+std_rewards, alpha=0.3, color='b')

        plt.ylabel("Average Reward")
        plt.xlabel("Time Step")
        plt.title("Average Reward Over Time")
        filename = save_location + "/avg_rewards_best_model.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

        # cumulative reward plot
        _ = plt.figure(figsize=(20, 10))
        cumulative_rewards = []
        for i in range(len(rewards)): cumulative_rewards.append(np.cumsum(rewards[i]))
        
        mean_cumulative_rewards = np.mean(cumulative_rewards, axis=0)
        std_cumulative_rewards = np.std(cumulative_rewards, axis=0)

        if time_step: # range(start, stop, step)
            x = range(0, len(mean_cumulative_rewards)*time_step, time_step)
            plt.plot(x, mean_cumulative_rewards, color='b', label=model_names[0])
            plt.fill_between(x=x, y1=mean_cumulative_rewards-std_cumulative_rewards, 
                y2=mean_cumulative_rewards+std_cumulative_rewards, alpha=0.3, color='b')
        else: 
            x = range(len(mean_cumulative_rewards))
            plt.plot(x, mean_cumulative_rewards, color='b', label=model_names[0])
            plt.fill_between(x=x, y1=mean_cumulative_rewards-std_cumulative_rewards, 
                y2=mean_cumulative_rewards+std_cumulative_rewards, alpha=0.3, color='b')

        plt.ylabel("Cumulative Reward")
        plt.xlabel("Time Step")
        plt.title("Cumulative Reward Over Time")
        filename = save_location + "/cum_rewards_best_model.png"
        plt.savefig(filename, bbox_inches="tight")
        plt.close()



if __name__ == '__main__':
    # loaded_json = get_json_data("openai_sac_agent/results/log.json")

    # reward_lists, names = [], []
    # for m in loaded_json:
    #     names.append(m['model_name'])
    #     reward_lists.append(m['list_of_rewards'])

    # save_location = "openai_sac_agent/results/"

    # plot_rewards(reward_lists, save_location, names)

    # json_list = ["ppo_agent/results/best_model/log_best_model.json", "vpg_agent/results/best_model/log_best_model.json",
    #     "ddpg_agent/results/best_model/log_best_model.json", "td3_agent/results/best_model/log_best_model.json"]
    # plot_best_model_rewards(json_list, save_location=".", model_names=['ppo', 'vpg', 'ddpg', 'td3'], 
    #     time_step=1000) # i.e. eval freq
    json_list = ["vpg_agent/results/best_model/log_best_model.json", "ppo_agent/results/best_model/log_best_model.json",
        "ddpg_agent/results/best_model/log_best_model.json", "td3_agent/results/best_model/log_best_model.json",
        "openai_sac_agent/results/best_model/log_best_model.json"]
    plot_best_model_rewards(json_list, save_location=".", model_names=['VPG', 'PPO', 'DDPG', 'TD3', 'SAC'], 
        time_step=1000) # i.e. eval freq
    # if only one name, still input it in a list (eg. ['ppo'])