import matplotlib.pyplot as plt
import numpy as np
from json_utils import get_json_data

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
        filename = location + "/avg_rewards_vpg.png"
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
        filename = location + "/cum_rewards_vpg.png"
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
        filename = location + "/avg_rewards_vpg_{}.png".format(last_reward)
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
        filename = location + "/cum_rewards_vpg_{}.png".format(last_reward)
        plt.savefig(filename, bbox_inches="tight")
        plt.close()


if __name__ == '__main__':
    loaded_json = get_json_data("results/2022-04-09_22h10m40/log.json")

    reward_lists, names = [], []
    for m in loaded_json:
        names.append(m['model_name'])
        reward_lists.append(m['list_of_rewards'])

    save_location = "results/2022-04-09_22h10m40"

    plot_rewards(reward_lists, save_location, names)