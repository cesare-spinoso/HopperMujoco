import numpy as np


def evaluate_agent(agent, env, n_episodes_to_evaluate):
    """Evaluates the agent for a provided number of episodes."""
    array_of_acc_rewards = []
    ep_lens = []
    for _ in range(n_episodes_to_evaluate):
        acc_reward = 0
        done = False
        curr_obs = env.reset()
        ep_len = 0
        while not done:
            action = agent.act(curr_obs, mode="eval")
            next_obs, reward, done, _ = env.step(action)
            acc_reward += reward
            curr_obs = next_obs
            ep_len += 1
        ep_lens.append(ep_len)
        array_of_acc_rewards.append(acc_reward)
    print(f"mean episode len during eval: {np.mean(np.array(ep_lens))}")
    return np.mean(np.array(array_of_acc_rewards))
