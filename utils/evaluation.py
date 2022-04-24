import numpy as np


def evaluate_agent(agent, env, num_seeds, n_episodes_to_evaluate):
    """Evaluates the agent for a provided number of episodes."""
    rewards_per_seed = []
    for seed in range(num_seeds):
        array_of_acc_rewards = []
        if num_seeds > 1:
            env.seed(seed)
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
        rewards_per_seed.append(np.mean(np.array(array_of_acc_rewards)))
    return np.mean(np.array(rewards_per_seed))
