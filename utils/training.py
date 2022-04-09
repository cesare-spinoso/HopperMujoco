import random
import numpy as np
import torch

from .evaluation import evaluate_agent


def train_agent(
    agent,
    env,
    env_eval,
    total_timesteps,
    evaluation_freq,
    n_episodes_to_evaluate,
    logger,
    seed=0,
    name=None,
    visualize=False,
):
    """Train the agent and return the average rewards and the path to the best model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.seed(seed)
    env_eval.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    timestep = 0
    array_of_mean_acc_rewards = []
    current_mean_acc_rewards = 0

    while timestep < total_timesteps:

        done = False
        curr_obs = env.reset()

        while not done:
            if visualize:
                env.render()

            action = agent.act(curr_obs, mode="train")
            next_obs, reward, done, _ = env.step(action)
            agent.update(curr_obs, action, reward, next_obs, done, timestep)
            curr_obs = next_obs
            timestep += 1
            if timestep % evaluation_freq == 0:
                mean_acc_rewards = evaluate_agent(
                    agent, env_eval, n_episodes_to_evaluate
                )
                logger.log(
                    "timestep: {ts}, acc_reward: {acr:.2f}".format(
                        ts=timestep, acr=mean_acc_rewards
                    )
                )
                array_of_mean_acc_rewards.append(mean_acc_rewards)

                # if we have improvement, save a checkpoint
                if mean_acc_rewards > current_mean_acc_rewards:
                    if name != None:
                        save_path = agent.save_checkpoint(
                            mean_acc_rewards, logger.location, name
                        )
                    else:
                        save_path = agent.save_checkpoint(
                            mean_acc_rewards, logger.location
                        )
                    logger.log("checkpoint saved: {}".format(save_path))
                    current_mean_acc_rewards = mean_acc_rewards

    return array_of_mean_acc_rewards, save_path
