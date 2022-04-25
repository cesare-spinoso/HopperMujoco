import gym

from environments import JellyBeanEnv, MujocoEnv


def get_environment(env_type):
    """Generates an environment specific to the agent type."""
    if "jellybean" in env_type:
        env = JellyBeanEnv(gym.make("JBW-COMP579-obj-v1"))
    elif "mujoco" in env_type:
        env = MujocoEnv(gym.make("Hopper-v2"))
    elif "ant" in env_type:
        env = MujocoEnv(gym.make("Ant-v2"))
    elif "walker" in env_type:
        env = MujocoEnv(gym.make("Walker2d-v2"))
    else:
        raise Exception(
            "ERROR: Please define your env_type to be either 'jellybean' or 'mujoco'!"
        )
    return env
