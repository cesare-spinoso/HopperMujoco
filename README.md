# COMP 579 - Reinforcement Learning Project
## Virtual Environment Installation
Create a virtual environment with the dependencies found in `environment.yml`.

```
git clone https://github.com/COMP579TA/COMP579-Project-Template
cd COMP579-Project-Template
conda env create environment.yml -n my-venv
conda activate my-venv
```

Note that this environment contains a cudatoolkit of version 10.2.89. Thus, you may have to uninstall and then reinstall it depending on your cuda version in the case that you are using GPUs.

## Mujoco Domain

MuJoCo is a general purpose physics engine that aims to facilitate research and development in robotics. It stands for Multi-Joint dynamics with contact. Mujoco has different environments from which we use [Hopper](https://gym.openai.com/envs/Hopper-v2/).
Hopper has a  11-dimensional state space, that is position and velocity of each joint. The initial states are uniformly randomized. The action is a 3-dimensional continuous space. This environment is terminated when the agent falls down.


### Mujoco Installation
We'll be using mujoco210 in this project. This page contains the mujoco210 releases:
https://github.com/deepmind/mujoco/releases/tag/2.1.0
Download the distribution compatible to your OS and extract the downloaded ```mujoco210``` directory into ```~/.mujoco/```.

#### GPU Machines
After activating the virtual environment, change the environment variable as:
```
conda env config vars set LD_LIBRARY_PATH=/usr/local/pkgs/cuda/latest/lib64:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia-460:/usr/lib/nvidia
```

Note that the GPU driver `nvidia-460` is only applicable for machines with GPUs and is machine specific.

#### CPU Machines
For installing mujoco on a CPU only machine do as follows:
1. Create the conda environment using ```conda create --name mujoco --file environment.yml```.

2. Set the conda environment variable to: ```LD_LIBRARY_PATH=/usr/local/pkgs/cuda/latest/lib64:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia```.

3. You can change the conda environment variable using ```conda env config vars set LD_LIBRARY_PATH=...```. If this command doesn't work in your setting, you can follow this [solution](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#macos-and-linux).


Install the required packages to run mujoco environment
```
!apt-get install -y \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common

!apt-get install -y patchelf
```
```
!pip3 install gym
!pip install free-mujoco-py
```
You can check the following notebook for further info: https://colab.research.google.com/drive/1Zmr96imxqHnbauXf8KbzKgwuF7TROc9F?usp=sharing

Once you are done with training the agent, please copy-paste the code and manually create an ```agent.py``` file and ```env_info.txt``` and upload it to your group folder.

## Training the agent
Your agent can be trained by running `python3 train_agent.py --group GROUP_XXX`. For example, for the group `GROUP_MJ1`, it can be trained as `python3 train_agent.py --group GROUP_MJ1`. You can also run other variants of training such as training from the best agent, training for sample efficiency and evaluating the agent.

## Paper
You can find our paper with results [here](RL_paper.pdf).

## Video
You can find a video highlighting our work [here](https://drive.google.com/file/d/178XOeBQlrf5vF3lTS4hY0BXyyBcEcKFM/view?usp=sharing).
