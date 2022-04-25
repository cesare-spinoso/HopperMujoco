import os
from pathlib import Path

files = os.listdir("/home/c_spino/comp_597/GROUP_013/openai_sac_agent/results/extras/from_using_cosing_lr/")

for file in files:
    print(file)
    print(Path(Path(file).stem).stem)
    os.system('python evaluate_agent.py --group openai_sac_agent --model_path /home/c_spino/comp_597/GROUP_013/openai_sac_agent/results/extras/from_using_cosing_lr/' + Path(Path(file).stem).stem)

print("*"*100)
print("*"*100)
print("*"*100)

files = os.listdir("/home/c_spino/comp_597/GROUP_013/openai_sac_agent/results/extras/from_varying_alpha_from_ckpt/")

for file in files:
    print(file)
    print(Path(Path(file).stem).stem)
    os.system('python evaluate_agent.py --group openai_sac_agent --model_path /home/c_spino/comp_597/GROUP_013/openai_sac_agent/results/extras/from_varying_alpha_from_ckpt/' + Path(Path(file).stem).stem)