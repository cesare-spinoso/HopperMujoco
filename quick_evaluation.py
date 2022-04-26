from fileinput import filelineno
import os
from pathlib import Path

files = os.listdir("/home/c_spino/comp_597/GROUP_013/openai_sac_agent/results/extras")

for file in files:
    print(file)
    print(Path(Path(file).stem).stem)
    os.system('python evaluate_agent.py --group openai_sac_agent --model_path /home/c_spino/comp_597/GROUP_013/openai_sac_agent/results/extras/' + Path(Path(file).stem).stem)