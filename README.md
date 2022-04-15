# COMP 579 - GROUP_013
## Training
Copy the agent module, environment info and weights (with the name `submitted_model.pth.tar` **NOTE: MAKE SURE THE AGENT MODULE LOADS WITH THIS DEFAULT NAME AND THAT IT'S HYPERPARAMETERS MATCH THE SUBMITTED MODEL!!!) to the root directory. To make sure that things will run, also copy these files to `submitted_agent/` folder. You can also add stuff to `agent_info.txt` to keep track of things.

Once you're done testing the module in the `submitted_agent/` folder move it to the root directory (so that we don't create two copies).

Test training with 
```
python train_agent --group submitted_agent
```

Test evaluation with
```
python evaluate --group submitted_agent --root_path /absolute/path/GROUP_013/submitted_agent
```