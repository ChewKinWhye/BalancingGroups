from train import parse_args, run_experiment
import os
import sys
import json
import time
import torch
import submitit
import argparse
import numpy as np

import models
from datasets import get_loaders

# python main.py --data_path data --output_dir results --num_init_seeds 5
if __name__ == "__main__":
    args = parse_args()
    commands = []
    # Repeat experiments
    # Set other hparams
    hparams_seed = 10
    torch.manual_seed(hparams_seed)
    args["hparams_seed"] = hparams_seed
    args["dataset"] = "waterbirds"
    # args["method"] = "erm"
    args["method"] = "lrr"

    args["num_epochs"] = {
        "waterbirds": 300 + 60,
        "celeba": 50 + 10,
        "multinli": 5 + 2,
        "civilcomments": 5 + 2
    }[args["dataset"]]

    # For DRO
    args["eta"] = 0.1
    args["lr"] = 1e-4
    args["lr_eta"] = 0.1
    args["weight_decay"] = 1e-2
    args["batch_size"] = 32
    # For JTT
    args["up"] = 100
    args["T"] = 60
    for init_seed in range(args["num_init_seeds"]):
        args["init_seed"] = init_seed
        commands.append(dict(args))

    os.makedirs(args["output_dir"], exist_ok=True)
    torch.manual_seed(0)
    commands = [commands[int(p)] for p in torch.randperm(len(commands))]
    print(commands)
    for command in commands:
        run_experiment(command)

