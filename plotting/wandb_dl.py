# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import pickle as pkl
import pandas as pd 
import matplotlib.pyplot as plt
from collections import defaultdict
import wandb
import numpy as np

from matplotlib import rc
import matplotlib.ticker as mticker
import seaborn as sns

import wandb
import tqdm
import copy
import argparse

import os

api = wandb.Api()

def find_next_hp(config_str):
    index = config_str.find("\n")
    if index == -1:
        return ""
    if config_str[index:index + 2] == "\n-":
        return find_next_hp(config_str[index + 2:])
    return config_str[index + 1:]

def parse(config_str):
    config = {}
    while True:
        config_str_nxt = find_next_hp(config_str)
        
        curr_seg = config_str[:len(config_str)-len(config_str_nxt)]
        k, v = curr_seg.split(": ")
        config[k] = v[:-1]
        
        if config_str_nxt == "":
            break
        config_str = config_str_nxt
    return config

def post_process(config):
    config = copy.deepcopy(config)
    items = list(config.items())
    for k, v in items:
        if type(v) == str and v.find("\n") != -1:
            config.pop(k)
            for kk, vv in parse(v).items():
                ak = k + "." + kk
                config[ak] = vv
    
    return config


def save(path, data):
    with open(path, 'wb') as f:
        pkl.dump(data, f)
        
def load(path):
    with open(path, 'rb') as f:
        return pkl.load(f)

class Data:
    def __init__(self, proj_names, keys, entity):
        self._runs = []
        self.keys = keys
        self.entity = entity
        
        for proj_name in proj_names:
            self._collect(proj_name)
        
    def _collect(self, proj_name):
        runs = api.runs(f"{self.entity}/{proj_name}")
        for run in tqdm.tqdm(runs):
            config = post_process(run.config)
            config["project_name"] = proj_name
            entry = {"config": config}
            for key in self.keys:
                df = run.history(keys=[key])
                if len(df.columns) == 0: continue
                
                steps = df[df.columns[0]].to_numpy()
                values = df[key].to_numpy()
                entry[key] = (steps, values)
            self._runs.append(entry)
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog='Wandb experiments downloader',
        description='Download experiment data from weights and biases server'
    )

    parser.add_argument('--entity', type=str)
    parser.add_argument('--project_name', type=str) 
    parser.add_argument('--domain', type=str, help="one of [antmaze, adroit, cog]") 
    args = parser.parse_args()

    assert args.domain in ["antmaze", "adroit", "cog"]
    keys = ["env_step", "evaluation/return"]
    if args.domain == "antmaze":
        keys.append("coverage")
    data = Data([args.project_name], keys=keys, entity=args.entity)
    os.makedirs('data', exist_ok=True)
    with open(f'data/{args.domain}.pkl', 'wb') as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)
