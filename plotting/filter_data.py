import pickle as pkl
from wandb_dl import Data
from itertools import chain

import numpy as np

def config_match(conf_src, conf_tgt):
    for k, v in conf_tgt.items():
        if v == "null":
            if k not in conf_src: 
                print("skipping", k)
                continue
        
        if k not in conf_src: return False
        if type(v) == tuple:
            min_v, max_v = v
            if conf_src[k] >= max_v: return False
            if conf_src[k] <= min_v: return False
        else:
            if conf_src[k] != v: return False
    return True


def normalize_performance(x, env_name):
    if env_name in ["door-binary-v0", "relocate-binary-v0"]:
        return (x + 200.) / 200.
    if env_name in ["pen-binary-v0"]:
        return (x + 100.) / 100.
    if env_name in ["Widow250PickTray-v0"]:
        return x / 40.
    if env_name in ["Widow250DoubleDrawerOpenGraspNeutral-v0"]:
        return x / 50.
    if env_name in ["Widow250DoubleDrawerCloseOpenGraspNeutral-v0"]:
        return x / 80.
    return x

def filter(data, conf):
    matched_runs = []
    indices = []

    if type(data) == list:
        iters = chain([d._runs for d in data])
    else:
        iters = data._runs
    for index, run in enumerate(iters):
        if config_match(run["config"], conf):
            matched_runs.append(run)
            indices.append(index)
    return matched_runs, indices

def interp(x1, x2, y1, y2, x):
    dy = y2 - y1
    dx = x2 - x1
    return (x - x1) / dx * dy + y1

def get_runs_with_metrics(data, conf, step_metric, value_metric, anc_steps, min_length_v=30, num_seeds=20, patch_name="arxiv-patch", extra_args="", is_pixel=False):
    metric_list = []
    runs, indices = filter(data, conf)
    min_length = None
    
    ids = []
    seeds = set()
    for run, idt in zip(runs, indices):

        if value_metric not in run: continue
        if step_metric not in run: continue
        
        steps_rs, steps = run[step_metric]
        values_rs, values = run[value_metric]
    
        env_steps = []
        step_index = 0
        failed = False
        for value_r, value in zip(values_rs, values):
            step_r = steps_rs[step_index]
            while step_r < value_r and step_index < len(steps_rs) - 1:
                step_index += 1
                step_r = steps_rs[step_index]
            if step_r < value_r:
                values = values[:len(env_steps)]
                break
            env_steps.append(steps[step_index])

        if failed: continue
        adjusted_values = []
        step_index = 0
        for anc_step in anc_steps:
            env_step = env_steps[step_index]
            while env_step < anc_step and step_index < len(values) - 1:
                step_index += 1
                env_step = env_steps[step_index]
            if env_step < anc_step:
                failed = True
                break
            if step_index == 0:
                adjusted_values.append(values[step_index])
            else:
                adjusted_values.append(interp(
                    env_steps[step_index - 1], env_steps[step_index], 
                    values[step_index - 1], values[step_index],
                    anc_step,
                ))
        if failed: continue
                
        if len(adjusted_values) < min_length_v:
            continue
            
        seed = run["config"]["seed"]
        if seed in seeds:
            continue
        seeds.add(seed)
        
        metric_list.append(adjusted_values)

        if min_length is None or min_length > len(adjusted_values):
            min_length = len(adjusted_values)
    
    if len(metric_list) == 0:
        return None
    
    additional_str = ""
    for k, v in conf.items():
        additional_str += f" --{k}={v}"
    additional_str += f" {extra_args}"
    for desired_seed in range(num_seeds):
        if desired_seed not in seeds:
            if is_pixel:
                cmd_file = "train_finetuning_pixels"
            else:
                cmd_file = "train_finetuning"
            print(f"python {cmd_file}.py --project_name={patch_name} --seed={desired_seed}" + additional_str)

    metrics = np.stack([metric[:min_length] for metric in metric_list], axis=0)
    metrics = normalize_performance(metrics, conf["env_name"])
    return anc_steps[:min_length].copy(), metrics, ids

if __name__ == '__main__':
    pass
