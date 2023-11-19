# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-j", default=4, type=int)
parser.add_argument("--partition", type=str)
parser.add_argument("--name", type=str)
parser.add_argument("--conda_env_name", type=str)
parser.add_argument("--constraint", type=str)

args, unknown = parser.parse_known_args()

partition = args.partition
name = args.name
conda_env_name = args.conda_env_name
constraint = args.constraint

print(unknown)


def parse(args):
    prefix = ""
    for index in range(len(args)):
        prefix += " "
        arg = args[index]
        i = arg.find("=")
        if i == -1:
            content = arg
        else:
            prefix += arg[: i + 1]
            content = arg[i + 1 :]

        if "," in content:
            elements = content.split(",")
            for r in parse(args[index + 1 :]):
                for element in elements:
                    yield prefix + element + r
            return
        else:
            prefix += content
    yield prefix


python_command_list = list(parse(unknown))

num_jobs = len(python_command_list)

num_arr = (num_jobs - 1) // args.j + 1

print("\n".join(python_command_list))

path = os.getcwd()

d_str = "\n ".join(
    [
        "[{}]='{}'".format(i + 1, command[1:])
        for i, command in enumerate(python_command_list)
    ]
)

sbatch_str = f"""#!/bin/bash
#SBATCH --job-name=explore
#SBATCH --open-mode=append
#SBATCH --output=logs/out/%x_%j.txt
#SBATCH --error=logs/err/%x_%j.txt
#SBATCH --time=48:00:00
#SBATCH --array=1-{num_arr}

#SBATCH --partition={partition}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=1
#SBATCH --constraint={constraint}

TASK_ID=$((SLURM_ARRAY_TASK_ID-1))
PARALLEL_N={args.j}
JOB_N={num_jobs}

COM_ID_S=$((TASK_ID * PARALLEL_N + 1))

source ~/.bashrc

conda activate {conda_env_name}

declare -a commands=(
 {d_str}
)

cd {path}

parallel --delay 20 --linebuffer -j {args.j} {{1}} ::: \"${{commands[@]:$COM_ID_S:$PARALLEL_N}}\"
"""

with open(f"sbatch/{name}.sh", "w") as f:
    f.write(sbatch_str)
