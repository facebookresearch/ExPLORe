# Exploration from Prior Data by Labeling Optimistic Reward (ExPLORe)

This is code to accompany the NeurIPS 2023 paper [Accelerating Exploration with Unlabeled Prior Data](https://arxiv.org/abs/2311.05067).

The code is built off from https://github.com/ikostrikov/rlpd/ and the ICVF implementation is from https://github.com/dibyaghosh/icvf_release/.

ExPLORe is licensed under CC-BY-NC, however portions of the project (indicated by the header in each file influenced) are available under separate license terms: 
- rlpd (https://github.com/ikostrikov/) is licensed under the MIT license.
- icvf (https://github.com/dibyaghosh/icvf_release) is licensed under the MIT license.

# Installation (assumes CUDA 11)

```bash
./create_env
```

## D4RL Antmaze
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning.py \
    --env_name=antmaze-large-diverse-v2 \
    --max_steps=300000 \
    --config.backup_entropy=False \
    --config.num_min_qs=1 \
    --offline_relabel_type=pred \
    --use_rnd_offline=True \
    --eval_episodes=10 \
    --project_name=explore-antmaze \
    --seed=0
```

## Adroit Binary

First, download and unzip `.npy` files into `~/.datasets/awac-data/` from [here](https://drive.google.com/file/d/1SsVaQKZnY5UkuR78WrInp9XxTdKHbF0x/view).

Make sure you have `mjrl` installed:
```bash
git clone https://github.com/aravindr93/mjrl
cd mjrl
pip install -e .
```

Then, recursively clone `mj_envs` from this fork:
```bash
git clone --recursive https://github.com/philipjball/mj_envs.git
```

Then sync the submodules (add the `--init` flag if you didn't recursively clone):
```bash
$ cd mj_envs  
$ git submodule update --remote
```

Finally:
```bash
$ pip install -e .
```

Now you can run the following in this directory
```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning.py \
    --env_name=pen-binary-v0 \
    --max_steps=1000000 \
    --config.backup_entropy=False \
    --offline_relabel_type=pred \
    --use_rnd_offline=True \
    --eval_episodes=10 \
    --project_name=explore-adroit  \
    --seed=0
```

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning.py \
    --env_name=relocate-binary-v0 \
    --reset_rm_every=1000 \
    --max_steps=1000000 \
    --config.backup_entropy=False \
    --offline_relabel_type=pred \
    --use_rnd_offline=True \
    --project_name=explore-adroit \
    --eval_episodes=10 \
    --seed=0
```

## COG
Based on https://github.com/avisingh599/cog. 

First, install roboverse for COG: https://github.com/avisingh599/roboverse, for the environment. Follow the instructions in `data/README.md` to obtain the dataset.

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false python train_finetuning_pixels.py \
    --env_name=Widow250PickTray-v0 \
    --project_name=explore-cog  \
    --offline_relabel_type=pred \
    --use_rnd_offline=True \
    --use_icvf=True \
    --dataset_subsample_ratio=0.1 \
    --seed=0
```

## For SLURM and reproducing the paper results

`./generate_all.sh [partition] [conda environment] [resource constraint]` will generate all the sbatch scripts (with three seeds. The paper uses 10 for AntMaze and Adroit, 20 for COG) under `sbatch/` and `./submit_all.sh` will launch all of them.

### To reproduce main figures in the paper
Assuming that all the experiments above completed successfully with wandb tracking. The following steps can be followed to generate paper-style figures. 
```
cd plotting
python wandb_dl.py --entity=[YOUR WANDB ENTITY/USERNAME] --domain=antmaze --project_name=release-explore-antmaze
python wandb_dl.py --entity=[YOUR WANDB ENTITY/USERNAME] --domain=adroit --project_name=release-explore-adroit
python wandb_dl.py --entity=[YOUR WANDB ENTITY/USERNAME] --domain=cog --project_name=release-explore-cog

python make_plots.py --domain=all
```

### To reproduce Figure 2
```
run_antmaze.sh
cd plotting
python visualize_maze.py
```

# Bibtex
```
@inproceedings{
li2023accelerating,
title={Accelerating Exploration with Unlabeled Prior Data},
author={Qiyang Li and Jason Zhang and Dibya Ghosh and Amy Zhang and Sergey Levine},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=Itorzn4Kwf}
}
```
