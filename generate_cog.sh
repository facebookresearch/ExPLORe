# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Ours-ICVF
python submit.py -j 2 --name cog-main-ours --partition $1 --conda_env_name $2 --constraint $3 python train_finetuning_pixels.py \
    --env_name=Widow250PickTray-v0,Widow250DoubleDrawerOpenGraspNeutral-v0,Widow250DoubleDrawerCloseOpenGraspNeutral-v0 \
    --project_name=release-explore-cog \
    --offline_relabel_type=pred \
    --seed=0,1,2 \
    --use_rnd_offline=True,False \
    --use_icvf=True,False \
    --checkpoint_model=True

# Oracle
python submit.py -j 3 --name cog-main-oracle --partition $1 --conda_env_name $2 --constraint $3 python train_finetuning_pixels.py \
    --env_name=Widow250PickTray-v0,Widow250DoubleDrawerOpenGraspNeutral-v0,Widow250DoubleDrawerCloseOpenGraspNeutral-v0 \
    --project_name=release-explore-cog \
    --offline_relabel_type=gt \
    --seed=0,1,2

# BC + JSRL
python submit.py -j 3 --name cog-bc-jsrl --partition $1 --conda_env_name $2 --constraint $3 python train_finetuning_pixels.py \
    --env_name=Widow250PickTray-v0,Widow250DoubleDrawerOpenGraspNeutral-v0,Widow250DoubleDrawerCloseOpenGraspNeutral-v0 \
    --project_name=release-explore-cog \
    --bc_pretrain_rollin=0.5 \
    --bc_pretrain_steps=100000 \
    --offline_ratio=0.0 \
    --seed=0,1,2

# Naive + BC
python submit.py -j 3 --name cog-bc --partition $1 --conda_env_name $2 --constraint $3 python train_finetuning_pixels.py \
    --env_name=Widow250PickTray-v0,Widow250DoubleDrawerOpenGraspNeutral-v0,Widow250DoubleDrawerCloseOpenGraspNeutral-v0 \
    --project_name=release-explore-cog \
    --seed=0,1,2 \
    --offline_relabel_type=pred \
    --offline_ratio=0.5 \
    --config.bc_coeff=0.01

# Online, Online + RND
python submit.py -j 3 --name cog-online --partition $1 --conda_env_name $2 --constraint $3 python train_finetuning_pixels.py \
    --env_name=Widow250PickTray-v0,Widow250DoubleDrawerOpenGraspNeutral-v0,Widow250DoubleDrawerCloseOpenGraspNeutral-v0 \
    --project_name=release-explore-cog \
    --offline_ratio=0 \
    --seed=0,1,2 \
    --use_rnd_online=True,False

# Min
python submit.py -j 3 --name cog-min --partition $1 --conda_env_name $2 --constraint $3 python train_finetuning_pixels.py \
    --env_name=Widow250PickTray-v0,Widow250DoubleDrawerOpenGraspNeutral-v0,Widow250DoubleDrawerCloseOpenGraspNeutral-v0 \
    --project_name=release-explore-cog \
    --offline_relabel_type=min \
    --offline_ratio=0.5 \
    --seed=0,1,2
