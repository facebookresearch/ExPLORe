# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

python train_finetuning.py --eval_episodes=10 --checkpoint_buffer=True --exp_prefix=exp_data/ours --env_name=antmaze-medium-diverse-v2 --max_steps=300000 --config.backup_entropy=False --config.num_min_qs=1 --project_name=release-explore-antmaze-save --offline_relabel_type=pred --seed=0 --use_rnd_offline=True --use_rnd_online=False
python train_finetuning.py --eval_episodes=10 --checkpoint_buffer=True --exp_prefix=exp_data/naive --env_name=antmaze-medium-diverse-v2 --max_steps=300000 --config.backup_entropy=False --config.num_min_qs=1 --project_name=release-explore-antmaze-save --offline_relabel_type=pred --seed=0 --use_rnd_offline=False --use_rnd_online=False
python train_finetuning.py --eval_episodes=10 --checkpoint_buffer=True --exp_prefix=exp_data/online_rnd --env_name=antmaze-medium-diverse-v2 --max_steps=300000 --config.backup_entropy=False --config.num_min_qs=1 --project_name=release-explore-antmaze-save --offline_ratio=0 --seed=0 --use_rnd_offline=False --use_rnd_online=True
python train_finetuning.py --eval_episodes=10 --checkpoint_buffer=True --exp_prefix=exp_data/online --env_name=antmaze-medium-diverse-v2 --max_steps=300000 --config.backup_entropy=False --config.num_min_qs=1 --project_name=release-explore-antmaze-save --offline_ratio=0 --seed=0 --use_rnd_offline=False --use_rnd_online=False
