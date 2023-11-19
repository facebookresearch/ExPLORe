# Ours
python submit.py -j 2 --name adroit-main --partition $1 --conda_env_name $2 --constraint $3 python train_finetuning.py \
    --env_name=pen-binary-v0,door-binary-v0,relocate-binary-v0 \
    --max_steps=1000000 \
    --config.backup_entropy=False  \
    --project_name=release-explore-adroit \
    --offline_relabel_type=pred \
    --seed=0,1,2 \
    --use_rnd_offline=True,False

# Reset reward function for relocate
python submit.py -j 2 --name adroit-main-relocate-reset --partition $1 --conda_env_name $2 --constraint $3 python train_finetuning.py \
    --env_name=relocate-binary-v0 \
    --reset_rm_every=1000 \
    --max_steps=1000000 \
    --config.backup_entropy=False  \
    --project_name=release-explore-adroit \
    --offline_relabel_type=pred \
    --seed=0,1,2 \
    --use_rnd_offline=True,False

# BC + JSRL
python submit.py -j 2 --name adroit-bc-jsrl --partition $1 --conda_env_name $2 --constraint $3 python train_finetuning.py \
    --env_name=pen-binary-v0,door-binary-v0,relocate-binary-v0 \
    --max_steps=1000000 \
    --config.backup_entropy=False \
    --project_name=release-explore-adroit \
    --bc_pretrain_rollin=0.5 \
    --offline_ratio=0.0 \
    --seed=0,1,2 \
    --bc_pretrain_steps=100000

# Naive + BC
python submit.py -j 2 --name adroit-bc --partition $1 --conda_env_name $2 --constraint $3 python train_finetuning.py \
    --env_name=pen-binary-v0,door-binary-v0,relocate-binary-v0 \
    --max_steps=1000000 \
    --config.backup_entropy=False \
    --project_name=release-explore-adroit \
    --seed=0,1,2 \
    --offline_relabel_type=pred \
    --offline_ratio=0.5 \
    --config.bc_coeff=0.01

# Oracle
python submit.py -j 2 --name adroit-oracle --partition $1 --conda_env_name $2 --constraint $3 python train_finetuning.py \
    --env_name=pen-binary-v0,door-binary-v0,relocate-binary-v0 \
    --max_steps=1000000 \
    --config.backup_entropy=False  \
    --project_name=release-explore-adroit \
    --offline_relabel_type=gt \
    --offline_ratio=0.5 \
    --seed=0,1,2

# Online, Online + RND
python submit.py -j 2 --name adroit-online --partition $1 --conda_env_name $2 --constraint $3 python train_finetuning.py \
    --env_name=pen-binary-v0,door-binary-v0,relocate-binary-v0 \
    --max_steps=1000000 \
    --config.backup_entropy=False  \
    --project_name=release-explore-adroit \
    --offline_relabel_type=gt \
    --offline_ratio=0 \
    --seed=0,1,2 \
    --use_rnd_online=True,False

# Min
python submit.py -j 2 --name adroit-min --partition $1 --conda_env_name $2 --constraint $3 python train_finetuning.py \
    --env_name=pen-binary-v0,door-binary-v0,relocate-binary-v0 \
    --max_steps=1000000 \
    --config.backup_entropy=False  \
    --project_name=release-explore-adroit \
    --offline_relabel_type=min \
    --offline_ratio=0.5 \
    --seed=0,1,2
