# Ours
python submit.py -j 3 --name antmaze-main --partition $1 --conda_env_name $2 --constraint $3 python train_finetuning.py \
    --env_name=antmaze-umaze-v2,antmaze-umaze-diverse-v2,antmaze-medium-diverse-v2,antmaze-medium-play-v2,antmaze-large-play-v2,antmaze-large-diverse-v2 \
    --max_steps=300000 \
    --config.backup_entropy=False  \
    --config.num_min_qs=1 \
    --project_name=release-explore-antmaze \
    --offline_relabel_type=pred \
    --use_rnd_offline=True,False \
    --seed=0,1,2

# BC + JSRL
python submit.py -j 3 --name antmaze-bc-jsrl --partition $1 --conda_env_name $2 --constraint $3 python train_finetuning.py \
    --env_name=antmaze-umaze-v2,antmaze-umaze-diverse-v2,antmaze-medium-diverse-v2,antmaze-medium-play-v2,antmaze-large-play-v2,antmaze-large-diverse-v2 \
    --max_steps=300000 \
    --config.backup_entropy=False \
    --config.num_min_qs=1 \
    --project_name=release-explore-antmaze \
    --bc_pretrain_rollin=0.9 \
    --offline_ratio=0.0 \
    --bc_pretrain_steps=5000 \
    --seed=0,1,2

# Naive + BC
python submit.py -j 3 --name antmaze-bc --partition $1 --conda_env_name $2 --constraint $3 python train_finetuning.py \
    --env_name=antmaze-umaze-v2,antmaze-umaze-diverse-v2,antmaze-medium-diverse-v2,antmaze-medium-play-v2,antmaze-large-play-v2,antmaze-large-diverse-v2 \
    --max_steps=300000 \
    --config.backup_entropy=False \
    --config.num_min_qs=1 \
    --project_name=release-explore-antmaze \
    --offline_relabel_type=pred \
    --offline_ratio=0.5 \
    --config.bc_coeff=0.01 \
    --seed=0,1,2

# Oracle
python submit.py -j 3 --name antmaze-oracle --partition $1 --conda_env_name $2 --constraint $3 python train_finetuning.py \
    --env_name=antmaze-umaze-v2,antmaze-umaze-diverse-v2,antmaze-medium-diverse-v2,antmaze-medium-play-v2,antmaze-large-play-v2,antmaze-large-diverse-v2 \
    --max_steps=300000 \
    --config.backup_entropy=False  \
    --config.num_min_qs=1 \
    --project_name=release-explore-antmaze \
    --offline_relabel_type=gt \
    --offline_ratio=0.5 \
    --seed=0,1,2

# Online, Online + RND
python submit.py -j 3 --name antmaze-online --partition $1 --conda_env_name $2 --constraint $3 python train_finetuning.py \
    --env_name=antmaze-umaze-v2,antmaze-umaze-diverse-v2,antmaze-medium-diverse-v2,antmaze-medium-play-v2,antmaze-large-play-v2,antmaze-large-diverse-v2 \
    --max_steps=300000 \
    --config.backup_entropy=False  \
    --config.num_min_qs=1 \
    --project_name=release-explore-antmaze \
    --offline_relabel_type=gt \
    --offline_ratio=0 \
    --use_rnd_online=True,False \
    --seed=0,1,2

# Min
python submit.py -j 3 --name antmaze-min --partition $1 --conda_env_name $2 --constraint $3 python train_finetuning.py \
    --env_name=antmaze-umaze-v2,antmaze-umaze-diverse-v2,antmaze-medium-diverse-v2,antmaze-medium-play-v2,antmaze-large-play-v2,antmaze-large-diverse-v2 \
    --max_steps=300000 \
    --config.backup_entropy=False  \
    --config.num_min_qs=1 \
    --project_name=release-explore-antmaze \
    --offline_relabel_type=min \
    --offline_ratio=0.5 \
    --seed=0,1,2
