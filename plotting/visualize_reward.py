import types
import sys
sys.path.append('../')

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.offsetbox import (DrawingArea, OffsetImage, AnnotationBbox)

import roboverse
import types
from visualize import *

from flax.training import checkpoints
from flax.core import frozen_dict

import roboverse

from rlpd.agents import PixelRND, PixelRM
from rlpd.wrappers import wrap_pixels
from gym.wrappers import FilterObservation, TimeLimit, RecordEpisodeStatistics

from collections import defaultdict

###### LOAD SUCCESSFUL PRIOR TRAJECTORY ######

successful_task1_path = '../data/closeddrawer_small/successful/prior_success.npy'
successful_task2_path = '../data/closeddrawer_small/successful/task_success.npy'

def dict_to_list(D):
    # https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
    return [dict(zip(D, t)) for t in zip(*D.values())]

def make_data_dict(tran):
    return dict(
        observations={"pixels": np.array(tran["observations"]["image"])[..., None]},
        actions=np.array(tran["actions"]),
        next_observations={"pixels": np.array(tran["next_observations"]["image"])[..., None]},
        rewards=np.array(tran["rewards"]),
        masks=1-np.array(tran["terminals"], dtype=float),
        dones=np.array(tran["agent_infos"]["done"])
    )

t1 = np.load(successful_task1_path, allow_pickle=True)
t2 = np.load(successful_task2_path, allow_pickle=True)
successful_t1_trajs = []
successful_t2_trajs = []

for traj in t1:
    trans = dict_to_list(traj)
    trans = [make_data_dict(tran) for tran in trans]
    successful_t1_trajs.append(trans)

for traj in t2:
    trans = dict_to_list(traj)
    trans = [make_data_dict(tran) for tran in trans]
    successful_t2_trajs.append(trans)

successful_trajs = [successful_t1_trajs[i] + successful_t2_trajs[i] \
                    for i in range(min(len(successful_t1_trajs), len(successful_t2_trajs)))]
images = []
for traj in successful_trajs:
    images.append([])
    for tran in traj:
        images[-1].append(tran['observations']['pixels'].squeeze())

###### RECREATE TRAIN STATE ######

def wrap(env):
    return wrap_pixels(
        env,
        action_repeat=1,
        image_size=48,
        num_stack=1,
        camera_id=0,
    )

def render(env, *args, **kwargs):
    return env.render_obs()

env_name = "Widow250DoubleDrawerOpenGraspNeutral-v0"

env = roboverse.make(env_name, transpose_image=False)
env.render = types.MethodType(render, env)
env = FilterObservation(env, ['image'])
env = TimeLimit(env, max_episode_steps=50)
env, pixel_keys = wrap(env)
env = RecordEpisodeStatistics(env, deque_size=1)
env.seed(0)

rnd_kwargs = dict(
    cnn_features = (32, 64, 128, 256),
    cnn_filters = (3, 3, 3, 3),
    cnn_strides = (2, 2, 2, 2),
    cnn_padding = "VALID",
    latent_dim = 50,
    encoder = "d4pg",
    lr=3e-4,
    hidden_dims=(256, 256),
    coeff=1.
)

rnd_base = PixelRND.create(
    0, env.observation_space, env.action_space, pixel_keys=pixel_keys, **rnd_kwargs
)

rm_kwargs = dict(
    cnn_features = (32, 64, 128, 256),
    cnn_filters = (3, 3, 3, 3),
    cnn_strides = (2, 2, 2, 2),
    cnn_padding = "VALID",
    latent_dim = 50,
    encoder = "d4pg",
    lr = 3e-4,
    hidden_dims = (256, 256),
)

rm_base = PixelRM.create(
    0, env.observation_space, env.action_space, pixel_keys=pixel_keys, **rm_kwargs
)

###### EVALUATE AND COLLECT REWARDS ######
seeds = list(range(20))
env_step = 25000

rm = PixelRM.create(
    0, env.observation_space, env.action_space, pixel_keys=pixel_keys, **rm_kwargs
)
icvf_rm = PixelRM.create(
    1, env.observation_space, env.action_space, pixel_keys=pixel_keys, **rm_kwargs
)
rnd = PixelRND.create(
    2, env.observation_space, env.action_space, pixel_keys=pixel_keys, **rnd_kwargs
)
icvf_rnd = PixelRND.create(
    3, env.observation_space, env.action_space, pixel_keys=pixel_keys, **rnd_kwargs
)

# seeds = []
icvf_rnd_rewards_ind_seed = []
rnd_rewards_ind_seed = []
icvf_rm_rewards_ind_seed = []
rm_rewards_ind_seed = []
for i, seed in enumerate(seeds):
    icvf_rnd_path = f"../exp_data_cog/{env_name}-s{seed}-icvf_True-ours_True/checkpoints/"
    rnd_path = f"../exp_data_cog/{env_name}-s{seed}-icvf_False-ours_True/checkpoints/"
    icvf_rm_path = f"../exp_data_cog/{env_name}-s{seed}-icvf_True-ours_True/checkpoints/"
    rm_path = f"../exp_data_cog/{env_name}-s{seed}-icvf_False-ours_True/checkpoints/"

    icvf_rnd = checkpoints.restore_checkpoint(icvf_rnd_path, target=icvf_rnd, prefix="rnd_checkpoint_", step=env_step)
    rnd = checkpoints.restore_checkpoint(rnd_path, target=rnd, prefix="rnd_checkpoint_", step=env_step)
    icvf_rm = checkpoints.restore_checkpoint(icvf_rm_path, target=icvf_rm, prefix="rm_checkpoint_", step=env_step)
    rm = checkpoints.restore_checkpoint(rm_path, target=rm, prefix="rm_checkpoint_", step=env_step)
    
    icvf_rnd_rewards_list = defaultdict(list)
    rnd_rewards_list = defaultdict(list)
    icvf_rm_rewards_list = defaultdict(list)
    rm_rewards_list = defaultdict(list)

    for t, successful_traj in enumerate(successful_trajs):
        icvf_rnd_rewards, rnd_rewards, icvf_rm_rewards, rm_rewards = \
             [], [], [], []
        for tran in successful_traj:
            icvf_rnd_rewards.append(icvf_rnd.get_reward(frozen_dict.freeze(tran)).item())
            rnd_rewards.append(rnd.get_reward(frozen_dict.freeze(tran)).item())
            icvf_rm_rewards.append(icvf_rm.get_reward(frozen_dict.freeze(tran)).item())
            rm_rewards.append(rm.get_reward(frozen_dict.freeze(tran)).item())
        
        icvf_rnd_rewards_list[t].append(np.array(icvf_rnd_rewards))
        rnd_rewards_list[t].append(np.array(rnd_rewards))
        icvf_rm_rewards_list[t].append(np.array(icvf_rm_rewards))
        rm_rewards_list[t].append(np.array(rm_rewards))
        
icvf_rnd_rewards = []
rnd_rewards = []
icvf_rm_rewards = []
rm_rewards = []
for t in range(len(successful_trajs)):
    icvf_rnd_rewards.append(np.stack(icvf_rnd_rewards_list[t], axis=0))
    rnd_rewards.append(np.stack(rnd_rewards_list[t], axis=0))
    icvf_rm_rewards.append(np.stack(icvf_rm_rewards_list[t], axis=0))
    rm_rewards.append(np.stack(rm_rewards_list[t], axis=0))

###### MAKING PLOTS ######

def plot_reward(icvf_rewards, norm_rewards, images, t):
    n, T = norm_rewards.shape
    def plot_single(ax, rewards, label):
        # normalize
        mean_traj = rewards.mean(axis=1, keepdims=True)
        std_traj = rewards.std(axis=1, keepdims=True)
        rewards = (rewards - mean_traj) / (std_traj + 1e-5)
        
        mean = rewards.mean(axis=0)
        sterr = rewards.std(axis=0) / np.sqrt(n)
        
        ax.plot(range(T), mean, label=label, linewidth=10)
        ax.fill_between(range(T), mean - sterr, mean + sterr, alpha=0.25)

    fig, ax = plt.subplots(figsize=(15, 5))
    plot_single(ax, icvf_rewards, 'Ours + ICVF')
    plot_single(ax, norm_rewards, 'Ours')

    for i in range(0, T, 5):
        image = images[i]
        imagebox = OffsetImage(image, zoom=1.7)
        imagebox.image.axes = ax

        ab = AnnotationBbox(
            imagebox, (i, 0),
            xybox=(0, -30),
            xycoords=("data", "axes fraction"),
            boxcoords="offset points",
            box_alignment=(.5, 1),
            bboxprops={"edgecolor": "none"}
        )

        ax.add_artist(ab)

    plt.legend(fontsize=20)
    plt.xticks(ticks=range(0, T, 5), fontsize=24)
    plt.tick_params(left=False, labelleft=False)
    plt.title(f'At {env_step // 1000}k Environment Steps', fontsize=28)
    plt.xlabel('Trajectory Steps', labelpad=-55, fontsize=24)
    plt.ylabel('Normalized Reward', fontsize=24)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    plt.savefig(f'figures/icvf_reward_effect-{env_step}-{t}.pdf')

for t in range(len(images)):
    plt.clf()
    plot_reward(icvf_rnd_rewards[t] + icvf_rm_rewards[t], 
                rnd_rewards[t] + rm_rewards[t], 
                images[t], t)
