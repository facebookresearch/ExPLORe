"""
Modified from https://github.com/ikostrikov/rlpd/blob/main/rlpd/train_finetuning.py

Original lincense information:

MIT License

Copyright (c) 2022 Ilya Kostrikov, Philip J. Ball, Laura Smith

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""


#! /usr/bin/env python
import os
import pickle
from functools import partial

import jax

import gym
from gym import spaces
import numpy as np
import tqdm
from absl import app, flags
from flax.core import frozen_dict
from absl import logging

import jax.numpy as jnp

logging.set_verbosity(logging.FATAL)

try:
    from flax.training import checkpoints
except:
    print("Not loading checkpointing functionality.")
from ml_collections import config_flags

from rlpd.agents import SACLearner, RM, RND, BCAgent

from rlpd.data import ReplayBuffer, Dataset
from rlpd.data.d4rl_datasets import D4RLDataset, filter_antmaze
try:
    from rlpd.data.binary_datasets import BinaryDataset
except:
    print("not importing binary dataset")
from rlpd.wrappers import wrap_gym

from rlpd.evaluation import evaluate

##### IMPORT NEEDED FOR GC-IQL #####
### logging imports ###
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import plotly.express as px
from visualize import plot_value, plot_trajectories, plot_points, plot_data_directions
from visualize import get_canvas_image
import wandb

import glob
import os
import time
from datetime import datetime

from absl import app, flags
from functools import partial
import jax
import jax.numpy as jnp
import flax

import tqdm
import wandb

from ml_collections import config_flags
import pickle

##### IMPORT NEEDED FOR GC-IQL #####

DEBUG = 1
FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "rlpd", "wandb project name.")
flags.DEFINE_string("env_name", "halfcheetah-expert-v2", "D4rl dataset name.")
flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 100, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 10000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", int(1e6), "Number of training steps.")
flags.DEFINE_integer("start_training", 5000, "Number of training steps to start training.")

flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_boolean("checkpoint_model", False, "Save agent checkpoint on evaluation.")
flags.DEFINE_boolean("checkpoint_buffer", False, "Save agent replay buffer on evaluation.")
flags.DEFINE_boolean("binary_include_bc", True, "Whether to include BC data in the binary datasets.")

flags.DEFINE_integer("utd_ratio", 20, "Update to data ratio.")

flags.DEFINE_string("offline_relabel_type", "gt", "one of [gt/pred/min]")
flags.DEFINE_string("exp_prefix", "exp_data/default", "log directory")

flags.DEFINE_boolean("use_rnd_offline", False, "Whether to use rnd offline.")
flags.DEFINE_boolean("use_rnd_online", False, "Whether to use rnd online.")

flags.DEFINE_float("bc_pretrain_rollin", 0.0, "rollin coeff")
flags.DEFINE_integer("bc_pretrain_steps", 5000, "Pre-train BC policy for a number of steps on pure offline data")

flags.DEFINE_integer("reset_rm_every", -1, "Reset the reward network every N env steps")

flags.DEFINE_string('filter_data_mode', 'all', 'Strategy to filter offline data')

config_flags.DEFINE_config_file(
    "config",
    "configs/rlpd_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "rm_config",
    "configs/rm_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "rnd_config",
    "configs/rnd_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "bc_config",
    "configs/bc_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def combine(one_dict, other_dict):
    combined = {}

    for k, v in one_dict.items():
        if isinstance(v, dict):
            combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty(
                (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
            )
            tmp[0::2] = v
            tmp[1::2] = other_dict[k]
            combined[k] = tmp

    return combined

def add_prefix(prefix, dict):
    return {prefix + k: v for k, v in dict.items()}

@partial(jax.jit, static_argnames=("R",))
def check_overlap(coord, observations, R):
    return jnp.any(jnp.all(jnp.abs(coord - observations[..., :2]) <= R, axis=-1))

def view_data_distribution(viz_env, ds):
    vobs = ds.dataset_dict['observations'][..., :2]
    return plot_points(viz_env, vobs[:,0], vobs[:,1])

def main(_):
    assert FLAGS.offline_ratio <= 1.0

    wandb.init(project=FLAGS.project_name)
    wandb.config.update(FLAGS)

    exp_prefix = f"{FLAGS.exp_prefix}-s{FLAGS.seed}"
    log_dir = os.path.join(FLAGS.log_dir, exp_prefix)

    if FLAGS.checkpoint_model:
        chkpt_dir = os.path.join(log_dir, "checkpoints")
        os.makedirs(chkpt_dir, exist_ok=True)

    if FLAGS.checkpoint_buffer:
        buffer_dir = os.path.join(log_dir, "buffers")
        os.makedirs(buffer_dir, exist_ok=True)

    ########### ENVIRONMENT ###########
    env = gym.make(FLAGS.env_name)
    env = wrap_gym(env, rescale_actions=True)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)

    eval_env = gym.make(FLAGS.env_name)
    eval_env = wrap_gym(eval_env, rescale_actions=True)
    eval_env.seed(FLAGS.seed + 42)

    if "binary" in FLAGS.env_name:
        ds = BinaryDataset(env, include_bc_data=FLAGS.binary_include_bc)
    else:
        ds = D4RLDataset(env)
    
    ds.seed(FLAGS.seed)

    if "antmaze" in FLAGS.env_name:
        from visualize import get_env_and_dataset
        viz_env, viz_dataset = get_env_and_dataset(FLAGS.env_name)
        coords, S = viz_env.get_coord_list()

    action_space = env.action_space

    ds_minr = ds.dataset_dict["rewards"].min()
    print(f"dataset minimum reward = {ds_minr}")
    print("observation shape:", env.observation_space.sample().shape)
    print("action shape:", action_space.sample().shape)

    replay_buffer = ReplayBuffer(
        env.observation_space, action_space, FLAGS.max_steps
    )
    replay_buffer.seed(FLAGS.seed)

    ########### MODELS ###########
    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed, env.observation_space, action_space, **kwargs
    )

    if FLAGS.use_rnd_offline or FLAGS.use_rnd_online:
        kwargs = dict(FLAGS.rnd_config)
        model_cls = kwargs.pop("model_cls")
        rnd = globals()[model_cls].create(
            FLAGS.seed + 123, env.observation_space, action_space, **kwargs
        )
    else:
        rnd = None

    if FLAGS.offline_relabel_type == "gt":
        rm = None
    else:
        kwargs = dict(FLAGS.rm_config)
        model_cls = kwargs.pop("model_cls")
        rm = globals()[model_cls].create(
            FLAGS.seed + 123, env.observation_space, action_space, **kwargs
        )

    if FLAGS.bc_pretrain_rollin > 0.:
        kwargs = dict(FLAGS.bc_config)
        model_cls = kwargs.pop("model_cls")
        bc_policy = globals()[model_cls].create(
            FLAGS.seed + 152, env.observation_space, action_space, **kwargs
        )
    else:
        bc_policy = None

    # Pre-training
    record_step = 0
    if bc_policy is not None:
        for i in tqdm.tqdm(range(FLAGS.bc_pretrain_steps), smoothing=0.1, disable=not FLAGS.tqdm):
            record_step += 1
            batch = ds.sample(
                int(FLAGS.batch_size * FLAGS.utd_ratio)
            )
            bc_policy, update_info = bc_policy.update(batch, FLAGS.utd_ratio)
            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log(add_prefix("bc/", {k: v}), step=record_step)

    observation, done = env.reset(), False

    online_trajs = []
    online_traj = [observation]

    rng = jax.random.PRNGKey(seed=FLAGS.seed)

    if FLAGS.bc_pretrain_rollin > 0.:
        curr_rng, rng = jax.random.split(rng)
        rollin_enabled = True if jax.random.uniform(key=curr_rng) < FLAGS.bc_pretrain_rollin else False
    else:
        rollin_enabled = False

    env_step = 0
    for i in tqdm.tqdm(
        range(0, FLAGS.max_steps + 1), smoothing=0.1, disable=not FLAGS.tqdm
    ):

        if FLAGS.reset_rm_every != -1 and i % FLAGS.reset_rm_every == FLAGS.reset_rm_every - 1:
            kwargs = dict(FLAGS.rm_config)
            model_cls = kwargs.pop("model_cls")
            rm = globals()[model_cls].create(
                FLAGS.seed + 123, env.observation_space, action_space, **kwargs
            )  

        if env_step > FLAGS.max_steps:  # done after max_steps achieved
            break
        
        record_step += 1
        if rollin_enabled:
            action, bc_policy = bc_policy.sample_actions(observation)
            curr_rng, rng = jax.random.split(rng)
            rollin_enabled = True if jax.random.uniform(key=curr_rng) <= agent.discount else False
        else:
            if i < FLAGS.start_training:
                action = action_space.sample()
            else:

                action, agent = agent.sample_actions(observation)
      
        next_observation, reward, done, info = env.step(action)
        env_step += 1

        online_traj.append(next_observation)

        timelimit_stop = "TimeLimit.truncated" in info

        if not done or timelimit_stop:
            mask = 1.0
        else:
            mask = 0.0

        replay_buffer.insert(
            dict(
                observations=observation,
                actions=action,
                rewards=reward,
                masks=mask,
                dones=done,
                next_observations=next_observation,
            )
        )

        if i >= FLAGS.start_training:
            # standard online batch
            online_batch_size = int(FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio))
            online_batch = replay_buffer.sample(online_batch_size)
            online_batch = online_batch.unfreeze()
            if "antmaze" in FLAGS.env_name:
                online_batch['rewards'] -= 1

            if FLAGS.use_rnd_online:
                online_batch["rewards"] += rnd.get_reward(online_batch["observations"], online_batch["actions"])
            
            batch = online_batch

            # append offline batch
            if FLAGS.offline_ratio > 0:
                offline_batch_size = int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio)
                offline_batch = ds.sample(offline_batch_size)

                offline_batch = offline_batch.unfreeze()

                if "antmaze" in FLAGS.env_name:
                    offline_batch["rewards"] -= 1

                if FLAGS.offline_relabel_type == "gt":
                    pass
                elif FLAGS.offline_relabel_type == "pred":
                    offline_batch["rewards"] = rm.get_reward(offline_batch["observations"], offline_batch["actions"])
                    offline_batch["masks"] = rm.get_mask(offline_batch["observations"], offline_batch["actions"])
                elif FLAGS.offline_relabel_type == "min":
                    offline_batch["rewards"][:] = ds_minr
                    offline_batch["masks"] = rm.get_mask(offline_batch["observations"], offline_batch["actions"])
                    if "antmaze" in FLAGS.env_name:
                        offline_batch["rewards"] -= 1
                else:
                    raise NotImplementedError

                if FLAGS.use_rnd_offline:
                    offline_batch["rewards"] = offline_batch["rewards"] + rnd.get_reward(offline_batch["observations"], offline_batch["actions"])

                # only enable the bc loss on offline data
                offline_batch["bc_masks"] = jnp.ones_like(offline_batch["masks"])
                batch["bc_masks"] = jnp.zeros_like(batch["masks"])
                batch = combine(offline_batch, batch)

            agent, update_info = agent.update(batch, FLAGS.utd_ratio)
            
            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log(add_prefix("agent/", {k: v}), step=record_step)

                print(update_info)

        if i >= FLAGS.start_training * 2 and (rm is not None or rnd is not None):  # start training.
            online_batch = replay_buffer.sample(int(FLAGS.batch_size * FLAGS.utd_ratio))
            online_batch = online_batch.unfreeze()
            if "antmaze" in FLAGS.env_name:
                online_batch['rewards'] -= 1
            
            if rm is not None:
                rm, rm_update_info = rm.update(online_batch, FLAGS.utd_ratio)

            offline_batch = ds.sample(int(FLAGS.batch_size * FLAGS.utd_ratio))

            offline_batch = offline_batch.unfreeze()
            if "antmaze" in FLAGS.env_name:
                offline_batch["rewards"] -= 1
            
            if rm is not None:
                rm_update_info.update(rm.evaluate(offline_batch))

            if rnd is not None:
                rnd, rnd_update_info = rnd.update({
                    "observations": observation[None],
                    "actions": action[None],
                    "next_observations": next_observation[None],
                    "rewards": np.array(reward)[None],
                    "masks": np.array(mask)[None],
                    "dones": np.array(done)[None],
                })

            if i % FLAGS.log_interval == 0:
                if rm is not None:
                    for k, v in rm_update_info.items():
                        wandb.log(add_prefix("rm/", {k: v}), step=record_step)
                if rnd is not None:
                    for k, v in rnd_update_info.items():
                        wandb.log(add_prefix("rnd/", {k: v}), step=record_step)
        
        if i % FLAGS.log_interval == 0:
            wandb.log({"env_step": env_step}, step=record_step)

        observation = next_observation

        if done:
            online_trajs.append({"observation": np.stack(online_traj, axis=0)})
            observation, done = env.reset(), False
            online_traj = [observation]
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log(add_prefix("episode/", {decode[k]: v}), step=record_step)

            if FLAGS.bc_pretrain_rollin > 0.:
                curr_rng, rng = jax.random.split(rng)
                rollin_enabled = True if jax.random.uniform(key=curr_rng) < FLAGS.bc_pretrain_rollin else False

        if i % FLAGS.eval_interval == 0:
            
            eval_info, trajs = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
            )

            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=record_step)

            if FLAGS.bc_pretrain_rollin > 0.:
                bc_eval_info, bc_trajs = evaluate(
                    bc_policy,
                    eval_env,
                    num_episodes=FLAGS.eval_episodes,
                )
                for k, v in bc_eval_info.items():
                    wandb.log({f"bc-evaluation/{k}": v}, step=record_step)

            if "antmaze" in FLAGS.env_name:

                num_overlapped = 0
                for (x, y) in coords:
                    coord = jnp.array([x, y])
                    overlapped = False
                    for batch in replay_buffer.get_iter(FLAGS.batch_size):
                        if check_overlap(coord, batch["observations"], S / 2):
                            overlapped = True
                            break
                    if overlapped:
                        num_overlapped += 1
                wandb.log({"coverage": num_overlapped / len(coords)}, step=record_step)

                fig = plt.figure(tight_layout=True, figsize=(4, 4), dpi=200)
                canvas = FigureCanvas(fig)
                plot_trajectories(viz_env, viz_dataset, online_trajs, fig, plt.gca())
                online_trajs = []
                image = wandb.Image(get_canvas_image(canvas))
                wandb.log({f"visualize/trajs": image}, step=record_step)
                plt.close(fig)

                data_distribution_im = view_data_distribution(viz_env, ds)
                image = wandb.Image(data_distribution_im)
                wandb.log({f"visualize/offline_data_dist": image}, step=record_step)

                data_directions_im = plot_data_directions(viz_env, ds)
                image = wandb.Image(data_directions_im)
                wandb.log({f'visualize/offline_data_directions': image}, step=record_step)

            if FLAGS.checkpoint_model:
                try:
                    checkpoints.save_checkpoint(
                        chkpt_dir, agent, step=i, keep=20, overwrite=True
                    )
                except:
                    print("Could not save model checkpoint.")

            if FLAGS.checkpoint_buffer:
                with open(os.path.join(buffer_dir, f"buffer.npz"), "wb") as f:
                    np.savez(f, observations=replay_buffer.dataset_dict["observations"][:len(replay_buffer)])

if __name__ == "__main__":
    app.run(main)
