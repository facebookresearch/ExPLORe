"""
Modified from https://github.com/ikostrikov/rlpd/blob/main/rlpd/train_finetuning_pixels.py

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

import numpy as np
import tqdm
from absl import app, flags
from flax.core import FrozenDict
from ml_collections import config_flags
from flax.core import frozen_dict
from flax.training import checkpoints

import wandb
from rlpd.agents import DrQLearner, PixelRND, PixelRM, PixelBCAgent
from rlpd.data import MemoryEfficientReplayBuffer, ReplayBuffer
from rlpd.evaluation import evaluate
from rlpd.wrappers import wrap_pixels
from rlpd.agents.drq.icvf import PixelICVF
from rlpd import gc_dataset

import matplotlib.pyplot as plt
import pickle

### cog imports ###
import roboverse
from gym.wrappers import TimeLimit, FilterObservation, RecordEpisodeStatistics
from rlpd.data import Dataset
from rlpd.data.cog_datasets import COGDataset
from functools import partial
import types

### cog imports ###

import jax
import jax.numpy as jnp

FLAGS = flags.FLAGS

flags.DEFINE_string("project_name", "explore-cog", "wandb project name.")
flags.DEFINE_string("env_name", "cheetah-run-v0", "Environment name.")

flags.DEFINE_float(
    "dataset_subsample_ratio", 0.1, "Ratio of the dataset to subsample (done twice)"
)

flags.DEFINE_bool("use_icvf", False, "Whether to use the icvf encoder")

flags.DEFINE_float("offline_ratio", 0.5, "Offline ratio.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("eval_episodes", 100, "Number of episodes used for evaluation.")
flags.DEFINE_integer("log_interval", 1000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 5000, "Eval interval.")
flags.DEFINE_integer("batch_size", 256, "Mini batch size.")
flags.DEFINE_integer("max_steps", 500000, "Number of training steps.")
flags.DEFINE_integer(
    "start_training", 5000, "Number of training steps to start training."
)
flags.DEFINE_boolean("tqdm", True, "Use tqdm progress bar.")
flags.DEFINE_string("save_dir", "exp_data_cog", "Directory to save checkpoints.")
flags.DEFINE_bool("checkpoint_model", False, "save model")
flags.DEFINE_bool("checkpoint_buffer", False, "save replay buffer")

flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")

flags.DEFINE_float("bc_pretrain_rollin", 0.0, "rollin coeff")
flags.DEFINE_integer(
    "bc_pretrain_steps",
    10000,
    "Pre-train BC policy for a number of steps on pure offline data",
)

config_flags.DEFINE_config_file(
    "config",
    "configs/rlpd_pixels_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "rm_config",
    "configs/pixel_rm_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "rnd_config",
    "configs/pixel_rnd_config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "bc_config",
    "configs/pixel_bc_config.py",
    "File path to the training hyperparameter configuration",
    lock_config=False,
)


flags.DEFINE_string(
    "offline_relabel_type",
    "gt",
    "Whether to use reward from the offline dataset. [gt/pred/min]",
)
flags.DEFINE_boolean("use_rnd_offline", False, "Whether to use rnd offline.")
flags.DEFINE_boolean("use_rnd_online", False, "Whether to use rnd online.")


def combine(one_dict, other_dict):
    combined = {}

    for k, v in one_dict.items():
        if isinstance(v, FrozenDict) or isinstance(v, dict):
            if len(v) == 0:
                combined[k] = v
            else:
                combined[k] = combine(v, other_dict[k])
        else:
            tmp = np.empty(
                (v.shape[0] + other_dict[k].shape[0], *v.shape[1:]), dtype=v.dtype
            )
            tmp[0::2] = v
            tmp[1::2] = other_dict[k]
            combined[k] = tmp

    return FrozenDict(combined)


def add_prefix(prefix, dict):
    return {prefix + k: v for k, v in dict.items()}


def main(_):
    wandb.init(project=FLAGS.project_name, mode="online")
    wandb.config.update(FLAGS)

    if FLAGS.save_dir is not None:
        log_dir = os.path.join(
            FLAGS.save_dir,
            f"{FLAGS.env_name}-s{FLAGS.seed}-icvf_{FLAGS.use_icvf}-ours_{FLAGS.use_rnd_offline}",
        )
        print("logging to", log_dir)

        if FLAGS.checkpoint_model:
            chkpt_dir = os.path.join(log_dir, "checkpoints")
            os.makedirs(chkpt_dir, exist_ok=True)

        if FLAGS.checkpoint_buffer:
            buffer_dir = os.path.join(log_dir, "buffers")
            os.makedirs(buffer_dir, exist_ok=True)

    def wrap(env):
        return wrap_pixels(
            env,
            action_repeat=1,
            num_stack=1,
            camera_id=0,
        )

    def render(env, *args, **kwargs):
        return env.render_obs()

    if FLAGS.env_name == "Widow250PickTray-v0":
        env_name_alt = "pickplace"
        cog_max_path_length = 40
    elif FLAGS.env_name == "Widow250DoubleDrawerOpenGraspNeutral-v0":
        env_name_alt = "closeddrawer_small"
        cog_max_path_length = 50
    elif FLAGS.env_name == "Widow250DoubleDrawerCloseOpenGraspNeutral-v0":
        env_name_alt = "blockeddrawer1_small"
        cog_max_path_length = 80

    env = roboverse.make(FLAGS.env_name, transpose_image=False)
    env.render = types.MethodType(render, env)
    env = FilterObservation(env, ["image"])
    env = TimeLimit(env, max_episode_steps=cog_max_path_length)  # TODO
    env, pixel_keys = wrap(env)
    env = RecordEpisodeStatistics(env, deque_size=1)
    env.seed(FLAGS.seed)

    eval_env = roboverse.make(FLAGS.env_name, transpose_image=False)
    eval_env.render = types.MethodType(render, eval_env)
    eval_env = FilterObservation(eval_env, ["image"])
    eval_env = TimeLimit(eval_env, max_episode_steps=cog_max_path_length)  # TODO
    eval_env, _ = wrap(eval_env)
    eval_env.seed(FLAGS.seed + 42)

    dataset_path = os.path.join("data", env_name_alt)

    print("Data Path:", dataset_path)

    np_rng = np.random.default_rng(FLAGS.seed)

    ds = COGDataset(
        env=env,
        dataset_path=dataset_path,
        capacity=300000,
        subsample_ratio=FLAGS.dataset_subsample_ratio,
        np_rng=np_rng,
    )
    ds.seed(FLAGS.seed)

    ds_minr = ds.dataset_dict["rewards"][: len(ds)].min()
    assert -10 < ds_minr < 10, "maybe sampling reward outside of buffer range"

    ds_iterator = ds.get_iterator(
        sample_args={
            "batch_size": int(FLAGS.batch_size * FLAGS.utd_ratio * FLAGS.offline_ratio),
            "pack_obs_and_next_obs": True,
        }
    )

    replay_buffer = MemoryEfficientReplayBuffer(
        env.observation_space, env.action_space, FLAGS.max_steps
    )
    replay_buffer_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": int(
                FLAGS.batch_size * FLAGS.utd_ratio * (1 - FLAGS.offline_ratio)
            ),
            "pack_obs_and_next_obs": True,
        }
    )
    replay_buffer.seed(FLAGS.seed)

    ########### MODELS ###########

    # Crashes on some setups if agent is created before replay buffer.
    kwargs = dict(FLAGS.config)
    model_cls = kwargs.pop("model_cls")
    agent = globals()[model_cls].create(
        FLAGS.seed,
        env.observation_space,
        env.action_space,
        pixel_keys=pixel_keys,
        **kwargs,
    )

    if FLAGS.offline_relabel_type != "gt":
        kwargs = dict(FLAGS.rm_config)
        model_cls = kwargs.pop("model_cls")
        rm = globals()[model_cls].create(
            FLAGS.seed + 123,
            env.observation_space,
            env.action_space,
            pixel_keys=pixel_keys,
            **kwargs,
        )
    else:
        rm = None

    if FLAGS.use_rnd_offline or FLAGS.use_rnd_online:
        kwargs = dict(FLAGS.rnd_config)
        model_cls = kwargs.pop("model_cls")
        rnd = globals()[model_cls].create(
            FLAGS.seed + 123,
            env.observation_space,
            env.action_space,
            pixel_keys=pixel_keys,
            **kwargs,
        )
    else:
        rnd = None

    # Pre-training
    record_step = 0
    # ICVF training and initialize RM and RND with ICVF encoder
    if FLAGS.use_icvf:
        # assert rm is not None or rnd is not None, "ICVF is not needed in this configuration"

        icvf = PixelICVF.create(
            FLAGS.seed,
            env.observation_space,
            env.action_space,
            pixel_keys=pixel_keys,
            **dict(FLAGS.config),
        )
        gc_ds = gc_dataset.GCSDataset(ds, **gc_dataset.GCSDataset.get_default_config())

        for i in tqdm.trange(75001):

            record_step += 1
            batch = gc_ds.sample(FLAGS.batch_size)
            icvf, update_info = icvf.update(frozen_dict.freeze(batch), 1)
            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log({f"icvf-training/{k}": v}, step=record_step)

        replace_keys = ["encoder_0"]
        replace = {k: icvf.net.params[k] for k in replace_keys}

        if rnd is not None:
            new_params = FrozenDict(rnd.net.params).copy(add_or_replace=replace)
            new_frozen_params = FrozenDict(rnd.frozen_net.params).copy(
                add_or_replace=replace
            )
            rnd = rnd.replace(
                net=rnd.net.replace(params=new_params),
                frozen_net=rnd.frozen_net.replace(params=new_frozen_params),
            )

        if rm is not None:
            new_params = FrozenDict(rm.r_net.params).copy(add_or_replace=replace)
            rm = rm.replace(r_net=rm.r_net.replace(params=new_params))

    if FLAGS.bc_pretrain_rollin > 0.0:
        kwargs = dict(FLAGS.bc_config)
        model_cls = kwargs.pop("model_cls")
        bc_policy = globals()[model_cls].create(
            FLAGS.seed + 152, env.observation_space, env.action_space, **kwargs
        )

        if FLAGS.use_icvf:
            new_params = FrozenDict(bc_policy.actor.params).copy(add_or_replace=replace)
            bc_policy = bc_policy.replace(
                actor=bc_policy.actor.replace(params=new_params)
            )
    else:
        bc_policy = None

    if bc_policy is not None:
        for i in tqdm.tqdm(
            range(FLAGS.bc_pretrain_steps), smoothing=0.1, disable=not FLAGS.tqdm
        ):
            record_step += 1
            batch = ds.sample(int(FLAGS.batch_size * FLAGS.utd_ratio))
            bc_policy, update_info = bc_policy.update(batch, FLAGS.utd_ratio)
            if i % FLAGS.log_interval == 0:
                for k, v in update_info.items():
                    wandb.log(add_prefix("bc/", {k: v}), step=record_step)

    # Training
    observation, done = env.reset(), False

    rng = jax.random.PRNGKey(seed=FLAGS.seed)
    if FLAGS.bc_pretrain_rollin > 0.0:
        curr_rng, rng = jax.random.split(rng)
        rollin_enabled = (
            True
            if jax.random.uniform(key=curr_rng) < FLAGS.bc_pretrain_rollin
            else False
        )
    else:
        rollin_enabled = False

    for i in tqdm.tqdm(
        range(1, FLAGS.max_steps + 1),
        smoothing=0.1,
        disable=not FLAGS.tqdm,
    ):
        record_step += 1
        logging_info = {}

        if rollin_enabled:
            action, bc_policy = bc_policy.sample_actions(observation)
            curr_rng, rng = jax.random.split(rng)
            rollin_enabled = (
                True if jax.random.uniform(key=curr_rng) < agent.discount else False
            )
        else:
            if i < FLAGS.start_training:
                action = env.action_space.sample()
            else:
                action, agent = agent.sample_actions(observation)

        next_observation, reward, done, info = env.step(action)

        if not done or "TimeLimit.truncated" in info:
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
        observation = next_observation

        if done:
            observation, done = env.reset(), False
            for k, v in info["episode"].items():
                decode = {"r": "return", "l": "length", "t": "time"}
                wandb.log({f"episode/{decode[k]}": v}, step=record_step)

            if FLAGS.bc_pretrain_rollin > 0.0:
                curr_rng, rng = jax.random.split(rng)
                rollin_enabled = (
                    True
                    if jax.random.uniform(key=curr_rng) < FLAGS.bc_pretrain_rollin
                    else False
                )

        # main updates
        if i >= FLAGS.start_training:
            online_batch = next(replay_buffer_iterator)

            if i >= FLAGS.start_training * 2:
                # update the reward model on the online batch
                if rm is not None:
                    rm, rm_update_info = rm.update(online_batch, FLAGS.utd_ratio)
                    logging_info.update(add_prefix("rm/", rm_update_info))

                if rnd is not None:
                    rnd, rnd_update_info = rnd.update(
                        frozen_dict.freeze(
                            {
                                "observations": {
                                    k: ob[None] for k, ob in observation.items()
                                },
                                "actions": action[None],
                                "next_observations": {
                                    k: ob[None] for k, ob in next_observation.items()
                                },
                                "rewards": np.array(reward)[None],
                                "masks": np.array(mask)[None],
                                "dones": np.array(done)[None],
                            }
                        )
                    )
                    logging_info.update(add_prefix("rnd/", rnd_update_info))

            # prepare the batch for the main agent
            online_replace = {"bc_masks": jnp.ones_like(online_batch["masks"])}
            if FLAGS.use_rnd_online:
                online_replace["rewards"] = online_batch["rewards"] + rnd.get_reward(
                    frozen_dict.freeze(online_batch)
                )
            online_batch = online_batch.copy(add_or_replace=online_replace)

            if FLAGS.offline_ratio > 0:
                offline_batch = next(ds_iterator)

                offline_replace = {
                    "bc_masks": jnp.ones_like(offline_batch["masks"]),
                    "rewards": offline_batch["rewards"],
                }
                if FLAGS.offline_relabel_type in ["pred", "min"]:
                    offline_replace["masks"] = rm.get_mask(offline_batch)
                if FLAGS.offline_relabel_type == "min":
                    offline_replace["rewards"] = (
                        offline_batch["rewards"].at[:].set(ds_minr)
                    )
                if FLAGS.offline_relabel_type == "pred":
                    offline_replace["rewards"] = rm.get_reward(offline_batch)

                if FLAGS.use_rnd_offline:
                    offline_replace["rewards"] = offline_replace[
                        "rewards"
                    ] + rnd.get_reward(frozen_dict.freeze(offline_batch))

                offline_batch = offline_batch.copy(add_or_replace=offline_replace)
                batch = combine(offline_batch, online_batch)
            else:
                batch = online_batch

            # update the main agent
            agent, update_info = agent.update(batch, FLAGS.utd_ratio)
            logging_info.update(add_prefix("agent/", update_info))

        if i % FLAGS.log_interval == 0:
            wandb.log({"env_step": i}, step=record_step)
            for k, v in logging_info.items():
                wandb.log({k: v}, step=record_step)

            # visualize rewards rm and rnd rewards along a successful offline trajectory
            traj = ds.load_successful_traj()

            rnd_reward = []
            rm_reward = []
            for tran in traj:
                if rnd is not None:
                    rnd_reward.append(rnd.get_reward(frozen_dict.freeze(tran)).item())
                if rm is not None:
                    rm_reward.append(rm.get_reward(frozen_dict.freeze(tran)).item())

            if rm is not None:
                plt.clf()
                plt.plot(rm_reward, label="rm")
                plt.xlabel("step in offline trajectory")
                plt.ylabel("reward")
                plt.legend()
                plt.title("predicted rewards in successful offline trajectory")

                wandb.log(
                    {"training/offline_success_traj_rewards_rm": plt}, step=record_step
                )

            if rnd is not None:
                plt.clf()
                plt.plot(rnd_reward, label="rnd")
                plt.xlabel("step in offline trajectory")
                plt.ylabel("reward")
                plt.legend()
                plt.title("predicted rewards in successful offline trajectory")

                wandb.log(
                    {"training/offline_success_traj_rewards_rnd": plt}, step=record_step
                )

        if i % FLAGS.eval_interval == 0:
            eval_info, _ = evaluate(
                agent,
                eval_env,
                num_episodes=FLAGS.eval_episodes,
            )
            for k, v in eval_info.items():
                wandb.log({f"evaluation/{k}": v}, step=record_step)

            if FLAGS.bc_pretrain_rollin > 0.0:
                bc_eval_info, _ = evaluate(
                    bc_policy,
                    eval_env,
                    num_episodes=FLAGS.eval_episodes,
                )
                for k, v in bc_eval_info.items():
                    wandb.log({f"bc-evaluation/{k}": v}, step=record_step)

            if FLAGS.save_dir is not None:
                if FLAGS.checkpoint_model:
                    try:
                        checkpoints.save_checkpoint(
                            chkpt_dir,
                            agent,
                            step=i,
                            keep=100,
                            overwrite=True,
                            prefix="agent_checkpoint_",
                        )
                        checkpoints.save_checkpoint(
                            chkpt_dir,
                            rm,
                            step=i,
                            keep=100,
                            overwrite=True,
                            prefix="rm_checkpoint_",
                        )
                        if rnd is not None:
                            checkpoints.save_checkpoint(
                                chkpt_dir,
                                rnd,
                                step=i,
                                keep=100,
                                overwrite=True,
                                prefix="rnd_checkpoint_",
                            )
                    except:
                        print("Could not save model checkpoint.")

                if FLAGS.checkpoint_buffer:
                    try:
                        with open(os.path.join(buffer_dir, f"buffer"), "wb") as f:
                            pickle.dump(replay_buffer, f, pickle.HIGHEST_PROTOCOL)
                    except:
                        print("Could not save agent buffer.")


if __name__ == "__main__":
    app.run(main)
