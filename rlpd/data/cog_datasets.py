"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import os

import numpy as np
import gym

from rlpd.data import MemoryEfficientReplayBuffer

def dict_to_list(D):
    # https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
    return [dict(zip(D, t)) for t in zip(*D.values())]

class COGDataset(MemoryEfficientReplayBuffer):
    def __init__(
        self,
        env: gym.Env,
        dataset_path: str,
        capacity: int = 500_000,
        subsample_ratio: float = 1.0,
        pixel_keys: tuple = ("pixels",),
        np_rng = None,
        load_successes: bool = True,
    ):
        self.np_rng = np_rng
        super().__init__(
            env.observation_space,
            env.action_space,
            capacity=capacity,
            pixel_keys=pixel_keys
        )
        self.successful_offline_prior_trajs = []
        self.successful_offline_task_trajs = []
        
        self._load_data_from_dir(dataset_path, subsample_ratio)
        
        self.load_successes = load_successes
        if self.load_successes:
            self._load_successful_traj(dataset_path)

    def load_successful_traj(self):
        assert self.load_successes, "did not load successful trajectories upon making this dataset"
        prior_idx = self.np_rng.integers(len(self.successful_offline_prior_trajs))
        task_idx = self.np_rng.integers(len(self.successful_offline_task_trajs))
        prior_traj = self.successful_offline_prior_trajs[prior_idx]
        task_traj = self.successful_offline_task_trajs[task_idx]
        return prior_traj + task_traj
    
    def _load_data_from_dir(self, dataset_path, subsample_ratio=1.0):
        print("subsample ratio:", subsample_ratio * subsample_ratio)  # sub-sampled twice
        for f in os.listdir(dataset_path):
            full_path = os.path.join(dataset_path, f)
            if f.endswith('.npy'):
                print("*"*20, "\nloading data from:", full_path)
                data = np.load(full_path, allow_pickle=True)
                print("prior subsampling # trajs:", len(data))
                data = self._subsample_data(data, subsample_ratio)
                self._load_data(data, subsample_ratio)
                print("post subsampling # trajs:", len(self))
    
    def _subsample_data(self, data, r=1.0):
        assert 0 <= r <= 1
        n = len(data)
        idxs = self.np_rng.choice(n, size=int(n*r), replace=False)
        return data[idxs]

    def _load_data(self, data, subsample_ratio=1.0):
        cutoff = int(len(data) * subsample_ratio)
        for i, traj in enumerate(data):
            if i > cutoff:
                break
            trans = dict_to_list(traj)
            for tran in trans:
                data_dict = self._make_data_dict(tran)
                self.insert(data_dict)
    
    def _load_successful_traj(self, dataset_path):
        # load successful offline trajectories for visualizations / evaluation
        prior_data = np.load(os.path.join(dataset_path, 'successful', 'prior_success.npy'), allow_pickle=True)
        task_data = np.load(os.path.join(dataset_path, 'successful', 'task_success.npy'), allow_pickle=True)

        for traj in prior_data:
            trans = dict_to_list(traj)
            trans = [self._make_data_dict(tran) for tran in trans]
            self.successful_offline_prior_trajs.append(trans)

        for traj in task_data:
            trans = dict_to_list(traj)
            trans = [self._make_data_dict(tran) for tran in trans]
            self.successful_offline_task_trajs.append(trans)

    def _make_data_dict(self, tran):
        return dict(
            observations={"pixels": np.array(tran["observations"]["image"])[..., None]},
            actions=np.array(tran["actions"]),
            next_observations={"pixels": np.array(tran["next_observations"]["image"])[..., None]},
            rewards=np.array(tran["rewards"]),
            masks=1-np.array(tran["terminals"], dtype=float),
            dones=np.array(tran["agent_infos"]["done"])
        )
