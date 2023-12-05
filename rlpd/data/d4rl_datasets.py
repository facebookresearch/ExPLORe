"""
Modified from https://github.com/ikostrikov/rlpd/blob/main/rlpd/data/d4rl_datasets.py

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


import d4rl
import gym
import numpy as np

from rlpd.data.dataset import Dataset
import copy


def filter_antmaze(tran, env, np_rng, mode="all"):
    if "large" in env:
        right_cutoff = 20
        bottom_cutoff = 10
        block_cutoff = 5
    elif "medium" in env:
        right_cutoff = 10
        bottom_cutoff = 10
        block_cutoff = 5
    else:
        raise NotImplementedError

    x, y = tran["observations"][:2]
    delta = (tran["next_observations"] - tran["observations"])[:2]

    # observation based filters
    if mode == "all":
        return True
    elif mode == "right":
        return x < right_cutoff
    elif mode == "bottom":
        return y < bottom_cutoff
    elif mode == "no_corner":
        return x < right_cutoff or y < bottom_cutoff
    elif mode == "stripes":
        return (x // block_cutoff) % 2 == 0
    elif mode == "checkers":
        return (x // block_cutoff) % 2 == (y // block_cutoff) % 2
    elif mode == "subopt":
        assert "large" in env
        return (
            (x < 5 and y < 10)
            or (x < 25 and 5 < y < 10)
            or (15 < x and y < 5)
            or (30 < x and y < 20)
            or (25 < x and 15 < y)
        )
    elif mode == "10perc":
        return np_rng.random() < 0.1
    elif mode == "1perc":
        return np_rng.random() < 0.01
    elif mode == "01perc":
        return np_rng.random() < 0.001
    elif mode == "001perc":
        return np_rng.random() < 0.0001

    # action based filters
    elif mode == "southwest":
        return np.dot(delta, np.ones(delta.shape)) < 0
    elif mode == "northeast":
        return np.dot(delta, np.ones(delta.shape)) > 0
    elif mode == "southwest-10perc":
        return np.dot(delta, np.ones(delta.shape)) < 0 and np_rng.random() < 0.1
    elif mode == "southwest-1perc":
        return np.dot(delta, np.ones(delta.shape)) < 0 and np_rng.random() < 0.01
    elif mode == "southwest-01perc":
        return np.dot(delta, np.ones(delta.shape)) < 0 and np_rng.random() < 0.001
    elif mode == "southwest-001perc":
        return np.dot(delta, np.ones(delta.shape)) < 0 and np_rng.random() < 0.0001
    else:
        raise NotImplementedError


class D4RLDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5):
        dataset_dict = d4rl.qlearning_dataset(env)

        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        dones = np.full_like(dataset_dict["rewards"], False, dtype=bool)

        for i in range(len(dones) - 1):
            if (
                np.linalg.norm(
                    dataset_dict["observations"][i + 1]
                    - dataset_dict["next_observations"][i]
                )
                > 1e-6
                or dataset_dict["terminals"][i] == 1.0
            ):
                dones[i] = True

        dones[-1] = True

        dataset_dict["masks"] = 1.0 - dataset_dict["terminals"]
        del dataset_dict["terminals"]

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        dataset_dict["dones"] = dones

        super().__init__(dataset_dict)
