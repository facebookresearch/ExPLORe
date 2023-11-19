"""
Modified from https://github.com/ikostrikov/rlpd/blob/main/rlpd/wrappers/frame_stack.py

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

import collections

import gym
import numpy as np
from gym.spaces import Box


class FrameStack(gym.Wrapper):
    def __init__(self, env, num_stack: int, stacking_key: str = "pixels"):
        super().__init__(env)
        self._num_stack = num_stack
        self._stacking_key = stacking_key

        assert stacking_key in self.observation_space.spaces
        pixel_obs_spaces = self.observation_space.spaces[stacking_key]

        self._env_dim = pixel_obs_spaces.shape[-1]

        low = np.repeat(pixel_obs_spaces.low[..., np.newaxis], num_stack, axis=-1)
        high = np.repeat(pixel_obs_spaces.high[..., np.newaxis], num_stack, axis=-1)
        new_pixel_obs_spaces = Box(low=low, high=high, dtype=pixel_obs_spaces.dtype)
        self.observation_space.spaces[stacking_key] = new_pixel_obs_spaces

        self._frames = collections.deque(maxlen=num_stack)

    def reset(self):
        obs = self.env.reset()
        for i in range(self._num_stack):
            self._frames.append(obs[self._stacking_key])
        obs[self._stacking_key] = self.frames
        return obs

    @property
    def frames(self):
        return np.stack(self._frames, axis=-1)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs[self._stacking_key])
        obs[self._stacking_key] = self.frames
        return obs, reward, done, info
