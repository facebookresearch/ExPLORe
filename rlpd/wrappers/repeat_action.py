"""
Modified from https://github.com/ikostrikov/rlpd/blob/main/rlpd/wrappers/repeat_action.py

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

import gym
import numpy as np


class RepeatAction(gym.Wrapper):
    def __init__(self, env, action_repeat=4):
        super().__init__(env)
        self._action_repeat = action_repeat

    def step(self, action: np.ndarray):
        total_reward = 0.0
        done = None
        combined_info = {}

        for _ in range(self._action_repeat):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            combined_info.update(info)
            if done:
                break

        return obs, total_reward, done, combined_info
