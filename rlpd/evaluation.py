"""
Modified from https://github.com/ikostrikov/rlpd/blob/main/rlpd/evaluation.py

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

from typing import Dict

import gym
import numpy as np


def evaluate(agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:

    trajs = []
    cum_returns = []
    cum_lengths = []
    for i in range(num_episodes):
        observation, done = env.reset(), False
        traj = [observation]
        cum_return = 0
        cum_length = 0
        while not done:
            action = agent.eval_actions(observation)
            observation, reward, done, _ = env.step(action)
            cum_return += reward
            cum_length += 1
            traj.append(observation)
        cum_returns.append(cum_return)
        cum_lengths.append(cum_length)
        trajs.append({"observation": np.stack(traj, axis=0)})
    return {"return": np.mean(cum_returns), "length": np.mean(cum_lengths)}, trajs
