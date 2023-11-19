"""
Modified from https://github.com/ikostrikov/rlpd/blob/main/rlpd/wrappers/pixels.py

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

from typing import Optional, Tuple

import gym
from gym.wrappers.pixel_observation import PixelObservationWrapper

from rlpd.wrappers.frame_stack import FrameStack
from rlpd.wrappers.repeat_action import RepeatAction
from rlpd.wrappers.universal_seed import UniversalSeed


def wrap_pixels(
    env: gym.Env,
    action_repeat: int,
    image_size: int = 84,
    num_stack: Optional[int] = 3,
    camera_id: int = 0,
    pixel_keys: Tuple[str, ...] = ("pixels",),
) -> gym.Env:
    if action_repeat > 1:
        env = RepeatAction(env, action_repeat)

    env = UniversalSeed(env)
    env = gym.wrappers.RescaleAction(env, -1, 1)

    env = PixelObservationWrapper(
        env,
        pixels_only=True,
        render_kwargs={
            "pixels": {
                "height": image_size,
                "width": image_size,
                "camera_id": camera_id,
            }
        },
        pixel_keys=pixel_keys,
    )

    if num_stack is not None:
        env = FrameStack(env, num_stack=num_stack)

    env = gym.wrappers.ClipAction(env)

    return env, pixel_keys
