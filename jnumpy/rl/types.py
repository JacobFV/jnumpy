# Copyright (c) 2021 Jacob F. Valdez. Released under the MIT license.
"""Type abstractions for RL.
This module is not imported at a lower scope in `jnumpy.rl`. Instead, 
classes in `jnumpy.rl.types` are directly imported into `jnumpy.rl` namespace."""


from __future__ import annotations

import numpy as np
from typing import List


class Step:
    """Single step."""
    
    def __init__(self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: bool,
        info: any
    ):
        self.obs = obs
        self.next_obs = next_obs
        self.action = action
        self.reward = reward
        self.done = done
        self.info = info

    @staticmethod
    def unbatch(step: Step) -> List[Step]:
        return [
            Step(
                obs=step.obs[i:i+1],
                next_obs=step.next_obs[i:i+1],
                action=step.action[i:i+1],
                reward=step.reward[i:i+1],
                done=step.done,
                info=step.info[i:i+1],
            )
            for i in range(step.obs.shape[0])
        ]

    @staticmethod
    def batch(steps: List[BatchStep]) -> BatchStep:
        return BatchStep(
            obs=np.concatenate([step.obs for step in steps], axis=0),
            next_obs=np.concatenate([step.next_obs for step in steps], axis=0),
            action=np.concatenate([step.action for step in steps], axis=0),
            reward=np.concatenate([step.reward for step in steps], axis=0),
            done=any(step.done for step in steps),
            info=[step.info for step in steps])

    @staticmethod
    def from_no_batch_axis(step: NoBatchStep) -> Step:
        return Step(
            obs=step.obs[None, ...],
            next_obs=step.next_obs[None, ...],
            action=step.action[None, ...],
            reward=step.reward[None, ...],
            done=step.done,
            info=[step.info]
        )

# dimensional type hinting
BatchStep = Step
NoBatchStep = Step

Traj = List[BatchStep]