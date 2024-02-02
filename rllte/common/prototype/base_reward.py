# =============================================================================
# MIT License

# Copyright (c) 2024 Reinforcement Learning Evolution Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# =============================================================================


from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional

import gymnasium as gym
import numpy as np
import torch as th

from rllte.common.preprocessing import process_action_space, process_observation_space
from rllte.common.utils import TorchRunningMeanStd

class BaseReward(ABC):
    """Base class of reward module.

    Args:
        observation_space (gym.Space): The observation space of environment.
        action_space (gym.Space): The action space of environment.
        n_envs (int): The number of parallel environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate.
        use_rms (bool): Use running mean and std for normalization.
        obs_rms (bool): Use running mean and std for observation normalization.
        is_sync (bool): Whether to synchronize the reward computation with environment step.

    Returns:
        Instance of the base reward module.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        n_envs: int,
        device: str = "cpu",
        beta: float = 1.0,
        kappa: float = 0.0,
        use_rms: bool = True,
        obs_rms: bool = False,
        is_sync: bool = False
    ) -> None:
        # get environment information
        self.obs_shape: Tuple = process_observation_space(observation_space)  # type: ignore
        self.action_shape, self.action_dim, self.policy_action_dim, self.action_type \
            = process_action_space(action_space)
        self.n_envs = n_envs

        # set device and parameters
        self.device = th.device(device)
        self.beta = beta
        self.kappa = kappa
        self.use_rms = use_rms
        self.obs_rms = obs_rms
        self.is_sync = is_sync
        self.global_step = 0

        # build the running mean and std for normalization
        self.rms = TorchRunningMeanStd() if self.use_rms else None
        self.obs_norm = TorchRunningMeanStd(shape=self.obs_shape) if self.obs_rms else None
        
    @property
    def weight(self) -> float:
        """Get the weighting coefficient of the intrinsic rewards.
        """
        return self.beta * np.power(1.0 - self.kappa, self.global_step)
    
    def scale(self, rewards: th.Tensor) -> th.Tensor:
        """Scale the intrinsic rewards.

        Args:
            rewards (th.Tensor): The intrinsic rewards with shape (n_steps, n_envs).
        
        Returns:
            The scaled intrinsic rewards.
        """
        if self.use_rms:
            self.rms.update(rewards)
            return rewards / self.rms.std * self.weight
        else:
            return rewards * self.weight
        
    def normalize(self, x: th.Tensor) -> th.Tensor:
        if self.obs_rms:
            x = (
                (
                    (x - self.obs_norm.mean.to(self.device))
                )
                    / th.sqrt(self.obs_norm.var.to(self.device))
            ).clip(-5, 5)
        else:
            x = x / 255.0 if len(self.obs_shape) > 2 else x
        return x
    
    @abstractmethod
    def watch(self,
              observations: th.Tensor,
              actions: th.Tensor,
              rewards: th.Tensor,
              terminateds: th.Tensor,
              truncateds: th.Tensor,
              next_observations: th.Tensor
              ) -> Optional[Dict[str, th.Tensor]]:
        """Watch the interaction processes and obtain necessary elements for reward computation.

        Args:
            observations (th.Tensor): The observations data with shape (n_steps, n_envs, *obs_shape).
            actions (th.Tensor): The actions data with shape (n_steps, n_envs, *action_shape).
            rewards (th.Tensor): The rewards data with shape (n_steps, n_envs).
            terminateds (th.Tensor): Termination signals with shape (n_steps, n_envs).
            truncateds (th.Tensor): Truncation signals with shape (n_steps, n_envs).
            next_observations (th.Tensor): The next observations data with shape (n_steps, n_envs, *obs_shape).

        Returns:
            None.
        """
    
    @abstractmethod
    def compute(self, samples: Dict[str, th.Tensor]) -> th.Tensor:
        """Compute the rewards for current samples.

        Args:
            samples (Dict): The collected samples. A python dict consists of multiple tensors, whose keys are
            'observations', 'actions', 'rewards', 'terminateds', 'truncateds', 'next_observations'. 
            The data shape of each tensor is consistent with the `watch` function.

        Returns:
            The intrinsic rewards.
        """
        for key in ['observations', 'actions', 'rewards', 'terminateds', 'truncateds', 'next_observations']:
            assert key in samples.keys(), f"Key {key} is not in samples."

        # update obs RMS if necessary
        if self.obs_rms:
            self.obs_norm.update(samples['observations'].reshape(-1, *self.obs_shape).cpu())

        self.global_step += 1
        

    @abstractmethod
    def update(self, samples: Dict[str, th.Tensor]) -> None:
        """Update the reward module if necessary.

        Args:
            samples (Dict): The collected samples same as the `compute` function.
                The `update` function will be invoked after the `compute` function.

        Returns:
            None.
        """
