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


from typing import Dict

import torch as th
from gymnasium.vector import VectorEnv

from .fabric import Fabric
from .pseudo_counts import PseudoCounts
from .rnd import RND


class NGU(Fabric):
    """Never Give Up: Learning Directed Exploration Strategies (NGU).
        See paper: https://arxiv.org/pdf/2002.06038

    Args:
        envs (VectorEnv): The vectorized environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate of the weighting coefficient.
        gamma (Optional[float]): Intrinsic reward discount rate, default is `None`.
        rwd_norm_type (str): Normalization type for intrinsic rewards from ['rms', 'minmax', 'none'].
        obs_norm_type (str): Normalization type for observations data from ['rms', 'none'].

        latent_dim (int): The dimension of encoding vectors.
        lr (float): The learning rate.
        batch_size (int): The batch size for update.
        k (int): Number of neighbors.
        kernel_cluster_distance (float): The kernel cluster distance.
        kernel_epsilon (float): The kernel constant.
        c (float): The pseudo-counts constant.
        sm (float): The kernel maximum similarity.
        mrs (float): The maximum reward scaling.
        update_proportion (float): The proportion of the training data used for updating the forward dynamics models.
        encoder_model (str): The network architecture of the encoder from ['mnih', 'pathak'].
        weight_init (str): The weight initialization method from ['default', 'orthogonal', 'kaiming he'].

    Returns:
        Instance of NGU.
    """

    def __init__(
        self,
        envs: VectorEnv,
        device: str = "cpu",
        beta: float = 1.0,
        kappa: float = 0.0,
        gamma: float = None,
        rwd_norm_type: str = "rms",
        obs_norm_type: str = "rms",
        latent_dim: int = 32,
        lr: float = 0.001,
        batch_size: int = 256,
        k: int = 10,
        kernel_cluster_distance: float = 0.008,
        kernel_epsilon: float = 0.0001,
        c: float = 0.001,
        sm: float = 8.0,
        mrs: float = 5.0,
        update_proportion: float = 1.0,
        encoder_model: str = "mnih",
        weight_init: str = "default",
    ) -> None:
        # build the rnd and pseudo-counts modules
        rnd = RND(
            envs=envs,
            device=device,
            beta=beta,
            kappa=kappa,
            rwd_norm_type=rwd_norm_type,
            obs_norm_type=obs_norm_type,
            gamma=gamma,
            latent_dim=latent_dim,
            lr=lr,
            batch_size=batch_size,
            update_proportion=update_proportion,
            encoder_model=encoder_model,
            weight_init=weight_init,
        )

        pseudo_counts = PseudoCounts(
            envs=envs,
            device=device,
            beta=beta,
            kappa=kappa,
            rwd_norm_type=rwd_norm_type,
            obs_norm_type=obs_norm_type,
            gamma=gamma,
            latent_dim=latent_dim,
            lr=lr,
            batch_size=batch_size,
            k=k,
            kernel_cluster_distance=kernel_cluster_distance,
            kernel_epsilon=kernel_epsilon,
            c=c,
            sm=sm,
            update_proportion=update_proportion,
            encoder_model=encoder_model,
            weight_init=weight_init,
        )

        self.mrs = mrs
        
        super().__init__(*[rnd, pseudo_counts])

    def compute(self, samples: Dict[str, th.Tensor], sync: bool = True) -> th.Tensor:
        """Compute the rewards for current samples.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples. A python dict consists of multiple tensors,
                whose keys are ['observations', 'actions', 'rewards', 'terminateds', 'truncateds', 'next_observations'].
                For example, the data shape of 'observations' is (n_steps, n_envs, *obs_shape).
            sync (bool): Whether to update the reward module after the `compute` function, default is `True`.

        Returns:
            The intrinsic rewards.
        """
        # get the number of steps and environments
        lifelong_rewards, episodic_rewards = super().compute(samples, sync)

        # compute the intrinsic rewards
        lifelong_rewards = 1.0 + lifelong_rewards
        lifelong_rewards = th.maximum(lifelong_rewards, th.ones_like(lifelong_rewards))
        lifelong_rewards = th.minimum(
            lifelong_rewards, th.ones_like(lifelong_rewards) * self.mrs
        )

        return lifelong_rewards * episodic_rewards
