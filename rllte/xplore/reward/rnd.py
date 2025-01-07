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


from typing import Dict, Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium.vector import VectorEnv
from torch.utils.data import DataLoader, TensorDataset

from rllte.common.prototype import BaseReward
from .model import ObservationEncoder


class RND(BaseReward):
    """Exploration by Random Network Distillation (RND).
        See paper: https://arxiv.org/pdf/1810.12894.pdf

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
        batch_size (int): The batch size for training.
        update_proportion (float): The proportion of the training data used for updating the forward dynamics models.
        encoder_model (str): The network architecture of the encoder from ['mnih', 'pathak'].
        weight_init (str): The weight initialization method from ['default', 'orthogonal', 'kaiming he'].

    Returns:
        Instance of RND.
    """

    def __init__(
        self,
        envs: VectorEnv,
        device: str = "cpu",
        beta: float = 1.0,
        kappa: float = 0.0,
        gamma: Optional[float] = None,
        rwd_norm_type: str = "rms",
        obs_norm_type: str = "rms",
        latent_dim: int = 128,
        lr: float = 0.001,
        batch_size: int = 256,
        update_proportion: float = 1.0,
        encoder_model: str = "mnih",
        weight_init: str = "orthogonal",
    ) -> None:
        super().__init__(envs, device, beta, kappa, gamma, rwd_norm_type, obs_norm_type)
        # build the predictor and target networks
        self.predictor = ObservationEncoder(
            obs_shape=self.obs_shape,
            latent_dim=latent_dim,
            encoder_model=encoder_model,
            weight_init=weight_init,
        ).to(self.device)
        self.target = ObservationEncoder(
            obs_shape=self.obs_shape,
            latent_dim=latent_dim,
            encoder_model=encoder_model,
            weight_init=weight_init,
        ).to(self.device)

        # freeze the randomly initialized target network parameters
        for p in self.target.parameters():
            p.requires_grad = False
        # set the optimizer
        self.opt = th.optim.Adam(self.predictor.parameters(), lr=lr)
        # set the parameters
        self.batch_size = batch_size
        self.update_proportion = update_proportion

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
        super().compute(samples)
        # get the number of steps and environments
        (n_steps, n_envs) = samples.get("next_observations").size()[:2]
        # get the next observations
        next_obs_tensor = samples.get("next_observations").to(self.device)
        # normalize the observations
        next_obs_tensor = self.normalize(next_obs_tensor)
        # compute the intrinsic rewards
        intrinsic_rewards = th.zeros(size=(n_steps, n_envs)).to(self.device)
        with th.no_grad():
            # get source and target features
            src_feats = self.predictor(next_obs_tensor.view(-1, *self.obs_shape))
            tgt_feats = self.target(next_obs_tensor.view(-1, *self.obs_shape))
            # compute the distance
            dist = F.mse_loss(src_feats, tgt_feats, reduction="none").mean(dim=1)
            intrinsic_rewards = dist.view(n_steps, n_envs)

        # update the reward module
        if sync:
            self.update(samples)

        # scale the intrinsic rewards
        return self.scale(intrinsic_rewards)

    def update(self, samples: Dict[str, th.Tensor]) -> None:
        """Update the reward module if necessary.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples same as the `compute` function.

        Returns:
            None.
        """
        # get the observations
        obs_tensor = (
            samples.get("observations").to(self.device).view(-1, *self.obs_shape)
        )
        # normalize the observations
        obs_tensor = self.normalize(obs_tensor)
        # create the dataset and loader
        dataset = TensorDataset(obs_tensor)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        avg_loss = []
        # update the predictor
        for _idx, batch_data in enumerate(loader):
            # get the batch data
            obs = batch_data[0]
            # zero the gradients
            self.opt.zero_grad()
            # get the source and target features
            src_feats = self.predictor(obs)
            with th.no_grad():
                tgt_feats = self.target(obs)

            # compute the loss
            loss = F.mse_loss(src_feats, tgt_feats, reduction="none").mean(dim=-1)
            # use a random mask to select a subset of the training data
            mask = th.rand(len(loss), device=self.device)
            mask = (mask < self.update_proportion).type(th.FloatTensor).to(self.device)
            # get the masked loss
            loss = (loss * mask).sum() / th.max(
                mask.sum(), th.tensor([1], device=self.device, dtype=th.float32)
            )
            # backward and update
            loss.backward()
            self.opt.step()
            avg_loss.append(loss.item())

        try:
            self.metrics["loss"].append([self.global_step, np.mean(avg_loss)])
        except:
            pass