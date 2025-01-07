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
from .model import ForwardDynamicsModel, ObservationEncoder


class Disagreement(BaseReward):
    """Self-Supervised Exploration via Disagreement (Disagreement).
        See paper: https://arxiv.org/pdf/1906.04161.pdf

    Args:
        envs (VectorEnv): The vectorized environments.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        beta (float): The initial weighting coefficient of the intrinsic rewards.
        kappa (float): The decay rate of the weighting coefficient.
        gamma (Optional[float]): Intrinsic reward discount rate, default is `None`.
        rwd_norm_type (str): Normalization type for intrinsic rewards from ['rms', 'minmax', 'none'].
        obs_norm_type (str): Normalization type for observations data from ['rms', 'none'].

        ensemble_size (int): The number of forward dynamics models in the ensemble.
        latent_dim (int): The dimension of encoding vectors.
        lr (float): The learning rate.
        batch_size (int): The batch size for training.
        update_proportion (float): The proportion of the training data used for updating the forward dynamics models.
        encoder_model (str): The network architecture of the encoder from ['mnih', 'pathak'].
        weight_init (str): The weight initialization method from ['default', 'orthogonal', 'kaiming he'].

    Returns:
        Instance of Disagreement.
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
        ensemble_size: int = 4,
        latent_dim: int = 128,
        lr: float = 0.001,
        batch_size: int = 256,
        update_proportion: float = 1.0,
        encoder_model: str = "mnih",
        weight_init: str = "orthogonal",
    ) -> None:
        super().__init__(envs, device, beta, kappa, gamma, rwd_norm_type, obs_norm_type)

        self.random_encoder = ObservationEncoder(
            obs_shape=self.obs_shape,
            latent_dim=latent_dim,
            encoder_model=encoder_model,
            weight_init=weight_init,
        ).to(self.device)

        # freeze the randomly initialized target network parameters
        for p in self.random_encoder.parameters():
            p.requires_grad = False

        # build the ensemble of forward dynamics models
        self.ensemble_size = ensemble_size
        self.ensemble = [
            ForwardDynamicsModel(
                latent_dim=latent_dim,
                action_dim=self.policy_action_dim,
                encoder_model=encoder_model,
            ).to(self.device)
            for _ in range(self.ensemble_size)
        ]
        self.opt = [
            th.optim.Adam(self.ensemble[i].parameters(), lr=lr)
            for i in range(self.ensemble_size)
        ]
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
        (n_steps, n_envs) = samples.get("observations").size()[:2]
        # get the observations
        obs_tensor = (
            samples.get("observations").to(self.device).view(-1, *self.obs_shape)
        )
        # normalize the observations
        obs_tensor = self.normalize(obs_tensor)
        actions_tensor = (
            samples.get("actions").to(self.device).view(-1, *self.action_shape)
        )
        # apply one-hot encoding if the action type is discrete
        if self.action_type == "Discrete":
            actions_tensor = F.one_hot(
                actions_tensor.long(), self.policy_action_dim
            ).float()
        # compute the intrinsic rewards
        with th.no_grad():
            random_feats = self.random_encoder(obs_tensor.view(-1, *self.obs_shape))
            preds = []
            for i in range(self.ensemble_size):
                next_obs_hat = self.ensemble[i](random_feats, actions_tensor)
                preds.append(next_obs_hat)
            preds = th.stack(preds, dim=0)
            intrinsic_rewards = th.var(preds, dim=0).mean(dim=-1).view(n_steps, n_envs)

        # update the reward module
        if sync:
            self.update(samples)

        # return the scaled intrinsic rewards
        return self.scale(intrinsic_rewards)

    def update(self, samples: Dict[str, th.Tensor]) -> None:
        """Update the reward module if necessary.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples same as the `compute` function.

        Returns:
            None.
        """
        # get the number of steps and environments
        (n_steps, n_envs) = samples.get("next_observations").size()[:2]
        # get the observations and actions
        obs_tensor = (
            samples.get("observations").to(self.device).view(-1, *self.obs_shape)
        )
        next_obs_tensor = (
            samples.get("next_observations").to(self.device).view(-1, *self.obs_shape)
        )
        # normalize the observations
        obs_tensor = self.normalize(obs_tensor)
        next_obs_tensor = self.normalize(next_obs_tensor)
        # apply one-hot encoding if the action type is discrete
        if self.action_type == "Discrete":
            actions_tensor = samples.get("actions").view(n_steps * n_envs)
            actions_tensor = F.one_hot(
                actions_tensor.long(), self.policy_action_dim
            ).float()
        else:
            actions_tensor = samples.get("actions").view(n_steps * n_envs, -1)
        # build the dataset and dataloader
        dataset = TensorDataset(obs_tensor, actions_tensor, next_obs_tensor)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)
        # update the ensemble of forward dynamics models
        avg_loss = []
        for _idx, batch_data in enumerate(loader):
            ensemble_idx = _idx % self.ensemble_size
            # get the batch data
            obs, actions, next_obs = batch_data
            obs, actions, next_obs = (
                obs.to(self.device),
                actions.to(self.device),
                next_obs.to(self.device),
            )
            # zero the gradients
            self.opt[ensemble_idx].zero_grad()
            # get the encoded observations and next observations
            with th.no_grad():
                encoded_obs = self.random_encoder(obs)
                encoded_next_obs = self.random_encoder(next_obs)
            # compute the predicted next observations
            pred_next_obs = self.ensemble[ensemble_idx](encoded_obs, actions)
            # compute the forward dynamics loss
            fm_loss = F.mse_loss(
                pred_next_obs, encoded_next_obs, reduction="none"
            ).mean(-1)
            # use a random mask to select a subset of the training data
            mask = th.rand(len(fm_loss), device=self.device)
            mask = (mask < self.update_proportion).type(th.FloatTensor).to(self.device)
            # get the masked loss
            fm_loss = (fm_loss * mask).sum() / th.max(
                mask.sum(), th.tensor([1], device=self.device, dtype=th.float32)
            )
            # backward and update
            fm_loss.backward()
            self.opt[ensemble_idx].step()
            # record the loss
            avg_loss.append(fm_loss.item())
        # save the loss
        self.metrics["loss"].append([self.global_step, np.mean(avg_loss)])