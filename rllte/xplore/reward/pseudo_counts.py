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


from typing import Dict, List, Optional

import numpy as np
import torch as th
from gymnasium.vector import VectorEnv
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from rllte.common.prototype import BaseReward
from rllte.common.utils import TorchRunningMeanStd
from .model import InverseDynamicsEncoder


class PseudoCounts(BaseReward):
    """Pseudo-counts based on "Never Give Up: Learning Directed Exploration Strategies (NGU)".
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
        update_proportion (float): The proportion of the training data used for updating the forward dynamics models.
        encoder_model (str): The network architecture of the encoder from ['mnih', 'pathak'].
        weight_init (str): The weight initialization method from ['default', 'orthogonal'].

    Returns:
        Instance of PseudoCounts.
    """

    def __init__(
        self,
        envs: VectorEnv,
        device: str = "cpu",
        beta: float = 1.0,
        kappa: float = 0.0,
        gamma: float = None,
        rwd_norm_type: str = "rms",
        obs_norm_type: str = "none",
        latent_dim: int = 32,
        lr: float = 0.001,
        batch_size: int = 256,
        k: int = 10,
        kernel_cluster_distance: float = 0.008,
        kernel_epsilon: float = 0.0001,
        c: float = 0.001,
        sm: float = 8.0,
        update_proportion: float = 1.0,
        encoder_model: str = "mnih",
        weight_init: str = "orthogonal",
    ) -> None:
        super().__init__(envs, device, beta, kappa, gamma, rwd_norm_type, obs_norm_type)
        # set parameters
        self.lr = lr
        self.batch_size = batch_size
        self.k = k
        self.kernel_cluster_distance = kernel_cluster_distance
        self.kernel_epsilon = kernel_epsilon
        self.c = c
        self.sm = sm
        self.update_proportion = update_proportion

        # build the episodic memory
        self.episodic_memory = [[] for _ in range(self.n_envs)]
        self.n_eps = [[] for _ in range(self.n_envs)]

        # build the encoder
        self.encoder = InverseDynamicsEncoder(
            obs_shape=self.obs_shape,
            action_dim=self.policy_action_dim,
            latent_dim=latent_dim,
            encoder_model=encoder_model,
            weight_init=weight_init,
        ).to(self.device)
        # set the optimizer and loss function
        self.opt = th.optim.Adam(self.encoder.parameters(), lr=lr)
        if self.action_type == "Discrete":
            self.loss = nn.CrossEntropyLoss(reduction="none")
        else:
            self.loss = nn.MSELoss(reduction="none")
        # rms for the intrinsic rewards
        self.dist_rms = TorchRunningMeanStd(shape=(1,), device=self.device)
        self.squared_distances = []
        # temporary buffers for intrinsic rewards and observations
        self.irs_buffer = []
        self.obs_buffer = []

    def watch(
        self,
        observations: th.Tensor,
        actions: th.Tensor,
        rewards: th.Tensor,
        terminateds: th.Tensor,
        truncateds: th.Tensor,
        next_observations: th.Tensor,
    ) -> Optional[Dict[str, th.Tensor]]:
        """Watch the interaction processes and obtain necessary elements for reward computation.

        Args:
            observations (th.Tensor): Observations data with shape (n_envs, *obs_shape).
            actions (th.Tensor): Actions data with shape (n_envs, *action_shape).
            rewards (th.Tensor): Extrinsic rewards data with shape (n_envs).
            terminateds (th.Tensor): Termination signals with shape (n_envs).
            truncateds (th.Tensor): Truncation signals with shape (n_envs).
            next_observations (th.Tensor): Next observations data with shape (n_envs, *obs_shape).

        Returns:
            Feedbacks for the current samples.
        """
        with th.no_grad():
            # data shape of embeddings: (n_envs, latent_dim)
            observations = self.normalize(observations)
            embeddings = self.encoder.encode(observations)
            for i in range(self.n_envs):
                if len(self.episodic_memory[i]) > 0:
                    # compute pseudo-counts
                    n_eps = self.pseudo_counts(
                        embeddings=embeddings[i].unsqueeze(0),
                        memory=self.episodic_memory[i],
                    )
                else:
                    n_eps = 0.0
                # store the pseudo-counts
                self.n_eps[i].append(n_eps)
                # update the episodic memory
                self.episodic_memory[i].append(embeddings[i])
                # clear the episodic memory if the episode is terminated or truncated
                if terminateds[i].item() or truncateds[i].item():
                    self.episodic_memory[i].clear()

    def pseudo_counts(
        self, embeddings: th.Tensor, memory: List[th.Tensor]
    ) -> th.Tensor:
        """Pseudo counts.

        Args:
            embeddings (th.Tensor): Encoded observations.
            memory (List[th.Tensor]): Episodic memory.

        Returns:
            Conut values.
        """
        memory = th.stack(memory)
        dist = (th.norm(embeddings - memory, p=2, dim=1).sort().values[: self.k]) ** 2
        self.squared_distances.append(dist)
        dist = dist / (self.dist_rms.mean + 1e-8)
        dist = th.maximum(dist - self.kernel_cluster_distance, th.zeros_like(dist))
        kernel = self.kernel_epsilon / (dist + self.kernel_epsilon)
        s = th.sqrt(kernel.sum()) + self.c

        if th.isnan(s) or s > self.sm:
            return 0.0
        else:
            return 1.0 / s

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
        super().compute(samples, sync)

        if sync:
            # compute the intrinsic rewards
            all_n_eps = [th.as_tensor(n_eps) for n_eps in self.n_eps]
            intrinsic_rewards = th.stack(all_n_eps).T.to(self.device)
            # update the running mean and std of the squared distances
            flattened_squared_distances = th.cat(self.squared_distances, dim=0)
            self.dist_rms.update(flattened_squared_distances)
            self.squared_distances.clear()
            # flush the episodic memory of intrinsic rewards
            self.n_eps = [[] for _ in range(self.n_envs)]
            # update the reward module
            self.update(samples)
            # scale the intrinsic rewards
            return self.scale(intrinsic_rewards)
        else:
            # TODO: first consider single environment for off-policy algorithms
            # compute the intrinsic rewards
            all_n_eps = [th.as_tensor(n_eps) for n_eps in self.n_eps]
            intrinsic_rewards = th.stack(all_n_eps).T.to(self.device)
            # temporarily store the intrinsic rewards and observations
            self.irs_buffer.append(intrinsic_rewards)
            self.obs_buffer.append(samples['observations'])
            if samples['truncateds'].item() or samples['terminateds'].item():
                # update the running mean and std of the squared distances
                flattened_squared_distances = th.cat(self.squared_distances, dim=0)
                self.dist_rms.update(flattened_squared_distances)
                self.squared_distances.clear()
                # update the running mean and std of the intrinsic rewards
                if self.rwd_norm_type == "rms":
                    self.rwd_norm.update(th.cat(self.irs_buffer))
                    self.irs_buffer.clear()
                if self.obs_norm_type == "rms":
                    self.obs_norm.update(th.cat(self.obs_buffer).cpu())
                    self.obs_buffer.clear()
            # flush the episodic memory of intrinsic rewards
            self.n_eps = [[] for _ in range(self.n_envs)]

            return (intrinsic_rewards / self.rwd_norm.std) * self.weight

    def update(self, samples: Dict[str, th.Tensor]) -> None:
        """Update the reward module if necessary.

        Args:
            samples (Dict[str, th.Tensor]): The collected samples same as the `compute` function.

        Returns:
            None.
        """
        # get the number of steps and environments
        (n_steps, n_envs) = samples.get("observations").size()[:2]
        obs_tensor = (
            samples.get("observations")
            .to(self.device)
            .view((n_steps * n_envs, *self.obs_shape))
        )
        next_obs_tensor = (
            samples.get("next_observations")
            .to(self.device)
            .view((n_steps * n_envs, *self.obs_shape))
        )
        # normalize the observations
        obs_tensor = self.normalize(obs_tensor)
        next_obs_tensor = self.normalize(next_obs_tensor)
        # apply one-hot encoding if the action type is discrete
        if self.action_type == "Discrete":
            actions_tensor = (
                samples.get("actions").view(n_steps * n_envs).to(self.device)
            )
            actions_tensor = F.one_hot(
                actions_tensor.long(), self.policy_action_dim
            ).float()
        else:
            actions_tensor = (
                samples.get("actions")
                .view((n_steps * n_envs, self.action_dim))
                .to(self.device)
            )
        # build the dataset and loader
        dataset = TensorDataset(obs_tensor, actions_tensor, next_obs_tensor)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)

        avg_loss = []
        # update the encoder
        for _idx, batch in enumerate(loader):
            # get the batch data
            obs, actions, next_obs = batch
            # zero the gradients
            self.opt.zero_grad()
            # get the predicted actions
            pred_actions = self.encoder(obs, next_obs)
            # compute the inverse dynamics loss
            im_loss = self.loss(pred_actions, actions)
            # use a random mask to select a subset of the training data
            mask = th.rand(len(im_loss), device=self.device)
            mask = (mask < self.update_proportion).type(th.FloatTensor).to(self.device)
            # expand the mask to match action spaces > 1
            mask = mask.unsqueeze(1).expand_as(im_loss)
            # get the masked loss
            im_loss = (im_loss * mask).sum() / th.max(
                mask.sum(), th.tensor([1], device=self.device, dtype=th.float32)
            )
            # backward and update
            im_loss.backward()
            self.opt.step()
            avg_loss.append(im_loss.item())

        try:
            self.metrics["loss"].append([self.global_step, np.mean(avg_loss)])
        except:
            pass