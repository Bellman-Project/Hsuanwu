from typing import Generator, Union

import gymnasium as gym
import torch as th
from omegaconf import DictConfig
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from hsuanwu.xploit.storage.base import BaseStorage


class DecoupledRolloutStorage(BaseStorage):
    """Decoupled rollout storage for on-policy algorithms like DAAC.

    Args:
        observation_space (Space or DictConfig): The observation space of environment. When invoked by Hydra,
            'observation_space' is a 'DictConfig' like {"shape": observation_space.shape, }.
        action_space (Space or DictConfig): The action space of environment. When invoked by Hydra,
            'action_space' is a 'DictConfig' like
            {"shape": (n, ), "type": "Discrete", "range": [0, n - 1]} or
            {"shape": action_space.shape, "type": "Box", "range": [action_space.low[0], action_space.high[0]]}.
        device (str): Device (cpu, cuda, ...) on which the code should be run.
        num_steps (int): The sample length of per rollout.
        num_envs (int): The number of parallel environments.
        discount (float): discount factor.
        gae_lambda (float): Weighting coefficient for generalized advantage estimation (GAE).

    Returns:
        Vanilla rollout storage.
    """

    def __init__(
        self,
        observation_space: Union[gym.Space, DictConfig],
        action_space: Union[gym.Space, DictConfig],
        device: str = "cpu",
        num_steps: int = 256,
        num_envs: int = 8,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        super().__init__(observation_space, action_space, device)
        self._num_steps = num_steps
        self._num_envs = num_envs
        self._discount = discount
        self._gae_lambda = gae_lambda

        # transition part
        self.obs = th.empty(
            size=(num_steps + 1, num_envs, *self._obs_shape),
            dtype=th.float32,
            device=self._device,
        )
        if self._action_type == "Discrete":
            self._action_dim = 1
            self.actions = th.empty(
                size=(num_steps, num_envs),
                dtype=th.float32,
                device=self._device,
            )
        elif self._action_type == "Box":
            self._action_dim = self._action_shape[0]
            self.actions = th.empty(
                size=(num_steps, num_envs, self._action_dim),
                dtype=th.float32,
                device=self._device,
            )
        elif self._action_type == "MultiBinary":
            self._action_dim = self._action_shape[0]
            self.actions = th.empty(
                size=(num_steps, num_envs, self._action_dim),
                dtype=th.float32,
                device=self._device,
            )
        else:
            raise NotImplementedError
        self.rewards = th.empty(size=(num_steps, num_envs), dtype=th.float32, device=self._device)
        self.terminateds = th.empty(size=(num_steps + 1, num_envs), dtype=th.float32, device=self._device)
        self.truncateds = th.empty(size=(num_steps + 1, num_envs), dtype=th.float32, device=self._device)
        # first next_terminated
        self.terminateds[0].copy_(th.zeros(num_envs).to(self._device))
        self.truncateds[0].copy_(th.zeros(num_envs).to(self._device))
        # extra part
        self.log_probs = th.empty(size=(num_steps, num_envs), dtype=th.float32, device=self._device)
        self.values = th.empty(size=(num_steps, num_envs), dtype=th.float32, device=self._device)
        self.returns = th.empty(size=(num_steps, num_envs), dtype=th.float32, device=self._device)
        self.advantages = th.empty(size=(num_steps, num_envs), dtype=th.float32, device=self._device)
        self.adv_preds = th.empty(size=(num_steps, num_envs), dtype=th.float32, device=self._device)

        self._global_step = 0

    def add(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        rewards: th.Tensor,
        terminateds: th.Tensor,
        truncateds: th.Tensor,
        next_obs: th.Tensor,
        log_probs: th.Tensor,
        values: th.Tensor,
        adv_preds: th.Tensor,
    ) -> None:
        """Add sampled transitions into storage.

        Args:
            obs (Tensor): Observations.
            actions (Tensor): Actions.
            rewards (Tensor): Rewards.
            terminateds (Tensor): Terminateds.
            truncateds (Tensor): Truncateds.
            next_obs (Tensor): Next observations.
            log_probs (Tensor): Log of the probability evaluated at `actions`.
            values (Tensor): Estimated values.
            adv_preds (Tensor): Predicted advantages.

        Returns:
            None.
        """
        self.obs[self._global_step].copy_(obs)
        self.actions[self._global_step].copy_(actions)
        self.rewards[self._global_step].copy_(rewards)
        self.terminateds[self._global_step + 1].copy_(terminateds)
        self.truncateds[self._global_step + 1].copy_(truncateds)
        self.obs[self._global_step + 1].copy_(next_obs)
        self.log_probs[self._global_step].copy_(log_probs)
        self.values[self._global_step].copy_(values[:, 0])
        self.adv_preds[self._global_step].copy_(adv_preds[:, 0])

        self._global_step = (self._global_step + 1) % self._num_steps

    def update(self) -> None:
        """Reset the terminal state of each env."""
        self.terminateds[0].copy_(self.terminateds[-1])
        self.truncateds[0].copy_(self.truncateds[-1])

    def compute_returns_and_advantages(self, last_values: th.Tensor) -> None:
        """Perform generalized advantage estimation (GAE).

        Args:
            last_values (Tensor): Estimated values of the last step.
            gamma (float): Discount factor.
            gae_lamdba (float): Coefficient of GAE.

        Returns:
            None.
        """
        gae = 0
        for step in reversed(range(self._num_steps)):
            if step == self._num_steps - 1:
                next_non_terminal = 1.0 - self.terminateds[-1]
                next_values = last_values[:, 0]
            else:
                next_non_terminal = 1.0 - self.terminateds[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self._discount * next_values * next_non_terminal - self.values[step]
            gae = delta + self._discount * self._gae_lambda * next_non_terminal * gae
            self.advantages[step] = gae

        self.returns = self.advantages + self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-5)

    def sample(self, num_mini_batch: int = 8) -> Generator:
        """Sample data from storage.

        Args:
            num_mini_batch (int): Number of mini-batches

        Returns:
            Batch data.
        """
        batch_size = self._num_envs * self._num_steps

        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(self._num_envs, self._num_steps, batch_size, num_mini_batch)
        )
        mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True)

        for indices in sampler:
            batch_obs = self.obs[:-1].view(-1, *self._obs_shape)[indices]
            batch_actions = self.actions.view(-1, self._action_dim)[indices]
            batch_values = self.values.view(-1, 1)[indices]
            batch_adv_preds = self.adv_preds.view(-1, 1)[indices]
            batch_returns = self.returns.view(-1, 1)[indices]
            batch_terminateds = self.terminateds[:-1].view(-1, 1)[indices]
            batch_truncateds = self.truncateds[:-1].view(-1, 1)[indices]
            batch_old_log_probs = self.log_probs.view(-1, 1)[indices]
            adv_targ = self.advantages.view(-1, 1)[indices]

            yield (
                batch_obs,
                batch_actions,
                batch_values,
                batch_adv_preds,
                batch_returns,
                batch_terminateds,
                batch_truncateds,
                batch_old_log_probs,
                adv_targ,
            )