import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from hsuanwu.common.typing import Batch, Device, Tensor, Tuple


class VanillaRolloutStorage:
    """Vanilla rollout storage for on-policy algorithms.

    Args:
        device (Device): Device (cpu, cuda, ...) on which the code should be run.
        obs_shape (Tuple): The data shape of observations.
        action_shape (Tuple): The data shape of actions.
        action_type (str): The type of actions, 'Discrete' or 'Box'.
        num_steps (int): The sample length of per rollout.
        num_envs (int): The number of parallel environments.
        discount (float): discount factor.
        gae_lambda (float): Weighting coefficient for generalized advantage estimation (GAE).

    Returns:
        Vanilla rollout storage.
    """

    def __init__(
        self,
        device: Device,
        obs_shape: Tuple,
        action_shape: Tuple,
        action_type: str,
        num_steps: int,
        num_envs: int,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        self._obs_shape = obs_shape
        self._action_shape = action_shape
        self._device = torch.device(device)
        self._num_steps = num_steps
        self._num_envs = num_envs
        self._discount = discount
        self._gae_lambda = gae_lambda

        # transition part
        self.obs = torch.empty(
            size=(num_steps, num_envs, *obs_shape),
            dtype=torch.float32,
            device=self._device,
        )
        if action_type == "Discrete":
            self._action_dim = 1
            self.actions = torch.empty(
                size=(num_steps, num_envs, self._action_dim),
                dtype=torch.float32,
                device=self._device,
            )
        elif action_type == "Box":
            self._action_dim = action_shape[0]
            self.actions = torch.empty(
                size=(num_steps, num_envs, self._action_dim),
                dtype=torch.float32,
                device=self._device,
            )
        else:
            raise NotImplementedError
        self.rewards = torch.empty(
            size=(num_steps, num_envs, 1), dtype=torch.float32, device=self._device
        )
        self.terminateds = torch.empty(
            size=(num_steps + 1, num_envs, 1), dtype=torch.float32, device=self._device
        )
        self.truncateds = torch.empty(
            size=(num_steps + 1, num_envs, 1), dtype=torch.float32, device=self._device
        )
        # first next_terminated
        self.terminateds[0].copy_(torch.zeros(num_envs, 1).to(self._device))
        self.truncateds[0].copy_(torch.zeros(num_envs, 1).to(self._device))
        # extra part
        self.log_probs = torch.empty(
            size=(num_steps, num_envs, 1), dtype=torch.float32, device=self._device
        )
        self.values = torch.empty(
            size=(num_steps, num_envs, 1), dtype=torch.float32, device=self._device
        )
        self.returns = torch.empty(
            size=(num_steps, num_envs, 1), dtype=torch.float32, device=self._device
        )
        self.advantages = torch.empty(
            size=(num_steps, num_envs, 1), dtype=torch.float32, device=self._device
        )

        self._global_step = 0

    def add(
        self,
        obs: Tensor,
        actions: Tensor,
        rewards: Tensor,
        terminateds: Tensor,
        truncateds: Tensor,
        log_probs: Tensor,
        values: Tensor,
    ) -> None:
        """Add sampled transitions into storage.

        Args:
            obs (Tensor): Observations.
            actions (Tensor): Actions.
            rewards (Tensor): Rewards.
            terminateds (Tensor): Terminateds.
            truncateds (Tensor): Truncateds.
            log_probs (Tensor): Log of the probability evaluated at `actions`.
            values (Tensor): Estimated values.

        Returns:
            None.
        """
        self.obs[self._global_step].copy_(obs)
        self.actions[self._global_step].copy_(actions)
        self.rewards[self._global_step].copy_(rewards)
        self.terminateds[self._global_step + 1].copy_(terminateds)
        self.truncateds[self._global_step + 1].copy_(truncateds)
        self.log_probs[self._global_step].copy_(log_probs)
        self.values[self._global_step].copy_(values)

        self._global_step = (self._global_step + 1) % self._num_steps

    def reset(self) -> None:
        """Reset the terminal state of each env."""
        self.terminateds[0].copy_(self.terminateds[-1])
        self.truncateds[0].copy_(self.truncateds[-1])

    def compute_returns_and_advantages(self, last_values: Tensor) -> None:
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
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.terminateds[step + 1]
                next_values = self.values[step + 1]
            delta = (
                self.rewards[step]
                + self._discount * next_values * next_non_terminal
                - self.values[step]
            )
            gae = delta + self._discount * self._gae_lambda * next_non_terminal * gae
            self.advantages[step] = gae

        self.returns = self.advantages + self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (
            self.advantages.std() + 1e-5
        )

    def generator(self, num_mini_batch: int = None) -> Batch:
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

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=True
        )

        for indices in sampler:
            batch_obs = self.obs.view(-1, *self._obs_shape)[indices]
            batch_actions = self.actions.view(-1, self._action_dim)[indices]
            batch_values = self.values.view(-1, 1)[indices]
            batch_returns = self.returns.view(-1, 1)[indices]
            batch_terminateds = self.terminateds[:-1].view(-1, 1)[indices]
            batch_truncateds = self.truncateds[:-1].view(-1, 1)[indices]
            batch_old_log_probs = self.log_probs.view(-1, 1)[indices]
            adv_targ = self.advantages.view(-1, 1)[indices]

            yield batch_obs, batch_actions, batch_values, batch_returns, batch_terminateds, batch_truncateds, batch_old_log_probs, adv_targ
