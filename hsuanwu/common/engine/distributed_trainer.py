import os

os.environ["OMP_NUM_THREADS"] = "1"
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch
from torch import multiprocessing as mp

from hsuanwu.common.engine import BasePolicyTrainer
from hsuanwu.common.logger import Logger
from hsuanwu.common.typing import (
    Dict,
    DictConfig,
    Env,
    List,
    Ndarray,
    NNModule,
    Storage,
    Tensor,
)


class Environment:
    """An env wrapper to adapt to the distributed trainer.

    Args:
        env (env): A Gym-like env.

    Returns:
        Processed env.
    """

    def __init__(self, env: Env) -> None:
        self.env = env
        self.episode_return = None
        self.episode_step = None

    def reset(self, seed) -> Dict[str, Tensor]:
        """Reset the environment."""
        init_reward = torch.zeros(1, 1)
        # This supports only single-tensor actions ATM.
        init_last_action = torch.zeros(1, 1, dtype=torch.int64)
        self.episode_return = torch.zeros(1, 1)
        self.episode_step = torch.zeros(1, 1, dtype=torch.int32)
        init_terminated = torch.ones(1, 1, dtype=torch.uint8)
        init_truncated = torch.ones(1, 1, dtype=torch.uint8)

        obs, info = self.env.reset(seed=seed)
        obs = self._format_obs(obs)

        return dict(
            obs=obs,
            reward=init_reward,
            terminated=init_terminated,
            truncated=init_truncated,
            episode_return=self.episode_return,
            episode_step=self.episode_step,
            last_action=init_last_action,
        )

    def step(self, action: Tensor) -> Dict[str, Tensor]:
        """Step function that returns a dict consists of current and history observation and action.

        Args:
            action (Tensor): Action tensor.

        Returns:
            Step information dict.
        """
        obs, reward, terminated, truncated, info = self.env.step(action.item())
        self.episode_step += 1
        self.episode_return += reward
        episode_step = self.episode_step
        episode_return = self.episode_return
        if terminated or truncated:
            obs, info = self.env.reset()
            self.episode_return = torch.zeros(1, 1)
            self.episode_step = torch.zeros(1, 1, dtype=torch.int32)

        obs = self._format_obs(obs)
        reward = torch.as_tensor(reward, dtype=torch.float32).view(1, 1)
        terminated = torch.as_tensor(terminated, dtype=torch.uint8).view(1, 1)
        truncated = torch.as_tensor(truncated, dtype=torch.uint8).view(1, 1)

        return dict(
            obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            episode_return=episode_return,
            episode_step=episode_step,
            last_action=action,
        )

    def close(self) -> None:
        """Close the environment."""
        self.env.close()

    def _format_obs(self, obs: Ndarray) -> Tensor:
        """Reformat the observation by adding an time dimension.

        Args:
            obs (NdArray): Observation.

        Returns:
            Formatted observation.
        """
        obs = torch.from_numpy(np.array(obs))
        return obs.view((1, 1) + obs.shape)


class DistributedTrainer(BasePolicyTrainer):
    """Trainer for on-policy algorithms.

    Args:
        train_env (Env): A list of Gym-like environments for training.
        test_env (Env): A Gym-like environment for testing.
        cfgs (DictConfig): Dict config for configuring RL algorithms.

    Returns:
        On-policy trainer instance.
    """

    def __init__(self, cfgs: DictConfig, train_env: Env, test_env: Env = None) -> None:
        super().__init__(cfgs, train_env, test_env)
        self._logger.info(f"Deploying DistributedTrainer...")
        # xploit part
        self._learner = hydra.utils.instantiate(self._cfgs.learner)
        self._learner.actor.encoder = hydra.utils.instantiate(self._cfgs.encoder)
        self._learner.learner.encoder = hydra.utils.instantiate(self._cfgs.encoder)

        self._learner.actor.share_memory()
        self._learner.learner.to(self._device)
        self._learner.opt = torch.optim.RMSprop(
            self._learner.learner.parameters(),
            lr=self._learner.lr,
            eps=self._learner.eps,
        )

        def lr_lambda(epoch):
            return (
                1.0
                - min(
                    epoch * self._cfgs.num_steps * self._cfgs.num_learners,
                    self._cfgs.num_train_steps,
                )
                / self._cfgs.num_train_steps
            )

        self._learner.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self._learner.opt, lr_lambda
        )

        ## TODO: build storage
        self._shared_storages = hydra.utils.instantiate(self._cfgs.storage)
        self._train_env = self._train_env.envs

        # xplore part
        ## TODO: build distribution
        if "Noise" in self._cfgs.distribution._target_:
            dist = hydra.utils.instantiate(self._cfgs.distribution)
        else:
            dist = hydra.utils.get_class(self._cfgs.distribution._target_)

        self._learner.actor.dist = dist
        self._learner.learner.dist = dist

    @staticmethod
    def act(
        cfgs: DictConfig,
        logger: Logger,
        gym_env: Env,
        actor_idx: int,
        actor_model: NNModule,
        free_queue: mp.SimpleQueue,
        full_queue: mp.SimpleQueue,
        storages: List[Storage],
        init_actor_state_storages: List[Tensor],
    ) -> None:
        """Sampling function for each actor.

        Args:
            cfgs (DictConfig): Training configs.
            logger (Logger): Hsuanwu logger.
            gym_env (Env): A Gym-like environment.
            actor_idx (int): The index of actor.
            actor_model (NNMoudle): Actor network.
            free_queue (Queue): Free queue for communication.
            full_queue (Queue): Full queue for communication.
            storages (List[Storage]): A list of shared storages.
            init_actor_state_storages (List[Tensor]): Initial states for LSTM.

        Returns:
            None.
        """
        try:
            logger.info(f"Actor {actor_idx} started!")

            env = Environment(gym_env)
            seed = actor_idx * int.from_bytes(os.urandom(4), byteorder="little")
            env_output = env.reset(seed)

            actor_state = actor_model.init_state(batch_size=1)
            actor_output, _ = actor_model.get_action(env_output, actor_state)

            while True:
                idx = free_queue.get()
                if idx is None:
                    break

                # Write old rollout end.
                for key in env_output:
                    storages[key][idx][0, ...] = env_output[key]
                for key in actor_output:
                    storages[key][idx][0, ...] = actor_output[key]
                for i, tensor in enumerate(actor_state):
                    init_actor_state_storages[idx][i][...] = tensor

                # Do new rollout.
                for t in range(cfgs.num_steps):
                    with torch.no_grad():
                        actor_output, actor_state = actor_model.get_action(
                            env_output, actor_state
                        )
                    env_output = env.step(actor_output["action"])

                    for key in env_output:
                        storages[key][idx][t + 1, ...] = env_output[key]
                    for key in actor_output:
                        storages[key][idx][t + 1, ...] = actor_output[key]

                full_queue.put(idx)

        except KeyboardInterrupt:
            pass  # Return silently.
        except Exception as e:
            logger.error(f"Exception in worker process {actor_idx}!")
            traceback.print_exc()
            raise e

    def train(self) -> None:
        """Training function"""
        global_step = 0
        global_episode = 0
        metrics = dict()
        episode_rewards = deque(maxlen=10)
        episode_steps = deque(maxlen=10)

        def sample_and_update(i, lock=threading.Lock()):
            """Thread target for the learning process."""
            nonlocal global_step, global_episode, metrics
            while global_step < self._cfgs.num_train_steps:
                batch, actor_states = self._shared_storages.sample(
                    device=self._cfgs.device,
                    batch_size=self._cfgs.storage.batch_size,
                    free_queue=free_queue,
                    full_queue=full_queue,
                    storages=self._shared_storages.storages,
                    init_actor_state_storages=init_actor_state_storages,
                )
                metrics = self._learner.update(
                    cfgs=self._cfgs,
                    actor_model=self._learner.actor,
                    learner_model=self._learner.learner,
                    batch=batch,
                    init_actor_states=actor_states,
                    optimizer=self._learner.opt,
                    lr_scheduler=self._learner.lr_scheduler,
                )
                with lock:
                    global_step += self._cfgs.num_steps * self._cfgs.storage.batch_size
                    global_episode += self._cfgs.storage.batch_size

        # TODO: Add initial RNN state.
        init_actor_state_storages = []
        for _ in range(self._cfgs.storage.num_storages):
            state = self._learner.actor.init_state(batch_size=1)
            for t in state:
                t.share_memory_()
            init_actor_state_storages.append(state)

        actor_pool = []
        ctx = mp.get_context("fork")
        free_queue = ctx.SimpleQueue()
        full_queue = ctx.SimpleQueue()

        for actor_idx in range(self._cfgs.num_actors):
            actor = ctx.Process(
                target=self.act,
                kwargs={
                    "cfgs": self._cfgs,
                    "logger": self._logger,
                    "gym_env": self._train_env[actor_idx],
                    "actor_idx": actor_idx,
                    "actor_model": self._learner.actor,
                    "free_queue": free_queue,
                    "full_queue": full_queue,
                    "storages": self._shared_storages.storages,
                    "init_actor_state_storages": init_actor_state_storages,
                },
            )
            actor.start()
            actor_pool.append(actor)

        for m in range(self._cfgs.storage.num_storages):
            free_queue.put(m)

        threads = []
        for i in range(self._cfgs.num_learners):
            thread = threading.Thread(
                target=sample_and_update, name="sample-and-update-%d" % i, args=(i,)
            )
            thread.start()
            self._logger.info(f"Learner {i} started!")
            threads.append(thread)

        try:
            while global_step < self._cfgs.num_train_steps:
                start_step = global_step
                time.sleep(5)

                if metrics.get("episode_returns"):
                    episode_rewards.extend(metrics["episode_returns"])
                    episode_steps.extend(metrics["episode_steps"])

                if len(episode_rewards) > 0:
                    episode_time, total_time = self._timer.reset()
                    train_metrics = {
                        "step": global_step,
                        "episode": global_episode,
                        "episode_length": np.mean(episode_steps),
                        "episode_reward": np.mean(episode_rewards),
                        "fps": (global_step - start_step) / episode_time,
                        "total_time": total_time,
                    }
                    self._logger.train(msg=train_metrics)

        except KeyboardInterrupt:
            # TODO: join actors then quit.
            return
        else:
            for thread in threads:
                thread.join()
            self._logger.info("Training Accomplished!")
            # save model
            self.save()
        finally:
            for _ in range(self._cfgs.num_actors):
                free_queue.put(None)
            for actor in actor_pool:
                actor.join(timeout=1)

    def test(self) -> None:
        """Testing function."""
        pass

    def save(self) -> None:
        """Save the trained model."""
        save_dir = Path.cwd() / "model"
        save_dir.mkdir(exist_ok=True)
        torch.save(self._learner.actor, save_dir / "actor.pth")
        torch.save(self._learner.learner, save_dir / "learner.pth")
