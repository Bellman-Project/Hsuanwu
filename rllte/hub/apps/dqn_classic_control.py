# =============================================================================
# MIT License

# Copyright (c) 2023 Reinforcement Learning Evolution Foundation

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


"""
The following hyperparameters are from the paper:
@article{raffin2021stable,
  title={Stable-baselines3: Reliable reinforcement learning implementations},
  author={Raffin, Antonin and Hill, Ashley and Gleave, Adam and Kanervisto, Anssi and Ernestus, Maximilian and Dormann, Noah},
  journal={The Journal of Machine Learning Research},
  volume={22},
  number={1},
  pages={12348--12355},
  year={2021},
  publisher={JMLRORG}
}
"""

import argparse

from rllte.agent import DQN
from rllte.env import make_rllte_env

parser = argparse.ArgumentParser()
parser.add_argument("--env-id", type=str, default="CartPole-v1")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num-train-steps", type=int, default=100000)

if __name__ == "__main__":
    args = parser.parse_args()
    # create env
    env = make_rllte_env(
        env_id=args.env_id,
        num_envs=1,
        device=args.device,
        seed=args.seed,
        parallel=False,
    )
    eval_env = make_rllte_env(
        env_id=args.env_id,
        num_envs=1,
        device=args.device,
        seed=args.seed,
        parallel=False,
    )
    # create agent
    agent = DQN(
        env=env,
        eval_env=eval_env,
        tag=f"dqn_classic_control_{args.env_id}_seed_{args.seed}",
        seed=args.seed,
        device=args.device,
        feature_dim=64,
        batch_size=128,
        lr=2.5e-4,
        eps=1e-8,
        hidden_dim=64,
        tau=1.0,
        update_every_steps=10,
        target_update_freq=500,
        num_init_steps=10000,
        init_fn="orthogonal",
    )
    # training
    agent.train(num_train_steps=args.num_train_steps)
