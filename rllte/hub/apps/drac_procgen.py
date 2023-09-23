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
@article{raileanu2021automatic,
  title={Automatic data augmentation for generalization in reinforcement learning},
  author={Raileanu, Roberta and Goldstein, Maxwell and Yarats, Denis and Kostrikov, Ilya and Fergus, Rob},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={5402--5415},
  year={2021}
}
"""

import argparse

from rllte.agent import DrAC
from rllte.env import make_procgen_env
from rllte.xploit.encoder import EspeholtResidualEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--env-id", type=str, default="bigfish")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--num-train-steps", type=int, default=2.5e7)

if __name__ == "__main__":
    args = parser.parse_args()
    # create env
    env = make_procgen_env(
        env_id=args.env_id,
        num_envs=64,
        device=args.device,
        seed=args.seed,
        gamma=0.99,
        num_levels=200,
        start_level=0,
        distribution_mode="easy",
    )
    eval_env = make_procgen_env(
        env_id=args.env_id,
        num_envs=1,
        device=args.device,
        seed=args.seed,
        gamma=0.99,
        num_levels=0,
        start_level=0,
        distribution_mode="easy",
    )
    # create agent
    feature_dim = 256
    agent = DrAC(
        env=env,
        eval_env=eval_env,
        tag=f"ppo_procgen_{args.env_id}_seed_{args.seed}",
        seed=args.seed,
        device=args.device,
        num_steps=256,
        feature_dim=feature_dim,
        batch_size=2048,
        lr=5e-4,
        eps=1e-5,
        clip_range=0.2,
        clip_range_vf=0.2,
        n_epochs=3,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        init_fn="xavier_uniform",
    )
    encoder = EspeholtResidualEncoder(observation_space=env.observation_space, feature_dim=feature_dim)
    agent.set(encoder=encoder)
    # training
    agent.train(num_train_steps=args.num_train_steps)
