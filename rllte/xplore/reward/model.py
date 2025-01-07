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


from typing import Tuple
from torch import nn
from torch.nn import functional as F

import numpy as np
import torch as th
import math

def orthogonal_layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    th.nn.init.orthogonal_(layer.weight, std)
    th.nn.init.constant_(layer.bias, bias_const)
    return layer

def kaiming_he_init(layer):
    th.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    if layer.bias is not None:
        th.nn.init.zeros_(layer.bias)
    return layer

def default_layer_init(layer):
    stdv = 1. / math.sqrt(layer.weight.size(1))
    layer.weight.data.uniform_(-stdv, stdv)
    if layer.bias is not None:
        layer.bias.data.uniform_(-stdv, stdv)
    return layer

class ObservationEncoder(nn.Module):
    """Encoder for encoding observations.

    Args:
        obs_shape (Tuple): The data shape of observations.
        latent_dim (int): The dimension of encoding vectors.
        encoder_model (str): The network architecture of the encoder from ['mnih', 'espeholt']. Defaults to 'mnih'
        weight_init (str): The weight initialization method from ['default', 'orthogonal', 'kaiming he']. Defaults to 'default'

    Returns:
        Encoder instance.
    """

    def __init__(self, obs_shape: Tuple, latent_dim: int, encoder_model:str = "mnih", weight_init="default") -> None:
        super().__init__()

        if weight_init == "orthogonal":
            init_ = orthogonal_layer_init
        elif weight_init == "kaiming he":
            init_ = kaiming_he_init
        elif weight_init == "default":
            init_ = default_layer_init
        else:
            raise ValueError("Invalid weight_init")


        # visual
        if encoder_model == "mnih" and len(obs_shape) > 2:
            self.trunk = nn.Sequential(
                init_(nn.Conv2d(obs_shape[0], 32, 8, stride=4)),
                nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                init_(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
            )

            with th.no_grad():
                sample = th.ones(size=tuple(obs_shape)).float()
                n_flatten = self.trunk(sample.unsqueeze(0)).shape[1]

            self.trunk.append(init_(nn.Linear(n_flatten, latent_dim)))
            self.trunk.append(nn.ReLU())
        elif encoder_model == "espeholt" and len(obs_shape) > 2:
            self.trunk = nn.Sequential(
                init_(nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=2, padding=1)),
                nn.ELU(),
                init_(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)),
                nn.ELU(),
                init_(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)),
                nn.ELU(),
                init_(nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)),
                nn.ELU(),
                nn.Flatten(),
            )
            with th.no_grad():
                sample = th.ones(size=tuple(obs_shape)).float()
                n_flatten = self.trunk(sample.unsqueeze(0)).shape[1]

            self.trunk.append(init_(nn.Linear(n_flatten, latent_dim)))
            self.trunk.append(nn.ReLU())
        else:
            self.trunk = nn.Sequential(
                init_(nn.Linear(obs_shape[0], 256)), 
                nn.ReLU()
            )
            self.trunk.append(init_(nn.Linear(256, latent_dim)))

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """Encode the input tensors.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Encoding tensors.
        """
        # normalization for intrinsic rewards is dealt with in the base intrinsic reward class
        return self.trunk(obs)
    
class InverseDynamicsEncoder(nn.Module):
    """Encoder with inverse dynamics prediction.

    Args:
        obs_shape (Tuple): The data shape of observations.
        action_dim (int): The dimension of actions.
        latent_dim (int): The dimension of encoding vectors.

    Returns:
        Encoder instance.
    """

    def __init__(self, obs_shape: Tuple, action_dim: int, latent_dim: int, encoder_model:str="mnih", weight_init="default") -> None:
        super().__init__()

        self.encoder = ObservationEncoder(obs_shape, latent_dim, encoder_model=encoder_model, weight_init=weight_init)
        self.policy = InverseDynamicsModel(latent_dim, action_dim, encoder_model=encoder_model, weight_init=weight_init)

    def forward(self, obs: th.Tensor, next_obs: th.Tensor) -> th.Tensor:
        """Forward function for outputing predicted actions.

        Args:
            obs (th.Tensor): Current observations.
            next_obs (th.Tensor): Next observations.

        Returns:
            Predicted actions.
        """
        h = self.encoder(obs)
        next_h = self.encoder(next_obs)

        actions = self.policy(h, next_h)
        return actions

    def encode(self, obs: th.Tensor) -> th.Tensor:
        """Encode the input tensors.

        Args:
            obs (th.Tensor): Observations.

        Returns:
            Encoding tensors.
        """
        return self.encoder(obs)
    

class InverseDynamicsModel(nn.Module):
    """Inverse model for reconstructing transition process.

    Args:
        latent_dim (int): The dimension of encoding vectors of the observations.
        action_dim (int): The dimension of predicted actions.

    Returns:
        Model instance.
    """

    def __init__(self, latent_dim, action_dim, encoder_model="mnih", weight_init="default") -> None:
        super().__init__()

        self.trunk = ObservationEncoder(obs_shape=(latent_dim * 2,), latent_dim=action_dim, encoder_model=encoder_model, weight_init=weight_init)

    def forward(self, obs: th.Tensor, next_obs: th.Tensor) -> th.Tensor:
        """Forward function for outputing predicted actions.

        Args:
            obs (th.Tensor): Current observations.
            next_obs (th.Tensor): Next observations.

        Returns:
            Predicted actions.
        """
        return self.trunk(th.cat([obs, next_obs], dim=1))

class ForwardDynamicsModel(nn.Module):
    """Forward model for reconstructing transition process.

    Args:
        latent_dim (int): The dimension of encoding vectors of the observations.
        action_dim (int): The dimension of predicted actions.

    Returns:
        Model instance.
    """

    def __init__(self, latent_dim, action_dim, encoder_model="mnih", weight_init="default") -> None:
        super().__init__()

        self.trunk = ObservationEncoder(obs_shape=(latent_dim + action_dim,), latent_dim=latent_dim, encoder_model=encoder_model, weight_init=weight_init)

    def forward(self, obs: th.Tensor, pred_actions: th.Tensor) -> th.Tensor:
        """Forward function for outputing predicted next-obs.

        Args:
            obs (th.Tensor): Current observations.
            pred_actions (th.Tensor): Predicted observations.

        Returns:
            Predicted next-obs.
        """
        return self.trunk(th.cat([obs, pred_actions], dim=1))