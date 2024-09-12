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


import torch as th
import torch.nn.functional as F
from rllte.common.prototype import BaseAugmentation

class PadCrop(BaseAugmentation):
    """
    Random shift operation for processing image-based observations.

    Args:
        pad (int): Padding size to apply before shifting.

    Returns:
        Augmented images with random shifts applied.
    """

    def __init__(self, pad: int) -> None:
        """
        Initializes the PadCrop with specified padding.

        Args:
            pad (int): The padding size.
        """
        super().__init__()
        self.pad = pad

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Applies random shifts to the input images.

        Args:
            x (th.Tensor): Input images.

        Returns:
            th.Tensor: Shifted images.
        """
        n, c, h, w = x.size()
        assert h == w, "Height and width must be equal."
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')

        # Compute the grid for shifting
        eps = 1.0 / (h + 2 * self.pad)
        arange = th.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = th.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        # Calculate random shifts
        shift = th.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        # Apply shifts to the grid and perform grid sampling
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode='zeros', align_corners=False)
