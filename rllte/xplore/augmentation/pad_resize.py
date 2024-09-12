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
import torchvision.transforms as T
from rllte.common.prototype import BaseAugmentation



class PadResizePlus(BaseAugmentation):
    """
    Pad and resize operation for processing image-based observations.

    This class pads the images randomly and then crops them back to their 
    original size, followed by resizing.

    Args:
        highest_pad_strength (int): The maximum strength of padding.
    """

    def __init__(self, highest_pad_strength: int) -> None:
        """
        Initializes the PadResizePlus with the highest padding strength.

        Args:
            highest_pad_strength (int): The maximum strength of padding.
        """
        super().__init__()
        self.highest_pad_strength = highest_pad_strength

    def crop(self, imgs: th.Tensor, pad_x: int, pad_y: int) -> th.Tensor:
        """
        Crops the padded images.

        Args:
            imgs (th.Tensor): Padded images.
            pad_x (int): Padding along the width.
            pad_y (int): Padding along the height.

        Returns:
            th.Tensor: Cropped images.
        """
        n, c, h_pad, w_pad = imgs.size()

        # Calculate the crop size
        crop_x = w_pad - pad_x
        crop_y = h_pad - pad_y

        # Create a grid for cropping
        eps_x = 1.0 / w_pad
        eps_y = 1.0 / h_pad
        x_range = th.linspace(-1.0 + eps_x, 1.0 - eps_x, w_pad, device=imgs.device, dtype=imgs.dtype)[:crop_x]
        y_range = th.linspace(-1.0 + eps_y, 1.0 - eps_y, h_pad, device=imgs.device, dtype=imgs.dtype)[:crop_y]
        grid_y, grid_x = th.meshgrid(y_range, x_range)
        base_grid = th.stack([grid_x, grid_y], dim=-1)

        # Calculate random shifts
        shift_x = th.randint(0, pad_x + 1, size=(n, 1, 1, 1), device=imgs.device, dtype=imgs.dtype)
        shift_y = th.randint(0, pad_y + 1, size=(n, 1, 1, 1), device=imgs.device, dtype=imgs.dtype)
        shift_x *= 2.0 / w_pad
        shift_y *= 2.0 / h_pad
        shift = th.cat([shift_x, shift_y], dim=-1)
        grid = base_grid + shift

        # Apply the grid to the input tensor to perform cropping
        padded_imgs_after_crop = F.grid_sample(imgs, grid)
        return padded_imgs_after_crop

    def forward(self, imgs: th.Tensor) -> th.Tensor:
        """
        Applies padding, cropping, and resizing to the input images.

        Args:
            imgs (th.Tensor): Input images.

        Returns:
            th.Tensor: Processed images.
        """
        strength = th.randint(0, self.highest_pad_strength + 1, (1,)).item()

        _, _, h, w = imgs.shape
        pad_x = th.randint(0, strength + 1, (1,)).item()
        pad_y = strength - pad_x
        padded_imgs_before_crop = F.pad(imgs, (pad_x, pad_x, pad_y, pad_y))

        padded_imgs_after_crop = self.crop(padded_imgs_before_crop, pad_x, pad_y)
        resize = T.Resize(size=(h, w))
        return resize(padded_imgs_after_crop)
