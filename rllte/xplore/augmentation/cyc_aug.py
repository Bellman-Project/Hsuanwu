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
from .pad_crop import PadCrop
from .pad_resize import PadResizePlus

class PeriodicPadCropResize(BaseAugmentation):
    """
    Periodically applies PadCrop and PadResizePlus transformations to images.

    Args:
        pad (int): The padding size.
        highest_pad_strength (int): The maximum strength of padding.
        T (int): Number of cycles to apply the transformations. Each cycle consists of one application of PadCrop followed by one application of PadResizePlus.

    
    """

    def __init__(self, pad: int, highest_pad_strength: int,T: int) -> None:

        super().__init__()
        self.pad_crop = PadCrop(pad)
        self.pad_resize_plus = PadResizePlus(highest_pad_strength)
        self.T = T

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Applies the PadCrop and PadResizePlus transformations periodically to the input images.

        Args:
            x (th.Tensor): The input images to be transformed.

        Returns:
            th.Tensor: The transformed images after applying the periodic transformations.
        """
        for _ in range(self.T):
            x = self.pad_crop(x)
            x = self.pad_resize_plus(x)
        return x

