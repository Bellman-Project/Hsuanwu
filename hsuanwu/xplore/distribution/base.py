import torch

from hsuanwu.common.typing import ABC, Tensor, TorchSize, abstractmethod


class BaseDistribution(ABC):
    """Abstract base class of distributions."""

    def __init__(self) -> None:
        super().__init__()
        self.dist = None

    @abstractmethod
    def sample(self, sample_shape: TorchSize = torch.Size()) -> Tensor:
        """Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.

        Args:
            sample_shape (TorchSize): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """

    @abstractmethod
    def rsample(self, sample_shape: TorchSize = torch.Size()) -> Tensor:
        """Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched.

        Args:
            sample_shape (TorchSize): The size of the sample to be drawn.

        Returns:
            A sample_shape shaped sample.
        """

    @abstractmethod
    def log_prob(self, value: Tensor) -> Tensor:
        """Returns the log of the probability density/mass function evaluated at `value`.

        Args:
            value (Tensor): The value to be evaluated.

        Returns:
            The log_prob value.
        """

    @abstractmethod
    def entropy(self) -> Tensor:
        """Returns the Shannon entropy of distribution."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the distribution."""

    @abstractmethod
    def mean(self) -> Tensor:
        """Returns the mean of the distribution."""

    @abstractmethod
    def mode(self) -> Tensor:
        """Returns the mode of the distribution."""
