"""
Abstract base class for the safeguards.
"""

__all__ = ["Safeguard"]

from abc import ABC, abstractmethod

import numpy as np
from typing_extensions import Self  # MSPV 3.11


class Safeguard(ABC):
    """
    Safeguard abstract base class.
    """

    __slots__ = ()
    kind: str
    """Safeguard kind."""

    @abstractmethod
    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        """
        Check if the `decoded` array upholds the property enforced by this
        safeguard.

        Parameters
        ----------
        data : np.ndarray
            Data to be encoded.
        decoded : np.ndarray
            Decoded data.

        Returns
        -------
        ok : bool
            `True` if the check succeeded.
        """
        pass

    @abstractmethod
    def get_config(self) -> dict:
        """
        Returns the configuration of the safeguard.

        The config must include a 'kind' field with the safeguard kind. All
        values must be compatible with JSON encoding.

        Returns
        -------
        config : dict
            Configuration of the safeguard.
        """

        pass

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """
        Instantiate the safeguard from a configuration [`dict`][dict].

        Parameters
        ----------
        config : dict
            Configuration of the safeguard.

        Returns
        -------
        safeguard : Self
            Instantiated safeguard.
        """

        return cls(**config)

    def __repr__(self) -> str:
        config = {k: v for k, v in self.get_config().items() if k != "kind"}

        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in config.items())})"
