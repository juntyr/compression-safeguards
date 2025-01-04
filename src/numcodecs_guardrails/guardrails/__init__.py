__all__ = ["Guardrail"]

from abc import ABC, abstractmethod
from typing import Self

import numpy as np


class Guardrail(ABC):
    """
    Guardrail abstract base class.
    """

    kind: str
    """Guardrail kind."""
    _priority: int

    @abstractmethod
    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        """
        Check if the `decoded` array upholds the property enforced by this
        guardrail.

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
        Returns the configuration of the guardrail.

        The config must include a 'kind' field with the guardrail kind. All
        values must be compatible with JSON encoding.

        Returns
        -------
        config : dict
            Configuration of the guardrail.
        """

        pass

    @classmethod
    def from_config(cls, config: dict) -> Self:
        """
        Instantiate the guardrail from a configuration [`dict`][dict].

        Parameters
        ----------
        config : dict
            Configuration of the guardrail.

        Returns
        -------
        guardrail : Self
            Instantiated guardrail.
        """

        return cls(**config)

    def __repr__(self) -> str:
        config = {k: v for k, v in self.get_config().items() if k != "kind"}

        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in config.items())})"
