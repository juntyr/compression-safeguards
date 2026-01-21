"""
Abstract base class for the safeguards.
"""

__all__ = ["Safeguard"]

from abc import ABC, abstractmethod
from collections.abc import Set
from typing import ClassVar, Literal, Self

import numpy as np
from typing_extensions import override  # MSPV 3.12

from ..utils.bindings import Bindings, Parameter
from ..utils.typing import JSON, S, T


class Safeguard(ABC):
    """
    Safeguard abstract base class.
    """

    __slots__: tuple[str, ...] = ()
    kind: ClassVar[str]
    """Safeguard kind."""

    @property
    @abstractmethod
    def late_bound(self) -> Set[Parameter]:
        """
        The set of late-bound parameters that this safeguard has.

        Late-bound parameters are only bound when checking and applying the
        safeguard, in contrast to the normal early-bound parameters that are
        configured during safeguard initialisation.

        Late-bound parameters can be used for parameters that depend on the
        specific data that is to be safeguarded.
        """

        return frozenset()

    @abstractmethod
    def check(
        self,
        data: np.ndarray[S, np.dtype[T]],
        prediction: np.ndarray[S, np.dtype[T]],
        *,
        late_bound: Bindings,
        where: Literal[True] | np.ndarray[S, np.dtype[np.bool]] = True,
    ) -> bool:
        """
        Check if the `prediction` array upholds the property enforced by this
        safeguard.

        Parameters
        ----------
        data : np.ndarray[S, np.dtype[T]]
            Original data array, relative to which the `prediction` is checked.
        prediction : np.ndarray[S, np.dtype[T]]
            Prediction for the `data` array.
        late_bound : Bindings
            Bindings for late-bound parameters, including for this safeguard.
        where : Literal[True] | np.ndarray[S, np.dtype[np.bool]]
            Only check at data points where the condition is [`True`][True].

        Returns
        -------
        ok : bool
            `True` if the check succeeded.
        """
        pass

    @abstractmethod
    def get_config(self) -> dict[str, JSON]:
        """
        Returns the configuration of the safeguard.

        The config must include a 'kind' field with the safeguard kind. All
        values must be compatible with JSON encoding.

        Returns
        -------
        config : dict[str, JSON]
            Configuration of the safeguard.
        """

        pass

    @classmethod
    def from_config(cls, config: dict[str, JSON]) -> Self:
        """
        Instantiate the safeguard from a configuration [`dict`][dict].

        Parameters
        ----------
        config : dict[str, JSON]
            Configuration of the safeguard.

        Returns
        -------
        safeguard : Self
            Instantiated safeguard.
        """

        return cls(**config)

    @override
    def __repr__(self) -> str:
        config: dict[str, JSON] = {
            k: v for k, v in self.get_config().items() if k != "kind"
        }

        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in config.items())})"
