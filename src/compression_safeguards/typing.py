"""
Commonly used type variables.
"""

__all__ = ["C", "F", "S", "T", "U"]

from typing import TypeVar

import numpy as np

C = TypeVar("C", bound=np.unsignedinteger)
""" The numpy data type for safeguard corrections. """

F = TypeVar("F", bound=np.floating)
""" Any numpy [`floating`][numpy.floating] data type. """

S = TypeVar("S", bound=tuple[int, ...])
""" Any array shape. """

T = TypeVar("T", bound=np.number)
""" Any numpy [`number`][numpy.number] data type. """

U = TypeVar("U", bound=np.unsignedinteger)
""" Any numpy [`unsignedinteger`][numpy.unsignedinteger] data type. """
