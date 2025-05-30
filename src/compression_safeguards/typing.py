"""
Commonly used type variables.
"""

from typing import TypeVar

import numpy as np

C = TypeVar("C", bound=np.unsignedinteger)
""" The numpy data type for safeguard corrections. """

F = TypeVar("F", bound=np.floating)
""" Any numpy [`floating`][numpy.floating] data type. """

T = TypeVar("T", bound=np.number)
""" Any numpy [`number`][numpy.number] data type. """

S = TypeVar("S", bound=tuple[int, ...])
""" Any array shape. """

U = TypeVar("U", bound=np.unsignedinteger)
""" Any numpy [`unsignedinteger`][numpy.unsignedinteger] data type. """
