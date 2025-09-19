"""
Commonly used type variables.
"""

__all__ = ["C", "F", "S", "Si", "T", "Ti", "U", "JSON"]

from typing import TypeAlias, TypeVar

import numpy as np

C = TypeVar("C", bound=np.unsignedinteger, covariant=True)
""" The numpy data type for safeguard corrections (covariant). """

F = TypeVar("F", bound=np.floating, covariant=True)
""" Any numpy [`floating`][numpy.floating]-point data type (covariant). """

S = TypeVar("S", bound=tuple[int, ...], covariant=True)
""" Any array shape (covariant). """

Si = TypeVar("Si", bound=tuple[int, ...])
""" Any array shape (invariant). """

T = TypeVar("T", bound=np.number, covariant=True)
""" Any numpy [`number`][numpy.number] data type (covariant). """

Ti = TypeVar("Ti", bound=np.number)
""" Any numpy [`number`][numpy.number] data type (invariant). """

U = TypeVar("U", bound=np.unsignedinteger, covariant=True)
""" Any numpy [`unsignedinteger`][numpy.unsignedinteger] data type (covariant). """

JSON: TypeAlias = None | int | float | str | bool | list["JSON"] | dict[str, "JSON"]
""" Types that are valid JSON and can be encoded with [`json.dumps`][json.dumps]. """
