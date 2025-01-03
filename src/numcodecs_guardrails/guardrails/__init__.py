__all__ = ["Guardrail"]

from abc import ABC, abstractmethod

import numpy as np


class Guardrail(ABC):
    kind: str
    _priority: int

    @abstractmethod
    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        return np.all(self.check_elementwise(data, decoded))

    @abstractmethod
    def get_config(self) -> dict:
        pass

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)
