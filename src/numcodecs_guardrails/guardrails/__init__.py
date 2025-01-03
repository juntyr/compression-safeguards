__all__ = ["Guardrail"]

from abc import ABC, abstractmethod

import numpy as np


class Guardrail(ABC):
    kind: str
    _priority: int

    @abstractmethod
    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        pass

    @abstractmethod
    def get_config(self) -> dict:
        pass

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

    def __repr__(self) -> str:
        config = {k: v for k, v in self.get_config().items() if k != "kind"}

        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in config.items())})"
