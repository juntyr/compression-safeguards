from abc import ABC, abstractmethod

import numpy as np


class Guardrail(ABC):
    kind: str

    @abstractmethod
    def check(self, data: np.ndarray, decoded: np.ndarray) -> bool:
        pass

    @abstractmethod
    def encode_correction(self, data: np.ndarray, decoded: np.ndarray) -> bytes:
        pass

    @abstractmethod
    def apply_correction(self, decoded: np.ndarray, correction: bytes) -> np.ndarray:
        pass

    @abstractmethod
    def get_config(self) -> dict:
        pass

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)
