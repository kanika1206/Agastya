from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

I = TypeVar("I")
O = TypeVar("O")


class Stage(ABC, Generic[I, O]):
    name: str = "stage"

    @abstractmethod
    def process(self, item: I) -> O:
        ...
