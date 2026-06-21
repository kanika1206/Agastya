from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TIn = TypeVar("TIn")
TOut = TypeVar("TOut")


class Stage(ABC, Generic[TIn, TOut]):
    name: str = "stage"

    @abstractmethod
    def process(self, item: TIn) -> TOut:
        ...
