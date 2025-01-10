from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Iterator

from ..base.response import Completion

OutputType = TypeVar('OutputType')

class OutputParser(ABC, Generic[OutputType]):
    """Output parser base class"""
    @abstractmethod
    def parse(self, completion: Completion) -> OutputType:
        pass

    @abstractmethod
    def stream_parse(self, completion: Iterator[Completion]) -> Iterator[OutputType]:
        pass


class TextOutputParser(OutputParser[str]):
    def parse(self, completion: Completion) -> str:
        return completion.content or ""

    def stream_parse(self, completion: Iterator[Completion]) -> Iterator[str]:
        for chunk in completion:
            if chunk.content:
                yield chunk.content
