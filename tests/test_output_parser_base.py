"""Tests for output parser base"""

from datetime import datetime
from typing import Iterator

import pytest

from langprompt.base.response import Completion
from langprompt.output_parser.base import OutputParser, TextOutputParser


def test_text_output_parser_parse():
    """Test TextOutputParser parse method"""
    parser = TextOutputParser()
    completion = Completion(
        content="Hello",
        role="assistant",
        id="test_id",
        created=int(datetime.now().timestamp()),
        model="test_model",
    )
    assert parser.parse(completion) == "Hello"

    # Test with empty content
    completion = Completion(
        content="",
        role="assistant",
        id="test_id",
        created=int(datetime.now().timestamp()),
        model="test_model",
    )
    assert parser.parse(completion) == ""

    # Test with None content
    completion = Completion(
        content=None,
        role="assistant",
        id="test_id",
        created=int(datetime.now().timestamp()),
        model="test_model",
    )
    assert parser.parse(completion) == ""


def test_text_output_parser_stream_parse():
    """Test TextOutputParser stream_parse method"""
    parser = TextOutputParser()

    # Create a list of completion chunks
    created_time = int(datetime.now().timestamp())
    chunks = [
        Completion(
            content="Hello",
            role="assistant",
            id="chunk1",
            created=created_time,
            model="test_model",
        ),
        Completion(
            content=" ",
            role="assistant",
            id="chunk2",
            created=created_time,
            model="test_model",
        ),
        Completion(
            content="World",
            role="assistant",
            id="chunk3",
            created=created_time,
            model="test_model",
        ),
        Completion(
            content=None,
            role="assistant",
            id="chunk4",
            created=created_time,
            model="test_model",
        ),
        Completion(
            content="!",
            role="assistant",
            id="chunk5",
            created=created_time,
            model="test_model",
        ),
    ]

    # Convert list to iterator
    def chunk_iterator() -> Iterator[Completion]:
        for chunk in chunks:
            yield chunk

    result = list(parser.stream_parse(chunk_iterator()))
    assert result == ["Hello", " ", "World", "!"]


def test_abstract_output_parser():
    """Test abstract OutputParser class"""
    # Verify that we can't instantiate the abstract class
    with pytest.raises(TypeError):
        OutputParser()  # type: ignore

    # Test that we must implement abstract methods
    class IncompleteParser(OutputParser):
        pass

    with pytest.raises(TypeError):
        IncompleteParser()  # type: ignore

    # Test that we can create a concrete implementation
    class ConcreteParser(OutputParser[str]):
        def parse(self, completion: Completion) -> str:
            return "test"

        def stream_parse(self, completion: Iterator[Completion]) -> Iterator[str]:
            yield "test"

    parser = ConcreteParser()
    assert isinstance(parser, OutputParser)
