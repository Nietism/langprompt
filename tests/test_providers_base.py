"""Tests for base provider"""

from datetime import datetime
from typing import Any, Dict, Iterator, List

import pytest

from langprompt.base.message import Message
from langprompt.base.response import Completion
from langprompt.llms.base import BaseLLM


def test_base_provider_abstract():
    """Test that BaseProvider is abstract and cannot be instantiated"""
    with pytest.raises(TypeError):
        BaseLLM()  # type: ignore


def test_base_provider_implementation():
    """Test that BaseProvider can be implemented"""

    class TestProvider(BaseLLM):
        def _chat(self, messages: List[Message], params: Dict[str, Any]) -> Completion:
            return Completion(
                content="test",
                role="assistant",
                id="test_id",
                created=int(datetime.now().timestamp()),
                model="test_model",
            )

        def _stream(
            self, messages: List[Message], params: Dict[str, Any]
        ) -> Iterator[Completion]:
            yield Completion(
                content="test",
                role="assistant",
                id="test_id",
                created=int(datetime.now().timestamp()),
                model="test_model",
            )

        def _prepare_params(self, messages: List[Message], **kwargs) -> Dict[str, Any]:
            return kwargs

    # Should be able to instantiate concrete implementation
    provider = TestProvider()
    assert isinstance(provider, BaseLLM)


def test_incomplete_provider():
    """Test that incomplete implementation raises TypeError"""

    class IncompleteProvider(BaseLLM):
        pass

    with pytest.raises(TypeError):
        IncompleteProvider()  # type: ignore


def test_partial_implementation():
    """Test that partial implementation raises TypeError"""

    class PartialProvider(BaseLLM):
        def chat(self, messages: List[Message], **kwargs) -> Completion:
            return Completion(
                content="test",
                role="assistant",
                id="test_id",
                created=int(datetime.now().timestamp()),
                model="test_model",
            )

    with pytest.raises(TypeError):
        PartialProvider()  # type: ignore
