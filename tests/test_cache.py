"""Tests for LLM cache functionality"""
from typing import List, Iterator
from datetime import datetime
import time
from unittest.mock import patch

from langprompt.llms.base import BaseLLM
from langprompt.base.message import Message
from langprompt.base.response import Completion
from langprompt.cache import MemoryCache, SQLiteCache

class MockLLM(BaseLLM):
    """A mock LLM implementation for testing cache"""
    def __init__(self):
        super().__init__()
        self.call_count = 0

    def _chat(self, messages: List[Message], **kwargs) -> Completion:
        response = Completion(
            content=f"Response #{self.call_count + 1}",
            role="assistant",
            id=f"test_id_{self.call_count + 1}",
            created=int(datetime.now().timestamp()),
            model="test_model"
        )
        self.call_count += 1
        return response

    def _stream(self, messages: List[Message], **kwargs) -> Iterator[Completion]:
        response = Completion(
            content=f"Response #{self.call_count + 1}",
            role="assistant",
            id=f"test_id_{self.call_count + 1}",
            created=int(datetime.now().timestamp()),
            model="test_model"
        )
        self.call_count += 1
        yield response

def test_memory_cache():
    """Test that MemoryCache works correctly with LLM"""
    llm = MockLLM()
    cache = MemoryCache()
    llm.cache = cache

    messages = [Message(role="user", content="Hello")]

    # First call should hit the LLM
    response1 = llm.chat(messages=messages)
    assert llm.call_count == 1
    assert response1.content == "Response #1"

    # Second call with same input should hit cache
    response2 = llm.chat(messages=messages)
    assert llm.call_count == 1  # Call count shouldn't increase
    assert response2.content == "Response #1"  # Should get same response

    # Different input should hit LLM again
    new_messages = [Message(role="user", content="Different message")]
    response3 = llm.chat(messages=new_messages)
    assert llm.call_count == 2
    assert response3.content == "Response #2"

def test_memory_cache_with_ttl():
    """Test that MemoryCache respects TTL"""
    import time
    from unittest.mock import patch

    llm = MockLLM()
    cache = MemoryCache(ttl=1)  # 1 second TTL
    llm.cache = cache

    messages = [Message(role="user", content="Hello")]

    # First call
    response1 = llm.chat(messages=messages)
    assert llm.call_count == 1

    # Immediate second call should hit cache
    response2 = llm.chat(messages=messages)
    assert llm.call_count == 1

    # Mock time.time() to simulate TTL expiration
    with patch('time.time', return_value=time.time() + 2):
        # Call after TTL should miss cache
        response3 = llm.chat(messages=messages)
        assert llm.call_count == 2

def test_sqlite_cache():
    """Test that SQLiteCache works correctly with LLM"""
    import os

    db_path = "test_cache.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    llm = MockLLM()
    cache = SQLiteCache(db_path=db_path)
    llm.cache = cache

    messages = [Message(role="user", content="Hello")]

    # First call should hit the LLM
    response1 = llm.chat(messages=messages)
    assert llm.call_count == 1

    # Second call with same input should hit cache
    response2 = llm.chat(messages=messages)
    assert llm.call_count == 1

    # Clean up
    os.remove(db_path)

def test_cache_with_different_params():
    """Test that cache considers all parameters when generating cache key"""
    llm = MockLLM()
    cache = MemoryCache()
    llm.cache = cache

    messages = [Message(role="user", content="Hello")]

    # Call with different parameters should result in cache misses
    response1 = llm.chat(messages=messages, temperature=0.5)
    assert llm.call_count == 1

    response2 = llm.chat(messages=messages, temperature=0.7)
    assert llm.call_count == 2

    response3 = llm.chat(messages=messages, temperature=0.5)
    assert llm.call_count == 2  # Should hit cache for same params