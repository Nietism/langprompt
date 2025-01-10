from abc import ABC, abstractmethod
from typing import List, Iterator, Optional

import hashlib
import json
from ..base.message import Message
from ..base.response import Completion
from ..cache import BaseCache
from ..trace import BaseStore, DuckDBStore, ResponseRecord


def _generate_key(messages: list, model: str, **kwargs) -> str:
    """generate cache key"""
    cache_dict = {
        "messages": messages,
        "model": model,
        **kwargs
    }
    cache_str = json.dumps(cache_dict, sort_keys=True)
    return hashlib.sha256(cache_str.encode()).hexdigest()

class BaseLLM(ABC):
    """Abstract base class for all language model providers"""

    model: str = ""

    def __init__(
        self,
        cache: Optional[BaseCache] = None,
        store: Optional[BaseStore] = None,
    ):
        """初始化 LLM

        Args:
            cache: 缓存实现，默认为 None（不使用缓存）
            store: Store 实例用于持久化追踪记录，默认为 None 时创建新的 DuckDBStore
        """
        self.cache = cache
        self.store = store if store is not None else DuckDBStore.connect()

    def _get_from_cache(self, messages: List[Message], **kwargs) -> Optional[Completion]:
        """从缓存中获取结果"""
        if not self.cache:
            return None

        # 移除不应该影响缓存键的参数
        cache_kwargs = kwargs.copy()
        cache_kwargs.pop('use_cache', None)
        cache_kwargs.pop('cache_ttl', None)

        key = _generate_key(
            messages=[msg.model_dump() for msg in messages],
            model=self.__class__.__name__,
            **cache_kwargs
        )
        cached = self.cache.get(key)
        if cached:
            return Completion(**cached)
        return None

    def _save_to_cache(self, messages: List[Message], completion: Completion, **kwargs):
        """保存结果到缓存"""
        if not self.cache:
            return

        # 移除不应该影响缓存键的参数
        cache_kwargs = kwargs.copy()
        cache_kwargs.pop('use_cache', None)
        cache_kwargs.pop('cache_ttl', None)

        key = _generate_key(
            messages=[msg.model_dump() for msg in messages],
            model=self.__class__.__name__,
            **cache_kwargs
        )
        self.cache.set(key, completion.model_dump())

    def chat(
        self,
        messages: List[Message],
        use_cache: bool = True,
        **kwargs
    ) -> Completion:
        """Send a chat completion request

        Args:
            messages: List of Message objects
            use_cache: Use cache or not
            **kwargs: Additional arguments to pass to the API

        Returns:
            Completion object
        """
        if use_cache:
            if cache := self._get_from_cache(messages, **kwargs):
                return cache

        try:
            completion = self._chat(messages, **kwargs)

            # 记录追踪信息
            entry = ResponseRecord.create(
                response=completion,
                messages=messages,
                model=f"{self.__class__.__name__}/{self.model}" # use class_name/model as model name
            )
            self.store.add(entry)

            if use_cache:
                self._save_to_cache(messages, completion, **kwargs)

            return completion

        except Exception as e:
            entry = ResponseRecord.create(
                error=e,
                messages=messages,
                model=f"{self.__class__.__name__}/{self.model}" # use class_name/model as model name
            )
            self.store.add(entry)
            raise e

    @abstractmethod
    def _chat(
        self,
        messages: List[Message],
        **kwargs
    ) -> Completion:
        pass

    def stream(
        self,
        messages: List[Message],
        **kwargs
    ) -> Iterator[Completion]:
        """Stream chat completion request

        Args:
            messages: List of Message objects
            use_cache: Use cache or not
            cache_ttl: Cache expiration time (seconds)
            **kwargs: Additional arguments to pass to the API

        Returns:
            Iterator of Completion objects
        """
        # TODO: stream support cache
        return self._stream(messages, **kwargs)

    @abstractmethod
    def _stream(
        self,
        messages: List[Message],
        **kwargs
    ) -> Iterator[Completion]:
        pass
