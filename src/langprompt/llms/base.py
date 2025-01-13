from abc import ABC, abstractmethod
from typing import List, Iterator, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from tqdm import tqdm
from tenacity import stop_after_attempt, wait_exponential, Retrying

from ..base.message import Message
from ..base.response import Completion, merge_stream_completions
from ..cache import BaseCache
from ..store import BaseStore, DuckDBStore, ResponseRecord
from .ratelimiter import ThreadingRateLimiter


def _generate_key(messages: List[Message], model: str, **kwargs) -> str:
    """generate cache key"""
    cache_dict = {
        "messages": [msg.model_dump() for msg in messages],
        "model": model,
        **kwargs
    }
    cache_str = json.dumps(cache_dict, sort_keys=True)
    cache_key = hashlib.sha256(cache_str.encode()).hexdigest()
    return cache_key

class BaseLLM(ABC):
    """Abstract base class for all language model providers"""

    model: str = ""

    def __init__(
        self,
        cache: Optional[BaseCache] = None,
        store: Optional[BaseStore] = None,
        query_per_second: float = 0,
    ):
        """初始化 LLM

        Args:
            cache: 缓存实现，默认为 None（不使用缓存）
            store: Store 实例用于持久化追踪记录，默认为 None 时创建新的 DuckDBStore
        """
        self.cache = cache
        self.store = store if store is not None else DuckDBStore.connect()
        self.rate_limiter = ThreadingRateLimiter(query_per_second)

    def _get_from_cache(self, messages: List[Message], params: Dict[str, Any]) -> Optional[Completion]:
        """从缓存中获取结果"""
        if not self.cache:
            return None

        # 移除不应该影响缓存键的参数
        cache_kwargs = params.copy()
        for key in ['use_cache', 'cache_ttl', 'stream']:
            cache_kwargs.pop(key, None)

        key = _generate_key(
            messages=messages,
            model=self.__class__.__name__,
            **cache_kwargs
        )
        cached = self.cache.get(key)
        if cached:
            cached["cache_key"] = key
            return Completion(**cached)
        return None

    def _save_to_cache(self, messages: List[Message], completion: Completion, params: Dict[str, Any]):
        """保存结果到缓存"""
        if not self.cache:
            return

        # 移除不应该影响缓存键的参数
        cache_kwargs = params.copy()
        for key in ['use_cache', 'cache_ttl', 'stream']:
            cache_kwargs.pop(key, None)

        key = _generate_key(
            messages=messages,
            model=self.__class__.__name__,
            **cache_kwargs
        )
        self.cache.set(key, completion.model_dump())

    def _handle_cache(self, messages: List[Message], params: Dict[str, Any], use_cache: bool = True) -> Optional[Completion]:
        """Handle cache logic

        Returns:
            Optional[Completion]: Cached completion if exists and cache is enabled
        """
        if not use_cache:
            return None

        return self._get_from_cache(messages, params)

    def _handle_store(self, messages: List[Message], completion: Optional[Completion] = None, error: Optional[Exception] = None, params: Optional[Dict[str, Any]] = None):
        """Handle store logic for tracking responses"""
        try:
            if completion:
                entry = ResponseRecord.create(
                    response=completion,
                    messages=messages,
                    model=f"{self.__class__.__name__}/{self.model}",
                    properties=params,
                )
                self.store.add(entry)
            elif error:
                entry = ResponseRecord.create(
                    error=error,
                    messages=messages,
                    model=f"{self.__class__.__name__}/{self.model}",
                    properties=params,
                )
                self.store.add(entry)
        except Exception as e:
            print(f"Error saving to store: {e}")  # Log error but don't raise

    def chat_with_retry(self, messages: List[Message], **kwargs) -> Completion:
        retryer = Retrying(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=6))
        return retryer(self.chat, messages, **kwargs)

    def chat(
        self,
        messages: List[Message],
        use_cache: bool = True,
        **kwargs
    ) -> Completion:
        params = self._prepare_params(messages, **kwargs)
        params["stream"] = False
        # Try to get from cache first
        if cached := self._handle_cache(messages, params, use_cache):
            self._handle_store(messages, completion=cached, params=params)
            return cached

        try:
            # Get completion from LLM
            with self.rate_limiter:
                completion = self._chat(messages, params)

            # Save to cache if enabled
            if completion and use_cache:
                self._save_to_cache(messages, completion, params)

            # Save to store
            self._handle_store(messages, completion=completion, params=params)

            return completion
        except Exception as e:
            # Handle error
            self._handle_store(messages, error=e, params=params)
            raise e

    @abstractmethod
    def _chat(
        self,
        messages: List[Message],
        params: Dict[str, Any]
    ) -> Completion:
        pass

    def stream(
        self,
        messages: List[Message],
        use_cache: bool = True,
        **kwargs
    ) -> Iterator[Completion]:
        params = self._prepare_params(messages, **kwargs)
        params["stream"] = True
        if cached := self._handle_cache(messages, use_cache, params):
            self._handle_store(messages, completion=cached, params=params)
            yield cached
            return

        try:
            # 获取流式响应
            completions = []
            for completion in self._stream(messages, params):
                completions.append(completion)
                yield completion

            # 合并所有completion并保存到缓存和store
            if completions:
                merged_completion = merge_stream_completions(completions)
                if use_cache:
                    self._save_to_cache(messages, merged_completion, params)
                self._handle_store(messages, completion=merged_completion, params=params)

        except Exception as e:
            self._handle_store(messages, error=e, params=params)
            raise e

    @abstractmethod
    def _stream(
        self,
        messages: List[Message],
        params: Dict[str, Any]
    ) -> Iterator[Completion]:
        pass

    def batch(self, messages: List[List[Message]], batch_size: int = 10, enable_retry: bool = False, **kwargs) -> List[Optional[Completion]]:
        """Batch run with multi-thread with progress bar

        Returns:
            List[Optional[Completion]]: List of completions, failed requests will return error completion
        """
        results: List[Optional[Completion]] = [None] * len(messages)
        chat_func = self.chat if not enable_retry else self.chat_with_retry
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(chat_func, msg, **kwargs): idx
                for idx, msg in enumerate(messages)
            }

            with tqdm(total=len(messages), desc="Processing batch") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        # 创建错误 Completion
                        error_completion = Completion(
                            id="",
                            created=0,
                            content=f"Error: {str(e)}",
                            finish_reason="error",
                            model=f"{self.__class__.__name__}/{self.model}"
                        )
                        results[idx] = error_completion
                    finally:
                        pbar.update(1)

        return results

    @abstractmethod
    def _prepare_params(self, messages: List[Message], **kwargs) -> Dict[str, Any]:
        pass
