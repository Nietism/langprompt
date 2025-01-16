import hashlib
import json
import logging
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterator, List, Optional

from tenacity import Retrying, stop_after_attempt, wait_exponential
from tqdm import tqdm  # type: ignore

from ..base.message import Message
from ..base.ratelimiter import ThreadingRateLimiter
from ..base.response import Completion, merge_stream_completions
from ..cache import BaseCache
from ..store import BaseStore, DuckDBStore, ResponseRecord

logger = logging.getLogger(__name__)


def _generate_key(model: str, params: Dict[str, Any]) -> str:
    """generate cache key"""
    cache_dict = {"model": model, "params": json.dumps(params, sort_keys=True)}
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
        """Initialize LLM

        Args:
            cache: Cache implementation, defaults to None (no caching)
            store: Store instance for persistent tracking records, defaults to None which creates a new DuckDBStore
        """
        self.cache = cache
        self.store = store
        self.rate_limiter = ThreadingRateLimiter(query_per_second)

    def _get_from_cache(
        self, params: Dict[str, Any], use_cache: bool = True
    ) -> Optional[Completion]:
        """Get results from cache"""
        if not self.cache or not use_cache:
            return None

        # Remove parameters that should not affect the cache key
        cache_kwargs = params.copy()
        for key in ["use_cache", "stream"]:
            cache_kwargs.pop(key, None)

        key = _generate_key(model=self.__class__.__name__, params=cache_kwargs)
        cached = self.cache.get(key)
        if cached:
            cached["cache_key"] = key
            return Completion(**cached)
        return None

    def _save_to_cache(self, completion: Completion, params: Dict[str, Any]):
        """Save results to cache"""
        if not self.cache:
            return

        # Remove parameters that should not affect the cache key
        cache_kwargs = params.copy()
        for key in ["use_cache", "stream"]:
            cache_kwargs.pop(key, None)

        key = _generate_key(model=self.__class__.__name__, params=cache_kwargs)
        self.cache.set(key, completion.model_dump())

    def _handle_store(
        self,
        messages: List[Message],
        completion: Optional[Completion] = None,
        error: Optional[Exception] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        """Handle store logic for tracking responses"""
        if not self.store:
            return

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
            logger.error(f"Error saving to store: {e}")

    def _with_retry(self, func: Callable[[], Any]) -> Any:
        """Generic retry wrapper"""
        retryer = Retrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=6),
        )
        return retryer(func)

    def chat(
        self,
        messages: List[Message],
        use_cache: bool = True,
        enable_retry: bool = False,
        **kwargs,
    ) -> Completion:
        params = self._prepare_params(messages, **kwargs)
        params["stream"] = False

        if use_cache:
            if cached := self._get_from_cache(params, use_cache):
                self._handle_store(messages, completion=cached, params=params)
                return cached

        def _do_chat():
            try:
                with self.rate_limiter:
                    completion = self._chat(params)
                    if completion and use_cache:
                        self._save_to_cache(completion, params)
                    self._handle_store(messages, completion=completion, params=params)
                    return completion
            except Exception as e:
                self._handle_store(messages, error=e, params=params)
                raise e

        return self._with_retry(_do_chat) if enable_retry else _do_chat()

    @abstractmethod
    def _chat(self, params: Dict[str, Any]) -> Completion:
        pass

    def stream(
        self, messages: List[Message], use_cache: bool = True, **kwargs
    ) -> Iterator[Completion]:
        params = self._prepare_params(messages, **kwargs)
        params["stream"] = True
        if cached := self._get_from_cache(params, use_cache):
            self._handle_store(messages, completion=cached, params=params)
            yield cached
            return

        try:
            completions = []
            for completion in self._stream(params):
                completions.append(completion)
                yield completion

            # merge all completions and save to cache and store
            if completions:
                merged_completion = merge_stream_completions(completions)
                if use_cache:
                    self._save_to_cache(merged_completion, params)
                self._handle_store(
                    messages, completion=merged_completion, params=params
                )

        except Exception as e:
            self._handle_store(messages, error=e, params=params)
            raise e

    @abstractmethod
    def _stream(
        self, params: Dict[str, Any]
    ) -> Iterator[Completion]:
        pass

    def batch(
        self,
        messages: List[List[Message]],
        batch_size: int = 10,
        enable_retry: bool = False,
        **kwargs,
    ) -> List[Completion]:
        """Batch run with multi-thread with progress bar

        Returns:
            List[Completion]: List of completions, failed requests will return error completion
        """
        # Use dict to store results
        results = {}
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {
                executor.submit(self.chat, msg, enable_retry=enable_retry, **kwargs): idx
                for idx, msg in enumerate(messages)
            }

            with tqdm(total=len(messages), desc="Processing batch") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result()
                    except Exception as e:
                        # Create fake completion for error
                        results[idx] = Completion(
                            id="",
                            created=0,
                            content=f"Error: {str(e)}",
                            finish_reason="error",
                            model=f"{self.__class__.__name__}/{self.model}",
                        )
                    finally:
                        pbar.update(1)

        # Convert dict to ordered list
        return [results[i] for i in range(len(messages))]

    @abstractmethod
    def _prepare_params(self, messages: List[Message], **kwargs) -> Dict[str, Any]:
        pass
