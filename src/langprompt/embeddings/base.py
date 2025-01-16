import hashlib
import json
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from typing import Any, Dict, List, Optional, Callable

from tenacity import Retrying, stop_after_attempt, wait_exponential
from tqdm import tqdm  # type: ignore

from ..base.embedding import Embedding
from ..base.ratelimiter import ThreadingRateLimiter
from ..cache import BaseCache

__all__ = ["BaseEmbedding"]


def _generate_key(model: str, params: Dict[str, Any]) -> str:
    """generate cache key"""
    cache_dict = {"model": model, "params": json.dumps(params, sort_keys=True)}
    cache_str = json.dumps(cache_dict, sort_keys=True)
    cache_key = hashlib.sha256(cache_str.encode()).hexdigest()
    return cache_key


class BaseEmbedding(ABC):
    """Abstract base class for all embedding model providers"""

    model: str = ""

    def __init__(
        self,
        cache: Optional[BaseCache] = None,
        query_per_second: float = 0,
    ):
        """Initialize Embedding

        Args:
            cache: Cache implementation, defaults to None (no cache)
        """
        self.cache = cache
        self.rate_limiter = ThreadingRateLimiter(query_per_second)

    def _get_from_cache(self, params: Dict[str, Any]) -> Optional[Embedding]:
        """Get result from cache"""
        if not self.cache:
            return None
        key = _generate_key(model=self.__class__.__name__, params=params.copy())
        cached = self.cache.get(key)
        if cached:
            cached["cache_key"] = key
            return Embedding(**cached)
        return None

    def _save_to_cache(self, embedding: Embedding, params: Dict[str, Any]):
        """Save result to cache"""
        if not self.cache:
            return
        key = _generate_key(model=self.__class__.__name__, params=params.copy())
        self.cache.set(key, embedding.model_dump())

    def _with_retry(self, func: Callable[[], Any]) -> Any:
        """Generic retry wrapper"""
        retryer = Retrying(
            reraise=True,
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=2, max=6),
        )
        return retryer(func)

    def embed(
        self,
        inputs: List[str],
        use_cache: bool = True,
        enable_retry: bool = False,
        **kwargs,
    ) -> Embedding:
        """Embed text with optional caching and retry"""
        params = self._prepare_params(inputs, **kwargs)

        if use_cache:
            if cached := self._get_from_cache(params):
                return cached

        def _do_embed():
            with self.rate_limiter:
                embedding = self._embed(inputs, params)
                if embedding and use_cache:
                    self._save_to_cache(embedding, params)
                return embedding

        return self._with_retry(_do_embed) if enable_retry else _do_embed()

    @abstractmethod
    def _embed(self, inputs: List[str], params: Dict[str, Any]) -> Embedding:
        pass

    @abstractmethod
    def _prepare_params(self, inputs: List[str], **kwargs) -> Dict[str, Any]:
        pass

    def batch_embed(
        self,
        inputs: List[str],
        batch_size: int = 10,
        per_batch: int = 10,
        enable_retry: bool = False,
        **kwargs,
    ) -> List[List[float]]:
        """Batch run with multi-thread with progress bar

        Returns:
            List[Optional[Embedding]]: List of embeddings, failed requests will return error embedding
        """
        inputs_list = [
            inputs[i : i + per_batch] for i in range(0, len(inputs), per_batch)
        ]
        results = [None] * len(inputs_list)
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures: Dict[Future, int] = {
                executor.submit(
                    self.embed, inputs, enable_retry=enable_retry, **kwargs
                ): idx
                for idx, inputs in enumerate(inputs_list)
            }

            with tqdm(total=len(inputs_list), desc="Processing batch") as pbar:
                for future in as_completed(futures):
                    idx = futures[future]
                    try:
                        results[idx] = future.result().data
                    except Exception as e:
                        raise e
                    finally:
                        pbar.update(1)
        rets: List[List[float]] = []
        for result in results:
            assert result is not None
            for item in result:
                rets.append(item)
        assert len(rets) == len(inputs)
        return rets
