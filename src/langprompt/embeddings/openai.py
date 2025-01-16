"""A provider implementation for OpenAI API"""

from typing import Any, Dict, List, Optional

try:
    import openai
except ImportError:
    raise ImportError(
        "OpenAI Python SDK is not installed. Please install it using `pip install openai`."
    )


from ..base.embedding import Embedding, EmbeddingUsage
from ..cache import BaseCache
from .base import BaseEmbedding


__all__ = ["OpenAIEmbedding"]


class OpenAIEmbedding(BaseEmbedding):
    """A simplified OpenAI embedding provider implementation"""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        cache: Optional[BaseCache] = None,
        query_per_second: float = 0,
        **kwargs,
    ):
        """Initialize OpenAI provider

        Args:
            model: Model to use
            cache: Cache implementation
            **kwargs: Additional arguments to pass to the OpenAI client
        """
        super().__init__(cache=cache, query_per_second=query_per_second)
        self.client = openai.OpenAI(**kwargs)
        self.model = model

    def _embed(
        self,
        inputs: List[str],
        params: Dict[str, Any],
    ) -> Embedding:
        """Send a embedding request

        Args:
            inputs: List of input strings
            params: Additional parameters for the embedding request

        Returns:
            Embedding object
        """

        # Get response from OpenAI
        raw_response = self.client.embeddings.create(**params)

        # Create Completion object
        data_pairs = [(e.index, e.embedding) for e in raw_response.data]
        # sort by index
        data_pairs.sort(key=lambda x: x[0])
        assert len(data_pairs) == len(inputs)
        return Embedding(
            model=raw_response.model,
            usage=EmbeddingUsage(
                prompt_tokens=raw_response.usage.prompt_tokens,
                total_tokens=raw_response.usage.total_tokens,
            ),
            data=[pair[1] for pair in data_pairs],
            raw_response=raw_response.model_dump(),
        )

    def _prepare_params(self, inputs: List[str], **kwargs) -> Dict[str, Any]:
        """Prepare parameters for OpenAI API"""
        params = {
            "model": self.model,
            "input": inputs,
        }
        params.update(kwargs)
        return params
