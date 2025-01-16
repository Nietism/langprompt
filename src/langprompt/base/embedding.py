from typing import Any, Dict, List, Optional

from pydantic import BaseModel

__all__ = ["Embedding", "EmbeddingUsage"]


class EmbeddingUsage(BaseModel):
    """Usage statistics for the embedding request."""

    prompt_tokens: int
    """The number of tokens in the prompt."""
    total_tokens: int
    """The total number of tokens in the embedding."""


class Embedding(BaseModel):
    """Response from an embedding model"""

    model: str
    """The model used for embedding."""

    usage: Optional[EmbeddingUsage] = None
    """Usage statistics for the embedding request."""

    data: List[List[float]]
    """The embedding vectors, order is same as input."""

    raw_response: Optional[Dict[str, Any]] = None
    """The raw response from the model."""

    cache_key: Optional[str] = None
    """The cache key for the response if hit cache"""
