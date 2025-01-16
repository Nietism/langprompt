"""This module contains the base class for llm message parameters.

from: https://github.com/Mirascope/mirascope/blob/main/mirascope/core/base/message_param.py
"""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, field_serializer

from .content import encode_content, TextPart, ImagePart

__all__ = ["Message"]


class Message(BaseModel):
    """A base class for llm message parameters.

    Attributes:
        role: The role of the message
        content: The content of the message
    """

    # Only OpenAI supports developer role
    role: Literal["developer", "system", "user", "assistant", "tool"]
    content: str | Sequence[TextPart | ImagePart]

    @property
    def content_str(self) -> str:
        """Return the content as a string."""
        return encode_content(self.content)

    @field_serializer("content")
    def serialize_content(self, content: str | Sequence[TextPart | ImagePart], _info):
        return encode_content(content)
