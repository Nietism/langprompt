"""This module contains the base class for llm message parameters.

from: https://github.com/Mirascope/mirascope/blob/main/mirascope/core/base/message_param.py
"""
import base64
from collections.abc import Sequence
from typing import Literal
from pydantic import BaseModel

class TextPart(BaseModel):
    """A content part for text.

    Attributes:
        type: Always "text"
        text: The text content
    """

    type: Literal["text"]
    text: str


class ImagePart(BaseModel):
    """A content part for images.

    Attributes:
        type: Always "image"
        media_type: The media type (e.g. image/jpeg)
        image: The raw image bytes
    """

    type: Literal["image"]
    media_type: str
    image: bytes
    # TODO: support OpenAI detail field: https://platform.openai.com/docs/guides/vision#low-or-high-fidelity-image-understanding


class Message(BaseModel):
    """A base class for llm message parameters.

    Attributes:
        role: The role of the message
        content: The content of the message
    """

    # Only OpenAI supports developer role
    role: Literal["developer", "system", "user", "assistant", "tool"]
    content: (
        str
        | Sequence[TextPart | ImagePart]
    )

    @property
    def content_str(self) -> str:
        """Return the content as a string."""
        if isinstance(self.content, str):
            return self.content
        content_str = ""
        for part in self.content:
            if isinstance(part, TextPart):
                content_str += part.text
            elif isinstance(part, ImagePart):
                content_str += f'<|image media_type="{part.media_type}"|>{base64.b64encode(part.image).decode("utf-8")}<|/image|>'
        return content_str

