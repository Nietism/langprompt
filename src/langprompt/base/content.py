import base64
import binascii
import re
from typing import List, Sequence, Literal
from pydantic import BaseModel

__all__ = ["TextPart", "ImagePart"]

# First pattern matches the entire image tag
IMAGE_TAG_PATTERN = re.compile(r"<\|image(.*?)\|>([^<]+)<\|/image\|>")

# Second pattern extracts attributes
ATTR_PATTERN = re.compile(r'media_type="([^"]+)"?\s*')


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


def decode_content(content: str) -> Sequence[TextPart | ImagePart]:
    """Decode the content into a list of TextPart / ImagePart / etc.

    Args:
        content: The content to decode.

    Returns:
        A list of TextPart/ImagePart/etc.
    """
    parts: List[TextPart | ImagePart] = []
    last_end = 0

    for match in IMAGE_TAG_PATTERN.finditer(content):
        # Handle text before image tag
        if match.start() > last_end:
            text = content[last_end : match.start()]
            if text.strip():
                parts.append(TextPart(type="text", text=text))

        # Extract attributes and base64 data
        attrs = match.group(1)  # Everything between <|image and |>
        base64_data = match.group(2)

        # Parse attributes with specific error handling
        attr_match = ATTR_PATTERN.match(attrs.strip())
        media_type = attr_match.group(1) if attr_match else None

        # Decode base64 with specific error handling
        try:
            binary_data = base64.b64decode(base64_data)
        except binascii.Error as e:
            raise ValueError(f"Invalid base64 image data: {str(e)}")

        # If media_type is None, detect it from binary data
        if media_type is None:
            try:
                media_type = detect_media_type(binary_data)
            except ValueError as e:
                raise ValueError(f"Failed to detect media type: {str(e)}")

        # Add image part
        parts.append(
            ImagePart(
                type="image",
                media_type=media_type,
                image=binary_data,
            )
        )

        last_end = match.end()

    # Handle remaining text
    if last_end < len(content):
        text = content[last_end:]
        if text.strip():
            parts.append(TextPart(type="text", text=text))

    return parts


def encode_content(content: str | Sequence[TextPart | ImagePart]) -> str:
    """Return the content as a string.

    Args:
        content: The content to encode, which can be a string or a list of TextPart/ImagePart/etc.

    Returns:
        The encoded content as a string.
    """
    if isinstance(content, str):
        return content
    content_str = ""
    for part in content:
        if isinstance(part, TextPart):
            content_str += part.text
        elif isinstance(part, ImagePart):
            # <|image media_type="image/jpeg"|>image_bytes_base64<|/image|>
            content_str += f'<|image media_type="{part.media_type}"|>{base64.b64encode(part.image).decode("utf-8")}<|/image|>'
    return content_str


def detect_media_type(image_data: bytes) -> str:
    """Detect the media type of the binary data.

    Args:
        image_data: The binary data to detect.

    Returns:
        The media type of the binary data.
    """
    if image_data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    elif image_data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    elif image_data.startswith(b"GIF87a") or image_data.startswith(b"GIF89a"):
        return "image/gif"
    elif image_data.startswith(b"RIFF") and image_data[8:12] == b"WEBP":
        return "image/webp"
    elif image_data[4:12] in (
        b"ftypmif1",
        b"ftypmsf1",
        b"ftypheic",
        b"ftypheix",
        b"ftyphevc",
        b"ftyphevx",
    ):
        subtype = image_data[8:12]
        if subtype in (b"heic", b"heix"):
            return "image/heic"
        elif subtype in (b"mif1", b"msf1", b"hevc", b"hevx"):
            return "image/heif"
    raise ValueError("Unsupported image type")
