import base64
import pytest
from langprompt.base.content import decode_content, TextPart, ImagePart, detect_media_type

def test_decode_content_text_only():
    """Test decoding content with only text."""
    content = "Hello, world!"
    result = decode_content(content)
    assert len(result) == 1
    assert isinstance(result[0], TextPart)
    assert result[0].text == "Hello, world!"

def test_decode_content_image_only():
    """Test decoding content with only an image."""
    # Create a dummy JPEG image (just the header)
    jpeg_header = b"\xff\xd8\xff"
    base64_data = base64.b64encode(jpeg_header).decode()
    content = f'<|image media_type="image/jpeg"|>{base64_data}<|/image|>'

    result = decode_content(content)
    assert len(result) == 1
    assert isinstance(result[0], ImagePart)
    assert result[0].media_type == "image/jpeg"
    assert result[0].image == jpeg_header

def test_decode_content_mixed():
    """Test decoding content with both text and image."""
    jpeg_header = b"\xff\xd8\xff"
    base64_data = base64.b64encode(jpeg_header).decode()
    content = f'Text before <|image media_type="image/jpeg"|>{base64_data}<|/image|> Text after'

    result = decode_content(content)
    assert len(result) == 3
    assert isinstance(result[0], TextPart)
    assert result[0].text.strip() == "Text before"
    assert isinstance(result[1], ImagePart)
    assert result[1].media_type == "image/jpeg"
    assert result[1].image == jpeg_header
    assert isinstance(result[2], TextPart)
    assert result[2].text.strip() == "Text after"

def test_decode_content_auto_detect_media_type():
    """Test auto-detection of media type when not specified."""
    # Test JPEG detection
    jpeg_header = b"\xff\xd8\xff"
    base64_data = base64.b64encode(jpeg_header).decode()
    content = f'<|image|>{base64_data}<|/image|>'

    result = decode_content(content)
    assert len(result) == 1
    assert isinstance(result[0], ImagePart)
    assert result[0].media_type == "image/jpeg"

def test_decode_content_invalid_image():
    """Test handling of invalid image data."""
    invalid_data = base64.b64encode(b"invalid").decode()
    content = f'<|image|>{invalid_data}<|/image|>'

    with pytest.raises(ValueError, match="Unsupported image type"):
        decode_content(content)

def test_decode_content_multiple_images():
    """Test decoding content with multiple images."""
    # Create dummy JPEG and PNG headers
    jpeg_header = b"\xff\xd8\xff"
    png_header = b"\x89PNG\r\n\x1a\n"

    jpeg_base64 = base64.b64encode(jpeg_header).decode()
    png_base64 = base64.b64encode(png_header).decode()

    content = (
        f'Start text '
        f'<|image media_type="image/jpeg"|>{jpeg_base64}<|/image|> '
        f'Middle text '
        f'<|image media_type="image/png"|>{png_base64}<|/image|> '
        f'End text'
    )

    result = decode_content(content)
    assert len(result) == 5

    # Verify parts sequence: text -> jpeg -> text -> png -> text
    assert isinstance(result[0], TextPart)
    assert result[0].text.strip() == "Start text"

    assert isinstance(result[1], ImagePart)
    assert result[1].media_type == "image/jpeg"
    assert result[1].image == jpeg_header

    assert isinstance(result[2], TextPart)
    assert result[2].text.strip() == "Middle text"

    assert isinstance(result[3], ImagePart)
    assert result[3].media_type == "image/png"
    assert result[3].image == png_header

    assert isinstance(result[4], TextPart)
    assert result[4].text.strip() == "End text"

def test_detect_media_type_jpeg():
    """Test JPEG image type detection."""
    jpeg_data = b"\xff\xd8\xff" + b"dummy data"
    assert detect_media_type(jpeg_data) == "image/jpeg"

def test_detect_media_type_png():
    """Test PNG image type detection."""
    png_data = b"\x89PNG\r\n\x1a\n" + b"dummy data"
    assert detect_media_type(png_data) == "image/png"

def test_detect_media_type_gif():
    """Test GIF image type detection."""
    gif87_data = b"GIF87a" + b"dummy data"
    gif89_data = b"GIF89a" + b"dummy data"
    assert detect_media_type(gif87_data) == "image/gif"
    assert detect_media_type(gif89_data) == "image/gif"

def test_detect_media_type_webp():
    """Test WebP image type detection."""
    webp_data = b"RIFF" + b"size" + b"WEBP" + b"dummy data"
    assert detect_media_type(webp_data) == "image/webp"

def test_detect_media_type_heic():
    """Test HEIC image type detection."""
    heic_data = b"size" + b"ftypheic" + b"dummy data"
    heix_data = b"size" + b"ftypheix" + b"dummy data"
    assert detect_media_type(heic_data) == "image/heic"
    assert detect_media_type(heix_data) == "image/heic"

def test_detect_media_type_heif():
    """Test HEIF image type detection."""
    heif_types = [b"mif1", b"msf1", b"hevc", b"hevx"]
    for subtype in heif_types:
        data = b"size" + b"ftyp" + subtype + b"dummy data"
        assert detect_media_type(data) == "image/heif"

def test_detect_media_type_unsupported():
    """Test unsupported image type detection."""
    invalid_data = b"invalid image data"
    with pytest.raises(ValueError, match="Unsupported image type"):
        detect_media_type(invalid_data)

def test_decode_content_invalid_base64():
    """Test handling of invalid base64 data in image tag."""
    content = '<|image|>invalid base64 data<|/image|>'
    with pytest.raises(ValueError, match="Invalid base64 image data"):
        decode_content(content)

