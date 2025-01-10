import base64
import pytest
from langprompt.prompt.base import Prompt
from langprompt.base.message import TextPart, ImagePart

class TestPrompt:
    def test_init(self):
        """Test Prompt initialization."""
        template = "<|system|>Test message<|end|>"
        prompt = Prompt(template)
        assert prompt.template == template

    def test_empty_template(self):
        """Test handling of empty template."""
        with pytest.raises(ValueError, match="Template cannot be None"):
            Prompt("").parse({})

    def test_no_valid_blocks(self):
        """Test handling of template with no valid blocks."""
        with pytest.raises(ValueError, match="Template must contain at least one valid message block"):
            Prompt("Invalid template").parse({})

    def test_parse_with_dict_input(self):
        """Test parsing template with dictionary input."""
        template = "<|system|>Hello {{ name }}!<|end|>"
        prompt = Prompt(template)
        messages = prompt.parse({"name": "World"})
        assert len(messages) == 1
        assert messages[0].role == "system"
        assert messages[0].content_str == "Hello World!"

    def test_parse_with_object_input(self):
        """Test parsing template with object input."""
        from dataclasses import dataclass

        @dataclass
        class Input:
            name: str

        template = "<|system|>Hello {{ name }}!<|end|>"
        prompt = Prompt(template)
        messages = prompt.parse(Input(name="World"))
        assert len(messages) == 1
        assert messages[0].role == "system"
        assert messages[0].content_str == "Hello World!"

    def test_parse_multiple_messages(self):
        """Test parsing template with multiple message blocks."""
        template = """
        <|system|>System message<|end|>
        <|user|>User message<|end|>
        <|assistant|>Assistant message<|end|>
        """
        prompt = Prompt(template)
        messages = prompt.parse({})
        assert len(messages) == 3
        assert [m.role for m in messages] == ["system", "user", "assistant"]
        assert [m.content_str for m in messages] == [
            "System message",
            "User message",
            "Assistant message"
        ]

    def test_parse_with_empty_content(self):
        """Test parsing template with empty content blocks."""
        template = """
        <|system|>Valid message<|end|>
        <|user|>  <|end|>
        """
        prompt = Prompt(template)
        messages = prompt.parse({})
        assert len(messages) == 1
        assert messages[0].role == "system"
        assert messages[0].content_str == "Valid message"

    def test_parse_with_image(self):
        """Test parsing template with image content."""
        jpeg_header = b"\xff\xd8\xff"
        base64_data = base64.b64encode(jpeg_header).decode()
        template = f'<|user|>Text <|image|>{base64_data}<|/image|> more text<|end|>'

        prompt = Prompt(template)
        messages = prompt.parse({})
        assert len(messages) == 1
        content = messages[0].content
        assert len(content) == 3
        assert isinstance(content[0], TextPart)
        assert isinstance(content[1], ImagePart)
        assert isinstance(content[2], TextPart)
        assert content[1].media_type == "image/jpeg"
