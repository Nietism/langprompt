from dataclasses import dataclass

import pytest

from langprompt.prompt import Prompt


@dataclass
class EmptyInput:
    pass


class TestPrompt:
    def test_init(self):
        """Test Prompt initialization."""
        template = "<|system|>Test message<|end|>"
        prompt = Prompt(template)
        assert prompt.template == template

    def test_empty_template(self):
        """Test handling of empty template."""
        with pytest.raises(ValueError, match="Template cannot be None"):
            Prompt("").parse(EmptyInput())

    def test_no_valid_blocks(self):
        """Test handling of template with no valid blocks."""
        with pytest.raises(
            ValueError, match="Template must contain at least one valid message block"
        ):
            Prompt("Invalid template").parse(EmptyInput())

    def test_parse_with_object_input(self):
        """Test parsing template with object input."""

        @dataclass
        class Input:
            name: str

        template = "<|system|>Hello {{ input.name }}!<|end|>"
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
        messages = prompt.parse(EmptyInput())
        assert len(messages) == 3
        assert [m.role for m in messages] == ["system", "user", "assistant"]
        assert [m.content_str for m in messages] == [
            "System message",
            "User message",
            "Assistant message",
        ]

    def test_parse_with_empty_content(self):
        """Test parsing template with empty content blocks."""
        template = """
        <|system|>Valid message<|end|>
        <|user|>  <|end|>
        """
        prompt = Prompt(template)
        messages = prompt.parse(EmptyInput())
        assert len(messages) == 1
        assert messages[0].role == "system"
        assert messages[0].content_str == "Valid message"
