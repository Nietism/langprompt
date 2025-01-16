import re
from abc import ABC
from typing import Generic, List, Pattern, TypeVar, Iterator, Optional

from jinja2 import Template

from langprompt.base.content import decode_content
from langprompt.base import Completion, Message
from langprompt.output_parser import OutputParser, OutputType

__all__ = ["Prompt"]

InputType = TypeVar("InputType")


class Prompt(ABC, Generic[InputType, OutputType]):
    """Base class for creating structured chat prompts with template support.

    This class provides functionality to convert templated strings into a list of
    chat completion messages compatible with OpenAI's chat API.

    Template Format:
        <|system|>string message content<|end|>
        <|user|><|image media_type="image/jpeg"|>image_bytes_base64<|/image|>sdfsdfssdf<|end|>

    Supported roles:
        - system: For system instructions
        - user: For user messages
        - assistant: For AI assistant responses

    Supported Part Types:
        - text: For text content
        - image: For image content, with optional media_type and detail attributes

    Args:
        template (str): A template string containing one or more message blocks
        output_parser (OutputParser[OutputType]): An output parser for the prompt

    Raises:
        ValueError: If template is empty or contains no valid message blocks
        ValueError: If image data is not valid.
    """

    # Regular expression pattern for matching role-based message blocks
    ROLE_PATTERN: Pattern = re.compile(r"<\|(\w+)\|>(.+?)<\|end\|>", re.DOTALL)

    def __init__(self, template: str, output_parser: Optional[OutputParser[OutputType]] = None):
        self.template = template
        self.output_parser = output_parser

    def parse(self, input: InputType) -> List[Message]:
        """Parse the template with given input and convert it to chat messages.

        Args:
            input (InputType): Input data for template rendering. The object's
                              attributes will be used for template variables.

        Returns:
            List[ChatCompletionMessageParam]: A list of chat completion message
                                            parameters ready for API use.

        Raises:
            ValueError: If template is None or no valid message blocks are found
            ValueError: If an unsupported role is encountered in the template
        """
        if not self.template:
            raise ValueError("Template cannot be None")

        rendered = Template(self.template).render(input=input)

        messages = []
        for match in self.ROLE_PATTERN.finditer(rendered):
            role = match.group(1)
            content = match.group(2).strip()
            if not content:
                # Ignore empty content
                continue

            # Decode content
            decoded_content = decode_content(content)
            messages.append(Message(role=role, content=decoded_content))  # type: ignore

        if not messages:
            raise ValueError("Template must contain at least one valid message block")

        return messages

    def parse_output(self, completion: Completion) -> OutputType:
        if self.output_parser is None:
            raise ValueError("Output parser is not set")
        return self.output_parser.parse(completion)

    def parse_output_stream(self, completion: Iterator[Completion]) -> Iterator[OutputType]:
        if self.output_parser is None:
            raise ValueError("Output parser is not set")
        return self.output_parser.stream_parse(completion)
