from typing import Iterator, Type

import json
import json_repair

from ..base.response import Completion

from langprompt.output_parser.base import OutputParser, OutputType

class JSONOutputParser(OutputParser[OutputType]):
    """A parser that converts JSON responses from LLM to Python objects.

    This parser handles JSON responses and converts them to either:
    1. A dictionary (if output_class is dict)
    2. An instance of the specified output_class (must be a dataclass or similar)
    """
    def __init__(self, output_class: Type[OutputType]) -> None:
        """Initialize the JSON parser with the target output class.

        Args:
            output_class: The class type to convert JSON responses into
        """
        self.output_class = output_class

    def parse(self, completion: Completion) -> OutputType:
        """Parse the LLM completion response into the target output type.

        Args:
            completion: The Completion response from the LLM

        Returns:
            The parsed output matching the output_class type

        Raises:
            ValueError: If JSON parsing fails or type conversion fails
        """
        content = completion.content
        if content is None:
            raise ValueError("Completion content is None")

        result = json_repair.loads(content)
        # If output type is dict, return the result directly
        if self.output_class is dict:
            return result  # type: ignore
        # Otherwise, convert dict to specified output_class
        if isinstance(result, dict):
            try:
                obj = self.output_class(**result)
                return obj
            except Exception as e:
                raise ValueError(f"Failed to convert JSON to {self.output_class.__name__}")
        # If result is not a dict, raise error
        raise ValueError(f"Expected dict from JSON, got {type(result)}")


    def stream_parse(self, completion: Iterator[Completion]) -> Iterator[OutputType]:
        """Stream parsing is not supported for JSON responses."""
        raise NotImplementedError("JSONOutputParser does not support streaming")
